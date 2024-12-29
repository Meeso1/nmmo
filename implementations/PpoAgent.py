from abc import ABC, abstractmethod
from typing import Any
import torch
from torch import optim
import numpy as np
from torch import Tensor

from implementations.PpoNetwork import PPONetwork
from implementations.Observations import Observations
from implementations.to_observations import to_observations
from implementations.observations_to_inputs import observations_to_network_inputs
from implementations.jar import Jar


class AgentBase(ABC):
    @abstractmethod
    def get_actions(self, states: dict[int, Observations]) \
        -> dict[int, tuple[dict[str, dict[str, int]], dict[str, Tensor], dict[str, Tensor]]]:
        pass

    def update(
        self,
        states: dict[int, list[Observations]],
        actions: dict[int, list[dict[str, Tensor]]],
        rewards: dict[int, list[float]],
        log_probs: dict[int, list[dict[str, Tensor]]],
        dones: dict[int, list[bool]]
    ) -> tuple[list[float], list[float], list[float]]:
        pass

    def save(self, name: str | None = None) -> None:
        pass

    def get_observations_from_state(self, obs: dict[str, Any]) -> Any:
        return to_observations(obs)

class PPOAgent(AgentBase):
    def __init__(
        self,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        epsilon: float = 0.2,
        epochs: int = 10,
        batch_size: int = 64
    ) -> None:
        self.gamma = gamma
        self.epsilon = epsilon
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.network = PPONetwork().to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)

    @staticmethod
    def _get_attack_target_index(x: float, y: float, state: Observations) -> int:
        if np.all(state.entities.id != state.agent_id):
            return 100

        attackable_idxs = [idx for idx, agent_id in enumerate(state.entities.id)
                           if agent_id != 0
                           and agent_id != state.agent_id
                           and state.action_targets.attack_target[idx] == 1]

        if len(attackable_idxs) == 0:
            return 100

        me_idx = np.where(state.entities.id == state.agent_id)[0][0]
        distances = [(state.entities.col[idx] - state.entities.col[me_idx] - x)**2
                     + (state.entities.row[idx] - state.entities.row[me_idx] - y)**2
                     for idx in attackable_idxs]
        return attackable_idxs[np.argmin(distances)]

    @staticmethod
    def _sampled_outputs_to_action_dict(sampled_outputs: dict[str, Tensor], state: Observations) -> dict[str, dict[str, int]]:
        attack_target_pos = sampled_outputs["AttackTargetPos"][0].cpu().numpy()
        return {
            "Move": {
                "Direction": sampled_outputs["Move"].item()
            },
            "Attack": {
                "Style": sampled_outputs["AttackStyle"].item(),
                "Target": 100 if sampled_outputs["AttackOrNot"].item() == 0
                else PPOAgent._get_attack_target_index(attack_target_pos[0], attack_target_pos[1], state)
            },
            "Use": {
                "InventoryItem": sampled_outputs["Use"].item()
            },
            "Destroy": {
                "InventoryItem": sampled_outputs["Destroy"].item()
            }
        }

    def get_actions(
        self,
        states: dict[int, Observations]
    ) -> dict[int, tuple[dict[str, dict[str, int]], dict[str, Tensor], dict[str, Tensor]]]:
        actions = {}
        
        self.network.eval()

        for agent_id, observations in states.items():
            inputs = observations_to_network_inputs(observations, self.device)
            action_probs, _ = self.network(*inputs)
            distributions = PPONetwork.get_distributions(action_probs)
            agent_actions = {name: dist.sample() for name, dist in distributions.items()}
            log_probs = {name: distributions[name].log_prob(action) for name, action in agent_actions.items()}

            # Get a single log prob value for each sample
            for name, log_prob in log_probs.items():
                if log_prob.dim() > 1:
                    log_probs[name] = log_prob.sum(dim=-1)

            actions[agent_id] = (
                PPOAgent._sampled_outputs_to_action_dict(agent_actions, observations),
                agent_actions,
                log_probs
            )
        return actions

    def update(
        self,
        states: dict[int, list[Observations]],
        actions: dict[int, list[dict[str, Tensor]]],
        rewards: dict[int, list[float]],
        log_probs: dict[int, list[dict[str, Tensor]]],
        dones: dict[int, list[bool]]
    ) -> tuple[list[float], list[float], list[float]]:
        all_returns: list[float] = []
        all_states: list[Observations] = []
        all_actions: dict[str, list[Tensor]] = {type: [] for type in PPONetwork.action_types()}
        all_old_log_probs: dict[str, list[Tensor]] = {type: [] for type in PPONetwork.action_types()}
        
        self.network.train()

        for agent_id in states.keys():
            R = 0.0
            agent_returns = []
            for r, d in zip(reversed(rewards[agent_id]), reversed(dones[agent_id])):
                R = r + self.gamma * R * (1-d)
                agent_returns.insert(0, R)

            all_returns.extend(agent_returns)
            all_states.extend(states[agent_id])

            for type in PPONetwork.action_types():
                all_actions[type].extend([actions_in_step[type] for actions_in_step in actions[agent_id]])
                all_old_log_probs[type].extend([log_probs_in_step[type] for log_probs_in_step in log_probs[agent_id]])

        network_inputs = [observations_to_network_inputs(obs, self.device) for obs in all_states]
        stacked_inputs = [torch.cat([inputs[i] for inputs in network_inputs], dim=0) for i in range(len(network_inputs[0]))]

        returns_tensor = torch.FloatTensor(all_returns).to(self.device)
        action_tensors = {name: torch.cat(acts, dim=0).to(self.device) for name, acts in all_actions.items()}
        old_log_prob_tensors = {name: torch.cat(lps, dim=0).to(self.device) for name, lps in all_old_log_probs.items()}

        actor_losses = []
        critic_losses = []
        total_losses = []

        for _ in range(self.epochs):
            epoch_actor_losses = torch.zeros(len(network_inputs), device=self.device)
            epoch_critic_losses = torch.zeros(len(network_inputs), device=self.device)

            indices = np.arange(len(network_inputs))
            np.random.shuffle(indices)

            for i in range(0, len(indices), self.batch_size):
                batch_indices = indices[i:i+self.batch_size]
                batch_inputs = [input[batch_indices] for input in stacked_inputs]
                batch_returns_tensor = returns_tensor[batch_indices]
                batch_action_tensors = {name: acts[batch_indices] for name, acts in action_tensors.items()}
                batch_old_log_prob_tensors = {name: lps[batch_indices] for name, lps in old_log_prob_tensors.items()}

                new_action_probs: list[Tensor]
                values: Tensor
                new_action_probs, values = self.network(*batch_inputs)
                distributions = PPONetwork.get_distributions(new_action_probs)
                advantages = (batch_returns_tensor - values).detach()

                actor_loss = 0
                for type in PPONetwork.action_types():
                    dist = distributions[type]
                    actions_i = batch_action_tensors[type]
                    old_lp = batch_old_log_prob_tensors[type].detach()

                    new_log_probs_i = dist.log_prob(actions_i)
                    if new_log_probs_i.dim() > 1:
                        new_log_probs_i = new_log_probs_i.sum(dim=-1)

                    ratio = torch.exp(new_log_probs_i - old_lp)

                    surr1 = ratio * advantages
                    surr2 = torch.clamp(ratio, 1-self.epsilon, 1+self.epsilon) * advantages
                    actor_loss += -torch.min(surr1, surr2).mean()
                    
                    # Save losses for history
                    epoch_actor_losses[batch_indices] = -torch.min(surr1, surr2).mean(dim=0)
                    epoch_critic_losses[batch_indices] = 0.5 * (values.squeeze(dim=-1) - batch_returns_tensor.detach()).pow(2)

                self.optimizer.zero_grad()
                actor_loss.backward()
                self.optimizer.step()

                critic_loss: Tensor = 0.5 * torch.nn.MSELoss()(values.squeeze(dim=-1), batch_returns_tensor.detach())
                self.optimizer.zero_grad()
                critic_loss.backward()
                self.optimizer.step()

            actor_losses.append(epoch_actor_losses.mean().item())
            critic_losses.append(epoch_critic_losses.mean().item())
            total_losses.append(epoch_actor_losses.mean().item() + epoch_critic_losses.mean().item())
            
        return actor_losses, critic_losses, total_losses

    def save(self, name: str | None = None) -> None:
        agent_name = name if name is not None else "PPOAgent"
        jar = Jar("saved_agents")

        constructor_params = {
            "learning_rate": self.learning_rate,
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "epochs": self.epochs,
            "batch_size": self.batch_size
        }

        jar.add(f"{agent_name}-params", constructor_params)
        jar.add(f"{agent_name}-state", self.network.state_dict(), kind="torch")

    @staticmethod
    def load(name: str | None = None) -> 'PPOAgent':
        agent_name = name if name is not None else "PPOAgent"
        jar = Jar("saved_agents")

        if f"{agent_name}-state" not in jar:
            raise FileNotFoundError(f"Could not find agent state for '{agent_name}'")
        if f"{agent_name}-params" not in jar:
            raise FileNotFoundError(f"Could not find agent parameters for '{agent_name}'")

        constructor_params = jar.get(f"{agent_name}-params")
        agent = PPOAgent(**constructor_params)

        state_dict = jar.get(f"{agent_name}-state")
        agent.network.load_state_dict(state_dict)
        agent.network.eval()
        return agent
