from dataclasses import dataclass
import torch
from torch import optim
import numpy as np
from torch import Tensor

from implementations.PpoNetwork import PPONetwork
from implementations.Observations import Observations, EntityData
from implementations.observations_to_inputs import observations_to_network_inputs


def dict_to_vector(dictionary: dict[str]) -> np.ndarray:
    segments = []
    for v in dictionary.values():
        if isinstance(v, dict):
            segment = dict_to_vector(v)
        else:
            segment = np.array(v).ravel()
        segments.append(segment)

    return np.concatenate(segments)
  

class PPOAgent:
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
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.network = PPONetwork().to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)

        self.actor_losses = []
        self.critic_losses = []
        self.total_losses = []


    @staticmethod
    def get_attack_target_index(output: np.ndarray, state: Observations) -> int:
        attackable_idxs = [idx for idx, agent_id in enumerate(state.entities.id)
                           if agent_id != 0 
                           and agent_id != state.agent_id 
                           and state.action_targets.attack_target[idx] == 1]
        # TODO: Continue
        pass

    @staticmethod
    def sampled_outputs_to_action_dict(sampled_outputs: dict[str, int], state: Observations) -> dict[str, dict[str, int]]:
        return {
            "Move": {
                "Direction": sampled_outputs["Move"]
            },
            "Attack": {
                "Style": sampled_outputs["AttackStyle"],
                "Target": 100 if sampled_outputs["AttackOrNot"] == 0 
                else PPOAgent.get_attack_target_index(sampled_outputs["AttackTargetPos"], state)
            },
            "Use": {
                "InventoryItem": sampled_outputs["Use"]
            },
            "Destroy": {
                "InventoryItem": sampled_outputs["Destroy"]
            }
        }

    def get_actions(
        self,
        states: dict[int, Observations]
    ) -> dict[int, tuple[dict[str, dict[str, int]], list[int], list[Tensor]]]:
        actions = {}
        
        for agent_id, observations in states.items():
            inputs = observations_to_network_inputs(observations, self.device)
            action_probs, _ = self.network(*inputs)
            distributions = PPONetwork.get_distributions(action_probs)
            agent_actions = {name: dist.sample() for name, dist in distributions.items()}
            log_probs = {name: distributions[name].log_prob(action) for name, action in agent_actions.items()}
            
            actions[agent_id] = (
                PPOAgent.sampled_outputs_to_action_dict({n: a.item() for n, a in agent_actions.items()}, observations), 
                {n: a.item() for n, a in agent_actions.items()}, 
                log_probs
            )
        return actions

    def update(
        self,
        states: dict[int, list[Observations]],
        actions: dict[int, list[dict[str, int]]],
        rewards: dict[int, list[float]],
        log_probs: dict[int, list[dict[str, Tensor]]],
        dones: dict[int, list[bool]]
    ) -> None:
        all_returns: list[float] = []
        all_states: list[Observations] = []
        all_actions: dict[str, list[int]] = {type: [] for type in PPONetwork.action_types()}
        all_old_log_probs: dict[str, list[Tensor]] = {type: [] for type in PPONetwork.action_types()}

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
        action_tensors = {name: torch.LongTensor(acts).to(self.device) for name, acts in all_actions.items()}
        old_log_prob_tensors = {name: torch.stack(lps).to(self.device) for name, lps in all_old_log_probs.items()}

        for _ in range(self.epochs):
            for i in range(0, len(network_inputs), self.batch_size):
                indices = range(i, min(i+self.batch_size, len(network_inputs)))
                batch_inputs = [input[indices] for input in stacked_inputs]
                batch_returns_tensor = returns_tensor[indices]
                batch_action_tensors = {name: acts[indices] for name, acts in action_tensors.items()}
                batch_old_log_prob_tensors = {name: lps[indices] for name, lps in old_log_prob_tensors.items()}

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
                    ratio = torch.exp(new_log_probs_i - old_lp)

                    surr1 = ratio * advantages
                    surr2 = torch.clamp(ratio, 1-self.epsilon, 1+self.epsilon) * advantages
                    actor_loss += -torch.min(surr1, surr2).mean()

                self.optimizer.zero_grad()
                actor_loss.backward()
                self.optimizer.step()

                critic_loss: Tensor = 0.5 * torch.nn.MSELoss()(values.squeeze(), batch_returns_tensor.detach())
                self.optimizer.zero_grad()
                critic_loss.backward()
                self.optimizer.step()

                self.actor_losses.append(actor_loss.item())
                self.critic_losses.append(critic_loss.item())
                self.total_losses.append(actor_loss.item() + critic_loss.item())
