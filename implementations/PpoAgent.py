from dataclasses import dataclass
import torch
from torch import optim
import numpy as np
from torch import Tensor

from implementations.PpoNetwork import PPONetwork
from implementations.Observations import Observations


def indexes_to_action_dict(indexes: list[int]) -> dict[str, dict[str, int]]:
    return {
        "Move": {
            "Direction": indexes[0]
        },
        "Attack": {
            "Style": indexes[1],
            "Target": indexes[2]
        },
        "Use": {
            "InventoryItem": indexes[3]
        },
        "Destroy": {
            "InventoryItem": indexes[4]
        }
    }


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
        self.action_types: list[str] = ["Move",
                                        "Attack style",
                                        "Attack target",
                                        "Use",
                                        "Destroy"]
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.network = PPONetwork().to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)

        self.actor_losses = []
        self.critic_losses = []
        self.total_losses = []

    def get_actions(
        self,
        states: dict[int, Observations]
    ) -> dict[int, tuple[dict[str, dict[str, int]], list[int], list[Tensor]]]:
        actions = {}
        
        for agent_id, observations in states.items():
            inputs = self._observations_to_network_inputs(observations)
            action_probs, _ = self.network(*inputs)
            
            distributions = [torch.distributions.Categorical(probs) for probs in action_probs]
            agent_actions = [dist.sample() for dist in distributions]
            log_probs = [dist.log_prob(action) for dist, action in zip(distributions, agent_actions)]
            
            actions[agent_id] = (
                indexes_to_action_dict([a.item() for a in agent_actions]), 
                [a.item() for a in agent_actions], 
                log_probs
            )
        return actions
    
    def _observations_to_network_inputs(self, obs: Observations) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        id_and_tick = torch.tensor(
            np.array([obs.agent_id, obs.current_tick]),
            dtype=torch.float32,
            device=self.device
        ).unsqueeze(0)

        tiles = torch.tensor(
            obs.tiles,
            dtype=torch.float32,
            device=self.device
        ).unsqueeze(0)
        
        inventory = torch.tensor(
            np.concatenate([
                obs.inventory,
                obs.action_targets.use_inventory_item[:12].reshape(12, 1),
                obs.action_targets.destroy_inventory_item[:12].reshape(12, 1)
            ], axis=1).ravel(),
            dtype=torch.float32,
            device=self.device
        ).unsqueeze(0)

        entities = torch.tensor(
            np.concatenate([
                obs.entities,
                obs.action_targets.attack_target[:100].reshape(100, 1)
            ], axis=1),
            dtype=torch.float32,
            device=self.device
        ).unsqueeze(0)
        
        return id_and_tick, tiles, inventory, entities

    def update(
        self,
        states: dict[int, list[Observations]],
        actions: dict[int, list[list[int]]],
        rewards: dict[int, list[float]],
        log_probs: dict[int, list[list[Tensor]]],
        dones: dict[int, list[bool]]
    ) -> None:
        all_returns: list[float] = []
        all_states: list[Observations] = []
        all_actions: dict[str, list[int]] = {type: [] for type in self.action_types}
        all_old_log_probs: dict[str, list[Tensor]] = {type: [] for type in self.action_types}

        for agent_id in states.keys():
            R = 0.0
            agent_returns = []
            for r, d in zip(reversed(rewards[agent_id]), reversed(dones[agent_id])):
                R = r + self.gamma * R * (1-d)
                agent_returns.insert(0, R)
            
            all_returns.extend(agent_returns)
            all_states.extend(states[agent_id])
            
            for i, type in enumerate(self.action_types):
                all_actions[type].extend([step[i] for step in actions[agent_id]])
                all_old_log_probs[type].extend([step[i] for step in log_probs[agent_id]])
        
        network_inputs = [self._observations_to_network_inputs(obs) for obs in all_states]
        stacked_inputs = [torch.cat([inputs[i] for inputs in network_inputs], dim=0) for i in range(len(network_inputs[0]))]
    
        returns_tensor = torch.FloatTensor(all_returns).to(self.device)
        action_tensors = [torch.LongTensor(acts).to(self.device) for acts in all_actions.values()]
        old_log_prob_tensors = [torch.stack(lps).to(self.device) for lps in all_old_log_probs.values()]

        for _ in range(self.epochs):
            for i in range(0, len(network_inputs), self.batch_size):
                indices = range(i, min(i+self.batch_size, len(network_inputs)))
                batch_inputs = [input[indices] for input in stacked_inputs]
                batch_returns_tensor = returns_tensor[indices]
                batch_action_tensors = [acts[indices] for acts in action_tensors]
                batch_old_log_prob_tensors = [lps[indices] for lps in old_log_prob_tensors]

                new_action_probs: list[Tensor]
                values: Tensor
                new_action_probs, values = self.network(*[input.clone() for input in batch_inputs])
                advantages = batch_returns_tensor.detach() - values.detach()

                actor_loss = 0
                for probs, actions_i, old_lp in zip(new_action_probs, batch_action_tensors, batch_old_log_prob_tensors):
                    dist = torch.distributions.Categorical(probs)
                    new_log_probs_i = dist.log_prob(actions_i.detach())
                    ratio = torch.exp(new_log_probs_i - old_lp.detach())

                    surr1 = ratio * advantages.detach()
                    surr2 = torch.clamp(ratio, 1-self.epsilon, 1+self.epsilon) * advantages.detach()
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
