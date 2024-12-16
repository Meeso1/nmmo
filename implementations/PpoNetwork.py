from torch import Tensor
from torch import nn
import torch
from implementations.InputNetwork import InputNetwork


class PPONetwork(nn.Module):
    def __init__(self) -> None:
        super(PPONetwork, self).__init__()

        self.action_dims: dict[str, int] = {
            "Move": 5,
            "AttackStyle": 3,
            "AttackTargetPos": 4,
            "AttackOrNot": 2,
            "Use": 13,
            "Destroy": 13
        }
        
        self.input_network = InputNetwork(output_dim=128)
        
        self.hidden_network = nn.Sequential(
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh()
        )
        
        self.action_heads = nn.ModuleDict({
            "Move": nn.Sequential(
                nn.Linear(64, self.action_dims["Move"]),
                nn.Softmax(dim=-1)
            ),
            "AttackStyle": nn.Sequential(
                nn.Linear(64, self.action_dims["AttackStyle"]),
                nn.Softmax(dim=-1)
            ),
            "AttackTargetPos": nn.Sequential(
                # Should return -1 to 1
                nn.Linear(64, self.action_dims["AttackTargetPos"]),
                nn.Tanh()
            ),
            "AttackOrNot": nn.Sequential(
                nn.Linear(64, self.action_dims["AttackOrNot"]),
                nn.Softmax(dim=-1)
            ),
            "Use": nn.Sequential(
                nn.Linear(64, self.action_dims["Use"]),
                nn.Softmax(dim=-1)
            ),
            "Destroy": nn.Sequential(
                nn.Linear(64, self.action_dims["Destroy"]),
                nn.Softmax(dim=-1)
            )
        })
        
        self.critic_input = InputNetwork(output_dim=64)
        self.critic = nn.Sequential(
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        
    @staticmethod
    def action_types() -> list[str]:
        return ["Move", "AttackStyle", "AttackTargetPos", "AttackOrNot", "Use", "Destroy"]
    
    @staticmethod
    def get_distributions(action_probs: dict[str, Tensor]) -> dict[str, torch.distributions.Distribution]:
        return {
            "Move": torch.distributions.Categorical(action_probs["Move"]),
            "AttackStyle": torch.distributions.Categorical(action_probs["AttackStyle"]),
            "AttackTargetPos": torch.distributions.Normal(
                action_probs["AttackTargetPos"][:, [0, 1]] * 7.5, 
                (action_probs["AttackTargetPos"][:, [2, 3]] + 1) / 2 * 7.5, 
                1e-5),
            "AttackOrNot": torch.distributions.Categorical(action_probs["AttackOrNot"]),
            "Use": torch.distributions.Categorical(action_probs["Use"]),
            "Destroy": torch.distributions.Categorical(action_probs["Destroy"])
        }
    
    def forward(
        self, 
        id_and_tick: Tensor, 
        tile_data: Tensor, 
        inventory_data: Tensor, 
        entity_data: Tensor
        ) -> tuple[dict[str, Tensor], Tensor]:
        """
        Forward pass through the network.
        
        Args:
            id_and_tick: Tensor of shape (batch_size, 2)
            tile_data: Tensor of shape (batch_size, 255, 3)
            inventory_data: Tensor of shape (batch_size, 12, 18)
            entity_data: Tensor of shape (batch_size, 100, 32)
        
        Returns:
            Tuple containing:
            - Dict of action probability tensors, each of shape (batch_size, action_dim)
            - Value tensor of shape (batch_size, 1)
        """
        
        x = self.input_network(id_and_tick.clone(), tile_data.clone(), inventory_data.clone(), entity_data.clone())
        hidden = self.hidden_network(x)
        action_probs = {type: self.action_heads[type](hidden.clone()) for type in PPONetwork.action_types()}
        
        x_critic = self.critic_input(id_and_tick.clone(), tile_data.clone(), inventory_data.clone(), entity_data.clone())
        value = self.critic(x_critic)
        return action_probs, value
