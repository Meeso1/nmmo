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
            "AttackTarget": 101,
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
            name: nn.Sequential(
            nn.Linear(64, dim),
            nn.Softmax(dim=-1)
            ) for name, dim in self.action_dims.items()
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
        return ["Move", "AttackStyle", "AttackTarget", "Use", "Destroy"]
    
    @staticmethod
    def get_distributions(action_probs: dict[str, Tensor]) -> dict[str, torch.distributions.Distribution]:
        return {name: torch.distributions.Categorical(probs) for name, probs in action_probs.items()}
    
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
