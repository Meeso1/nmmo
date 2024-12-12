from dataclasses import dataclass
import numpy as np
import torch
from torch import Tensor
from torch import nn


class PPONetwork(nn.Module):
    def __init__(self) -> None:
        # Inputs:
        
        super(PPONetwork, self).__init__()
        
        input_dim = 4104 # Size with no market/comm/giving
        action_dims: list[int] = [5,   # Move
                                  3,   # Attack style
                                  101, # Attack target
                                  13,  # Use
                                  13]  # Destroy 
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh()
        )
        
        self.action_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(64, dim),
                nn.Softmax(dim=-1)
            ) for dim in action_dims
        ])
        
        self.critic = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
    
    def forward(self, x: Tensor) -> tuple[list[Tensor], Tensor]:
        """
        Forward pass through the network.
        
        Args:
            x: Input state tensor of shape (batch_size, input_dim)
        
        Returns:
            Tuple containing:
            - List of action probability tensors, each of shape (batch_size, action_dim)
            - Value tensor of shape (batch_size, 1)
        """
        hidden: Tensor = self.network(x.clone())
        action_probs = [net(hidden.clone()) for net in self.action_heads]
        value = self.critic(x.clone())
        return action_probs, value
    