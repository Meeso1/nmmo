"""
Defines networks needed for a SAC agent, leveraging logic from SimplierInputNetwork
to handle observation inputs and generate latent representations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from implementations.SimplierNetwork import SimplierInputNetwork

# ...existing code (e.g., reusing ideas from SimplierInputNetwork)...


class SACPolicyNetwork(nn.Module):
    """
    Outputs action distribution parameters (e.g. for continuous actions: mean, log_std).
    For discrete or hybrid tasks, adapt appropriately.
    """
    def __init__(self, latent_dim: int, action_dim: int):
        super().__init__()
        # Use SimplierInputNetwork as the first step if you have observation inputs
        self.obs_encoder = SimplierInputNetwork(output_dim=latent_dim)
        self.mean_fc = nn.Linear(latent_dim, action_dim)
        self.log_std_fc = nn.Linear(latent_dim, action_dim)

    def forward(self, tile_data: torch.Tensor, self_data: torch.Tensor):
        """
        Args:
            tile_data: shape [batch_size, ...]
            self_data: shape [batch_size, ...]

        Returns:
            (mean, log_std) each shape [batch_size, action_dim]
        """
        # First, encode observations with SimplierInputNetwork
        latent = self.obs_encoder(tile_data, self_data)
        mean = self.mean_fc(latent)
        log_std = self.log_std_fc(latent).clamp(-10, 2)
        return mean, log_std


class SACCriticNetwork(nn.Module):
    """
    Q-network that estimates Q-value for (state_latent, action) pairs.
    """
    def __init__(self, latent_dim: int, action_dim: int):
        super().__init__()
        self.obs_encoder = SimplierInputNetwork(output_dim=latent_dim)
        self.fc1 = nn.Linear(latent_dim + action_dim, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, tile_data: torch.Tensor, self_data: torch.Tensor, action: torch.Tensor):
        """
        Args:
            tile_data: shape [batch_size, ...]
            self_data: shape [batch_size, ...]
            action: shape [batch_size, action_dim]

        Returns:
            Q-value of shape [batch_size, 1]
        """
        # Encode observations, then concatenate with action
        latent = self.obs_encoder(tile_data, self_data)
        x = torch.cat([latent, action], dim=-1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)