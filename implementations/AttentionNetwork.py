import torch
from torch import Tensor, nn


class AttentionNetwork(nn.Module):
    def __init__(self, input_dim, output_dim=128, attention_dim=64):
        """
        Args:
            input_dim (int): Dimension N of each input vector
            output_dim (int): Desired output dimension (default: 128)
            attention_dim (int): Internal attention dimension (default: 64)
        """
        super().__init__()
        
        # Layer for transforming vectors before attention
        self.feature_projection = nn.Linear(input_dim, attention_dim)
        
        # Layers for computing attention scores
        self.attention_projection = nn.Linear(input_dim, attention_dim)
        self.attention_scorer = nn.Linear(attention_dim, 1)
        
        self.output_projection = nn.Linear(attention_dim, output_dim)
        
    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, N, 100)
                where N is vector dimension, 100 is max sequence length
        
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_dim)
        """
        x = x.permute(0, 2, 1)
        
        # Create mask for zero vectors
        # Sum across vector dimension and check if all zeros
        mask = (x.sum(dim=-1) != 0).float().unsqueeze(-1)  # (batch_size, 100, 1)
        
        # Compute attention scores
        attention_features = torch.tanh(self.attention_projection(x))
        scores = self.attention_scorer(attention_features)  # (batch_size, 100, 1)
        
        # Apply mask to scores
        scores = scores * mask - 1e9 * (1 - mask)
        attention_weights = nn.functional.softmax(scores, dim=1)  # (batch_size, 100, 1)
        
        # Transform input vectors
        features = self.feature_projection(x)  # (batch_size, 100, attention_dim)
        
        # Apply attention
        weighted_sum = torch.sum(features * attention_weights, dim=1)  # (batch_size, attention_dim)
        
        # Final projection to desired output dimension
        output = self.output_projection(weighted_sum)  # (batch_size, output_dim)
        
        return output
