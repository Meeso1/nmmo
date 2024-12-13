import torch
from torch import Tensor
from torch import nn
from implementations.AttentionNetwork import AttentionNetwork


class InputNetwork(nn.Module):
    def __init__(self, output_dim: int = 128) -> None:
        super(InputNetwork, self).__init__()
        
        self.id_and_tick = nn.Sequential(
            # TODO: Encode time maybe?
            nn.Linear(2, 16),
            nn.ReLU()
        )
        
        self.tiles = nn.Sequential(
            nn.Conv2d(in_channels=17, out_channels=16, kernel_size=3, padding=1),  # (batch_size, 17, 15, 15) -> (batch_size, 16, 15, 15)
            nn.ReLU(),
            nn.MaxPool2d(2),  # (batch_size, 16, 15, 15) -> (batch_size, 16, 7, 7)
            nn.Flatten(),     # (batch_size, 16, 7, 7) -> (batch_size, 784)
            nn.Linear(784, 64),
            nn.ReLU()
        )
        
        self.inventory_and_masks = nn.Sequential(
            nn.Linear(12*(16 + 1 + 1), 64), # 12 slots x (16 items + <use mask> + <destroy mask>)
            nn.ReLU()
        )
        
        self.entities = AttentionNetwork(31 + 1, output_dim=128)
        
        self.combined = nn.Sequential(
            nn.Linear(16 + 64 + 64 + 128, output_dim),
            nn.ReLU()
        )
    
    def forward(
        self, 
        id_and_tick: Tensor, 
        tile_data: Tensor, 
        inventory_data: Tensor, 
        entity_data: Tensor
        ) -> Tensor:
        """
        Forward pass through the network.
        
        Args:
            id_and_tick: Tensor of shape (batch_size, 2)
            tile_data: Tensor of shape (batch_size, 15, 15, 17)
            inventory_data: Tensor of shape (batch_size, 12, 18)
            entity_data: Tensor of shape (batch_size, 100, 31)
        
        Returns:
            Hidden tensor of shape (batch_size, output_dim)
        """
        x1 = self.id_and_tick(id_and_tick)
        x1 = x1.view(-1, 16)
        
        x2 = tile_data.permute(0, 2, 1)
        x2 = self.tiles(x2)
        x2 = x2.view(-1, 64)
        
        x3 = self.inventory_and_masks(inventory_data)
        x3 = x3.view(-1, 64)
        
        x4 = self.entities(entity_data)
        x4 = x4.view(-1, 128)

        x = torch.cat((x1, x2, x3, x4), dim=1)
        return self.combined(x)
    