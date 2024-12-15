import torch
from torch import Tensor
from torch import nn
from implementations.AttentionNetwork import AttentionNetwork


class InputNetwork(nn.Module):
    def __init__(self, output_dim: int = 128) -> None:
        super(InputNetwork, self).__init__()
        
        self.id_and_tick = nn.Sequential(
            nn.Linear((6+1), 16), # 6 values for agent ID + 1 value for tick
            nn.ReLU()
        )
        
        self.tiles_and_entities = nn.Sequential(
            nn.Conv2d(in_channels=(20+3), out_channels=16, kernel_size=3, padding=1),  # (batch_size, 23, 15, 15) -> (batch_size, 16, 15, 15)
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
        
        self.self_data = nn.Sequential(
            nn.Linear(21, 64),
            nn.ReLU()
        )
        
        self.combined = nn.Sequential(
            nn.Linear(16 + 64 + 64 + 64, output_dim),
            nn.ReLU()
        )
    
    def forward(
        self, 
        id_and_tick: Tensor, 
        tile_and_entity_data: Tensor, 
        inventory_data: Tensor, 
        self_data: Tensor
        ) -> Tensor:
        """
        Forward pass through the network.
        
        Args:
            id_and_tick: Tensor of shape (batch_size, 7)
            tile_and_entity_data: Tensor of shape (batch_size, 15, 15, 23)
            inventory_data: Tensor of shape (batch_size, 12, 18)
            self_data: Tensor of shape (batch_size, 21)
        
        Returns:
            Hidden tensor of shape (batch_size, output_dim)
        """
        x1 = self.id_and_tick(id_and_tick)
        x1 = x1.view(-1, 16)
        
        x2 = tile_and_entity_data.permute(0, 3, 1, 2)
        x2 = self.tiles_and_entities(x2)
        x2 = x2.view(-1, 64)
        
        x3 = self.inventory_and_masks(inventory_data)
        x3 = x3.view(-1, 64)
        
        x4 = self.self_data(self_data)
        x4 = x4.view(-1, 64)

        x = torch.cat((x1, x2, x3, x4), dim=1)
        return self.combined(x)
    