import torch
from torch import Tensor
from torch import nn


class InputNetwork(nn.Module):
    def __init__(self, output_dim: int = 128) -> None:
        super(InputNetwork, self).__init__()

        self.id_and_tick = nn.Sequential(
            nn.Linear((6+1), 16), # 6 values for agent ID + 1 value for tick
            nn.ReLU()
        )

        self.tiles_and_entities = nn.Sequential(
            nn.Conv2d(in_channels=(21+3), out_channels=16, kernel_size=3, padding=1),  # (batch_size, 24, 15, 15) -> (batch_size, 16, 15, 15)
            nn.ReLU(),
            nn.MaxPool2d(2),  # (batch_size, 16, 15, 15) -> (batch_size, 16, 7, 7)
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),  # (batch_size, 16, 7, 7) -> (batch_size, 32, 7, 7)
            nn.ReLU(),
            nn.MaxPool2d(2),  # (batch_size, 32, 7, 7) -> (batch_size, 32, 3, 3)
            nn.Flatten(),     # (batch_size, 32, 3, 3) -> (batch_size, 288)
            nn.Linear(288, 64),
            nn.ReLU()
        )

        self.item_embedding = nn.Sequential(
            nn.Embedding(256, 32),      # (batch_size, 12, 2) -> (batch_size, 12, 2, 32)
            nn.Flatten(start_dim=2)     # (batch_size, 12, 2, 32) -> (batch_size, 12, 2*32)
        )
        self.item_network = nn.Sequential(
            nn.Linear(2*32 + 14, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.LayerNorm(32)
        )

        self.inventory = nn.Sequential(
            nn.Flatten(),
            nn.Linear(12*32, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )

        self.self_data = nn.Sequential(
            nn.Linear(21, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
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
        items_discrete: Tensor,
        items_continuous: Tensor,
        self_data: Tensor
        ) -> Tensor:
        """
        Forward pass through the network.

        Args:
            id_and_tick: Tensor of shape (batch_size, 7)
            tile_and_entity_data: Tensor of shape (batch_size, 15, 15, 24)
            items_discrete: Tensor of shape (batch_size, 12, 2) (int64)
            items_continuous: Tensor of shape (batch_size, 12, 14)
            self_data: Tensor of shape (batch_size, 21)

        Returns:
            Hidden tensor of shape (batch_size, output_dim)
        """
        x1 = self.id_and_tick(id_and_tick)
        x1 = x1.view(-1, 16)

        x2 = tile_and_entity_data.permute(0, 3, 1, 2)
        x2 = self.tiles_and_entities(x2)
        x2 = x2.view(-1, 64)

        x3 = self.item_embedding(items_discrete.clip(0, 255))
        x3 = torch.cat((x3, items_continuous), dim=2)
        x3 = self.item_network(x3)        
        x3 = self.inventory(x3)
        x3 = x3.view(-1, 64)

        x4 = self.self_data(self_data)
        x4 = x4.view(-1, 64)

        x = torch.cat((x1, x2, x3, x4), dim=1)
        return self.combined(x)
