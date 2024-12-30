from torch import Tensor
from torch import nn
import torch


class SimplierInputNetwork(nn.Module):
    def __init__(self, output_dim: int = 128) -> None:
        super(SimplierInputNetwork, self).__init__()

        self.tiles_and_entities = nn.Sequential(
            nn.Conv2d(in_channels=(9+19), out_channels=32, kernel_size=3, padding=1),  # (batch_size, 28, 15, 15) -> (batch_size, 32, 15, 15)
            nn.ReLU(),
            nn.MaxPool2d(2),  # (batch_size, 16, 15, 15) -> (batch_size, 16, 7, 7)
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),  # (batch_size, 32, 7, 7) -> (batch_size, 32, 7, 7)
            nn.ReLU(),
            nn.MaxPool2d(2),  # (batch_size, 32, 7, 7) -> (batch_size, 32, 3, 3)
            nn.Flatten(),     # (batch_size, 32, 3, 3) -> (batch_size, 288)
            nn.Linear(288, 128),
            nn.ReLU()
        )

        self.self_data = nn.Sequential(
            nn.Linear(5, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )

        self.combined = nn.Sequential(
            nn.Linear(128 + 128, output_dim),
            nn.ReLU()
        )

    def forward(
        self,
        tile_and_entity_data: Tensor,
        self_data: Tensor
        ) -> Tensor:
        """
        Forward pass through the network.

        Args:
            tile_and_entity_data: Tensor of shape (batch_size, 15, 15, 28)
            self_data: Tensor of shape (batch_size, 5)

        Returns:
            Hidden tensor of shape (batch_size, output_dim)
        """
        x1 = tile_and_entity_data.permute(0, 3, 1, 2)
        x1 = self.tiles_and_entities(x1)
        x1 = x1.view(-1, 128)

        x2 = self.self_data(self_data)
        x2 = x2.view(-1, 128)

        x = torch.cat((x1, x2), dim=1)
        return self.combined(x)


class SimplierNetwork(nn.Module):
    def __init__(self) -> None:
        super(SimplierNetwork, self).__init__()

        self.action_dims: dict[str, int] = {
            "Move": 5,
            "AttackStyle": 3,
            "AttackTargetPos": 4,
            "AttackOrNot": 2
        }

        self.input_network = SimplierInputNetwork(output_dim=128)

        self.hidden_network = nn.Sequential(
            nn.Linear(128, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, 64),
            nn.Tanh()
        )

        # For actions with mask, max will be added before softmax
        # So, softmax is not included in these action heads
        self.action_heads = nn.ModuleDict({
            "Move": nn.Sequential(
                nn.Linear(64, self.action_dims["Move"])
            ),
            "AttackStyle": nn.Sequential(
                nn.Linear(64, self.action_dims["AttackStyle"])
            ),
            "AttackTargetPos": nn.Sequential(
                # Should return -1 to 1
                nn.Linear(64, self.action_dims["AttackTargetPos"]),
                nn.Tanh()
            ),
            "AttackOrNot": nn.Sequential(
                nn.Linear(64, self.action_dims["AttackOrNot"]),
                nn.Softmax(dim=-1)
            )
        })

        self.critic_input = SimplierInputNetwork(output_dim=64)
        self.critic = nn.Sequential(
            nn.Linear(64, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, 1)
        )

    @staticmethod
    def action_types() -> list[str]:
        return ["Move", "AttackStyle", "AttackTargetPos", "AttackOrNot"]

    @staticmethod
    def get_distributions(action_probs: dict[str, Tensor]) -> dict[str, torch.distributions.Distribution]:
        return {
            "Move": torch.distributions.Categorical(action_probs["Move"]),
            "AttackStyle": torch.distributions.Categorical(action_probs["AttackStyle"]),
            "AttackTargetPos": torch.distributions.Normal(
                action_probs["AttackTargetPos"][:, [0, 1]] * 7.5,
                (action_probs["AttackTargetPos"][:, [2, 3]] + 1) / 2 * 7.5,
                1e-5),
            "AttackOrNot": torch.distributions.Categorical(action_probs["AttackOrNot"])
        }

    def forward(
        self,
        tile_data: Tensor,
        self_data: Tensor,
        *masks: Tensor
        ) -> tuple[dict[str, Tensor], Tensor]:
        """
        Forward pass through the network.

        Args:
            tile_data: Tensor of shape (batch_size, 15, 15, 28)
            self_data: Tensor of shape (batch_size, 21)
            masks: List of masks for some action types, each of shape (batch_size, action_dim)

        Returns:
            Tuple containing:
            - Dict of action probability tensors, each of shape (batch_size, action_dim)
            - Value tensor of shape (batch_size, 1)
        """
        if len(masks) != 2:
            raise ValueError("Expected 2 action mask tensors, got " + str(len(masks)))

        action_masks = {
            "Move": masks[0],
            "AttackStyle": masks[1]
        }

        x = self.input_network(tile_data.clone(), self_data.clone())
        hidden: Tensor = self.hidden_network(x)
        action_probs = {}
        for action_type in self.action_types():
            output = self.action_heads[action_type](hidden.clone())
            if action_type in action_masks:
                # Change the 0s in the mask to -inf, and the 1s to 0
                # This way, we can add the mask to the output to zero out the invalid actions
                softmax_mask = action_masks[action_type].clone()
                softmax_mask[softmax_mask == 0] = float("-inf")
                softmax_mask[softmax_mask == 1] = 0
                output += softmax_mask
                output = nn.functional.softmax(output, dim=-1)

            action_probs[action_type] = output

        x_critic = self.critic_input(tile_data.clone(), self_data.clone())
        value = self.critic(x_critic)
        return action_probs, value
