from torch import Tensor
from torch.distributions import Distribution
from dataclasses import dataclass


@dataclass(frozen=True)
class ActionData:
    action_dict: dict[str, dict[str, int]]
    distributions: dict[str, Distribution]
    sampled_actions: dict[str, Tensor]
    log_probs: dict[str, Tensor]
