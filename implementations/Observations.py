from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class ActionTargets:
    move_direction: np.ndarray # 5 (4 directions + noop, one-hot encoded)
    attack_style: np.ndarray # 3 (one-hot encoded)
    attack_target: np.ndarray # 101 (100 entities + noop, one-hot encoded)
    use_inventory_item: np.ndarray # 13 (12 items + noop, one-hot encoded)
    destroy_inventory_item: np.ndarray # 13 (12 items + noop, one-hot encoded)


@dataclass(frozen=True)
class Observations:
    agent_id: int
    current_tick: int
    tiles: np.ndarray # 225x3 (15x15, flattened, 3 possible values, one-hot encoded)
    inventory: np.ndarray # 12x16 (12 slots, 16 possible items, one-hot encoded)
    entities: np.ndarray # 100x31 (100 entities, ???)
    # 0 - id
    # 1 - ...
    # 2 - x
    # 3 - y
    # ...
    # 12-14 - health/water/food
    action_targets: ActionTargets
    