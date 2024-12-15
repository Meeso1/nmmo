from dataclasses import dataclass
import numpy as np
from nmmo.entity.entity import EntityState


@dataclass(frozen=True)
class ActionTargets:
    move_direction: np.ndarray # 5 (4 directions + noop, one-hot encoded)
    attack_style: np.ndarray # 3 (one-hot encoded)
    attack_target: np.ndarray # 101 (100 entities + noop, one-hot encoded)
    use_inventory_item: np.ndarray # 13 (12 items + noop, one-hot encoded)
    destroy_inventory_item: np.ndarray # 13 (12 items + noop, one-hot encoded)


@dataclass(frozen=True)
class EntityData:
    # All arrays: (100,) int
    id: np.ndarray
    npc_type: np.ndarray
    row: np.ndarray
    col: np.ndarray
    damage: np.ndarray
    time_alive: np.ndarray
    freeze: np.ndarray
    item_level: np.ndarray
    attacker_id: np.ndarray
    latest_combat_tick: np.ndarray
    message: np.ndarray
    gold: np.ndarray
    health: np.ndarray
    food: np.ndarray
    water: np.ndarray
    melee_level: np.ndarray
    melee_exp: np.ndarray
    range_level: np.ndarray
    range_exp: np.ndarray
    mage_level: np.ndarray
    mage_exp: np.ndarray
    fishing_level: np.ndarray
    fishing_exp: np.ndarray
    herbalism_level: np.ndarray
    herbalism_exp: np.ndarray
    prospecting_level: np.ndarray
    prospecting_exp: np.ndarray
    carving_level: np.ndarray
    carving_exp: np.ndarray
    alchemy_level: np.ndarray
    alchemy_exp: np.ndarray


@dataclass(frozen=True)
class Observations:
    agent_id: int
    current_tick: int
    tiles: np.ndarray # 225x3 (15x15, flattened, 3 possible values, one-hot encoded)
    inventory: np.ndarray # 12x16 (12 slots, 16 possible items, one-hot encoded)
    entities: EntityData
    action_targets: ActionTargets


def to_observations(obs: dict[str]) -> Observations:
    return Observations(
        agent_id=obs["AgentId"],
        current_tick=obs["CurrentTick"],
        inventory=obs["Inventory"],
        tiles=obs["Tile"],
        entities=EntityData(
            **{feature: obs["Entity"][:, idx] 
            for feature, idx in EntityState.State.attr_name_to_col.items()}
        ),
        action_targets=ActionTargets(
            move_direction=obs["ActionTargets"]["Move"]["Direction"],
            attack_style=obs["ActionTargets"]["Attack"]["Style"],
            attack_target=obs["ActionTargets"]["Attack"]["Target"],
            use_inventory_item=obs["ActionTargets"]["Use"]["InventoryItem"],
            destroy_inventory_item=obs["ActionTargets"]["Destroy"]["InventoryItem"]
        )
    )
