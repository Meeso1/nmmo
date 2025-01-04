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
class SingleEntity:
	id: int
	npc_type: int
	row: int
	col: int
	damage: int
	time_alive: int
	freeze: int
	item_level: int
	attacker_id: int
	latest_combat_tick: int
	message: int
	gold: int
	health: int
	food: int
	water: int
	melee_level: int
	melee_exp: int
	range_level: int
	range_exp: int
	mage_level: int
	mage_exp: int
	fishing_level: int
	fishing_exp: int
	herbalism_level: int
	herbalism_exp: int
	prospecting_level: int
	prospecting_exp: int
	carving_level: int
	carving_exp: int
	alchemy_level: int
	alchemy_exp: int


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
    
    def get_by_index(self, idx: int) -> SingleEntity:
        return SingleEntity(
			id=self.id[idx],
			npc_type=self.npc_type[idx],
			row=self.row[idx],
			col=self.col[idx],
			damage=self.damage[idx],
			time_alive=self.time_alive[idx],
			freeze=self.freeze[idx],
			item_level=self.item_level[idx],
			attacker_id=self.attacker_id[idx],
			latest_combat_tick=self.latest_combat_tick[idx],
			message=self.message[idx],
			gold=self.gold[idx],
			health=self.health[idx],
			food=self.food[idx],
			water=self.water[idx],
			melee_level=self.melee_level[idx],
			melee_exp=self.melee_exp[idx],
			range_level=self.range_level[idx],
			range_exp=self.range_exp[idx],
			mage_level=self.mage_level[idx],
			mage_exp=self.mage_exp[idx],
			fishing_level=self.fishing_level[idx],
			fishing_exp=self.fishing_exp[idx],
			herbalism_level=self.herbalism_level[idx],
			herbalism_exp=self.herbalism_exp[idx],
			prospecting_level=self.prospecting_level[idx],
			prospecting_exp=self.prospecting_exp[idx],
			carving_level=self.carving_level[idx],
			carving_exp=self.carving_exp[idx],
			alchemy_level=self.alchemy_level[idx],
			alchemy_exp=self.alchemy_exp[idx]
		)
    
    def get_by_id(self, entity_id: int) -> SingleEntity | None:
        idx = np.where(self.id == entity_id)[0]
        if len(idx) == 0:
            return None
        return self.get_by_index(idx[0])


@dataclass(frozen=True)
class Observations:
    agent_id: int
    current_tick: int
    tiles: np.ndarray # 225x3 (15x15, flattened, 3 possible values, one-hot encoded)
    inventory: np.ndarray # 12x16 (12 slots, 16 possible items, one-hot encoded)
    entities: EntityData
    action_targets: ActionTargets
