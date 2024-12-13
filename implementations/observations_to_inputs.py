from dataclasses import dataclass
import torch
from implementations.Observations import Observations, EntityData
from torch import Tensor
import numpy as np


_max_level = 10


def observations_to_network_inputs(obs: Observations, device: torch.device) \
    -> tuple[Tensor, Tensor, Tensor, Tensor]:
	id_and_tick = torch.tensor(
		np.array([obs.agent_id, obs.current_tick]),
		dtype=torch.float32,
		device=device
	).unsqueeze(0)

	tiles = torch.tensor(
		np.concatenate([
    		obs.tiles.reshape(15, 15, 3),
			_get_cnn_entity_data(obs)
		], axis=2),
		dtype=torch.float32,
		device=device
	).unsqueeze(0)
	
	inventory = torch.tensor(
		np.concatenate([
			obs.inventory,
			obs.action_targets.use_inventory_item[:12].reshape(12, 1),
			obs.action_targets.destroy_inventory_item[:12].reshape(12, 1)
		], axis=1).ravel(),
		dtype=torch.float32,
		device=device
	).unsqueeze(0)

	entities = torch.tensor(
		np.stack([
			obs.entities.id < 0,
			obs.entities.id > 0,
			obs.entities.npc_type,
			obs.entities.damage / 100,
			np.log(obs.entities.time_alive + 1),
			obs.entities.freeze / 3,
			obs.entities.item_level / _max_level,
			np.log(obs.entities.latest_combat_tick + 1),
			obs.entities.health / 100,
			obs.entities.food / 100,
			obs.entities.water / 100,
			obs.entities.melee_level / _max_level,
			obs.entities.range_level / _max_level,
			obs.entities.mage_level	/ _max_level,
			obs.entities.fishing_level / _max_level,
			obs.entities.herbalism_level / _max_level,
			obs.entities.prospecting_level / _max_level,
			obs.entities.carving_level / _max_level,
			obs.entities.alchemy_level / _max_level,
			obs.action_targets.attack_target[:100]
		], axis=-1),
		dtype=torch.float32,
		device=device
	).unsqueeze(0)
	
	return id_and_tick, tiles, inventory, entities


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
 
	@staticmethod
	def from_entity_data(entity_data: EntityData, entity_id: int) -> 'SingleEntity':
		return SingleEntity(
			**{field: getattr(entity_data, field)[entity_id] 
      		for field in entity_data.__annotations__.keys()}
		)


def _get_cnn_entity_data(obs: Observations) -> np.ndarray:
	me_idx = np.where(obs.entities.id == obs.agent_id)[0][0]
	me = SingleEntity.from_entity_data(obs.entities, me_idx)
 
	other_entites = [SingleEntity.from_entity_data(obs.entities, i) 
                  	for i, a_id in enumerate(obs.entities.id) 
                   	if a_id != obs.agent_id and a_id != 0]
 
	cnn_data = np.zeros((15, 15, 19))
	for entity in other_entites:
		cnn_data[entity.row - me.row + 7, entity.col - me.col + 7, :] = np.array([
			1 if entity.id < 0 else 0, # is NPC
			1 if entity.id > 0 else 0, # is player
			entity.npc_type,
			entity.damage / 100,
			np.log(entity.time_alive + 1),
			entity.freeze / 3,
			entity.item_level / _max_level,
			np.log(entity.latest_combat_tick + 1),
			entity.health / 100,
			entity.food / 100,
			entity.water / 100,
			entity.melee_level / _max_level,
			entity.range_level / _max_level,
			entity.mage_level / _max_level,
			entity.fishing_level / _max_level,
			entity.herbalism_level / _max_level,
			entity.prospecting_level / _max_level,
			entity.carving_level / _max_level,
			entity.alchemy_level / _max_level
		])
  
	return cnn_data
 