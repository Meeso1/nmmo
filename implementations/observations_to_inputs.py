import torch
from implementations.Observations import Observations, SingleEntity
from torch import Tensor
import numpy as np
from nmmo.core.config import Config, Resource, Progression
from nmmo import material


_map_size_x = 144
_map_size_y = 144
_view_radius = 7
_view_size = 2 * _view_radius + 1


def observations_to_network_inputs(obs: Observations, device: torch.device) \
	-> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, list[Tensor]]:
	id_and_tick = torch.tensor(
		np.concatenate([
	  		_encode_id(obs.agent_id),
			np.array([
				np.log(obs.current_tick+1)
			])
		]),
		dtype=torch.float32,
		device=device
	).unsqueeze(0)

	cnn_entiy_data, self_data = _get_entity_data(obs)
	tiles = torch.tensor(
		np.concatenate([
			_encode_tiles(obs.tiles),
			cnn_entiy_data
		], axis=2),
		dtype=torch.float32,
		device=device
	).unsqueeze(0)

	continuous, discrete = _encode_inventory(obs)
	inventory_continuous = torch.tensor(
		continuous,
		dtype=torch.float32,
		device=device
	).unsqueeze(0)
	inventory_discrete = torch.tensor(
		discrete,
		dtype=torch.int64,
		device=device
	).unsqueeze(0)

	self_data = torch.tensor(
		self_data,
		dtype=torch.float32,
		device=device
	).unsqueeze(0)

	masks = [
		torch.tensor(obs.action_targets.move_direction, dtype=torch.float32, device=device).unsqueeze(0),
		torch.tensor(obs.action_targets.attack_style, dtype=torch.float32, device=device).unsqueeze(0),
		torch.tensor(obs.action_targets.use_inventory_item, dtype=torch.float32, device=device).unsqueeze(0),
		torch.tensor(obs.action_targets.destroy_inventory_item, dtype=torch.float32, device=device).unsqueeze(0)
	]

	return id_and_tick, tiles, inventory_discrete, inventory_continuous, self_data, *masks


def observations_to_inputs_simplier(obs: Observations, device: torch.device) \
	-> tuple[Tensor, Tensor, list[Tensor]]:
	cnn_entiy_data, self_data = _get_entity_data(obs, simlified=True)
	tiles = torch.tensor(
		np.concatenate([
			_encode_tiles(obs.tiles),
			cnn_entiy_data
		], axis=2),
		dtype=torch.float32,
		device=device
	).unsqueeze(0)

	self_data = torch.tensor(
		self_data,
		dtype=torch.float32,
		device=device
	).unsqueeze(0)

	masks = [
		torch.tensor(obs.action_targets.move_direction, dtype=torch.float32, device=device).unsqueeze(0),
		torch.tensor(obs.action_targets.attack_style, dtype=torch.float32, device=device).unsqueeze(0)
	]

	return tiles, self_data, *masks


def _encode_id(single_id: int) -> np.ndarray:
	"""
	Encode agent ID into some constant representations - (6,) array
	"""
	encoded = np.array([
		(single_id >> i) & 1 for i in range(6)
	])

	return encoded


def entity_to_input_array(entity: SingleEntity) -> np.ndarray:
	"""
	Get (19,) input array for the entity
	"""
	return np.array([
		1 if entity.id < 0 else 0, # is NPC
		1 if entity.id > 0 else 0, # is player
		entity.npc_type,
		entity.damage / 100,
		np.log(entity.time_alive + 1),
		entity.freeze / 3,
		entity.item_level / Progression.PROGRESSION_LEVEL_MAX,
		np.log(entity.latest_combat_tick + 1),
		entity.health / Config.PLAYER_BASE_HEALTH,
		entity.food / Resource.RESOURCE_BASE,
		entity.water / Resource.RESOURCE_BASE,
		entity.melee_level / Progression.PROGRESSION_LEVEL_MAX,
		entity.range_level / Progression.PROGRESSION_LEVEL_MAX,
		entity.mage_level / Progression.PROGRESSION_LEVEL_MAX,
		entity.fishing_level / Progression.PROGRESSION_LEVEL_MAX,
		entity.herbalism_level / Progression.PROGRESSION_LEVEL_MAX,
		entity.prospecting_level / Progression.PROGRESSION_LEVEL_MAX,
		entity.carving_level / Progression.PROGRESSION_LEVEL_MAX,
		entity.alchemy_level / Progression.PROGRESSION_LEVEL_MAX
	])
  
  
def entity_to_input_array_simplier(entity: SingleEntity) -> np.ndarray:
	"""
	Get (7,) input array for the entity
	"""
	return np.array([
		1 if entity.id < 0 else 0, # is NPC
		1 if entity.id > 0 else 0, # is player
		entity.npc_type,
		entity.item_level / Progression.PROGRESSION_LEVEL_MAX,
		entity.health / Config.PLAYER_BASE_HEALTH,
		entity.food / Resource.RESOURCE_BASE,
		entity.water / Resource.RESOURCE_BASE
	])


def _get_entity_data(obs: Observations, simlified: bool = False) -> tuple[np.ndarray, np.ndarray]:
	"""
	Get entity data for the CNN (of shape (15, 15, 21)) and self data (of shape (21,))
	If simplified=True, CNN output will be of shape (15, 15, 9) and self data will be of shape (5,)
	"""
	me = obs.entities.get_by_id(obs.agent_id)
	if me is None:
		print(f"[{obs.current_tick:4d}] Agent {obs.agent_id} not found in entities ({np.sum(obs.entities.id != 0)} entities seen)")
		return np.zeros((_view_size, _view_size, 21)), np.zeros((21,))

	other_entites_with_indexes = [(i, obs.entities.get_by_index(i))
                                  for i, a_id in enumerate(obs.entities.id)
				   				  if a_id != obs.agent_id and a_id != 0]

	cnn_data = np.zeros((_view_size, _view_size, 21)) if not simlified else np.zeros((_view_size, _view_size, 9))
	for idx, entity in other_entites_with_indexes:
		cnn_data[entity.row - me.row + _view_radius, entity.col - me.col + _view_radius, :] = \
			np.concatenate([
				entity_to_input_array(entity) if not simlified else entity_to_input_array_simplier(entity),
				np.array([
					obs.action_targets.attack_target[idx],
					np.sqrt((entity.row - me.row) ** 2 + (entity.col - me.col) ** 2) / np.sqrt(_view_radius ** 2 + _view_radius ** 2),
			  	]),
			])

	my_data_input = entity_to_input_array(me) if not simlified \
      				else entity_to_input_array_simplier(me)[-3:] # Only keep health, food, water
	return cnn_data, np.concatenate([
			my_data_input,
			np.array([
				me.col / _map_size_x,
				me.row / _map_size_y
			])
		])


def _encode_inventory(obs: Observations) -> tuple[np.ndarray, np.ndarray]:
	discrete_idxs = [1, 14]
	discrete_offset = np.array([2, 0])
	continuous_idxs = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15]
	continuous_scale = np.array(
		[
			1 / 10,
			1 / 10,
			1 / 10,
			1 / 100,
			1 / 100,
			1 / 100,
			1 / 40,
			1 / 40,
			1 / 40,
			1 / 100,
			1 / 100,
			1 / 100,
		]
	)
 
	inventory = obs.inventory
	discrete = inventory[:, discrete_idxs] + discrete_offset
 
	continuous = np.concatenate([
		inventory[:, continuous_idxs] * continuous_scale,
		obs.action_targets.use_inventory_item[:12].reshape(12, 1),
		obs.action_targets.destroy_inventory_item[:12].reshape(12, 1)
	], axis=1)

	return continuous, discrete # (12, 14), (12, 2)


def _encode_tiles(tiles: np.ndarray) -> np.ndarray:
	"""
	Encode tiles into a (15, 15, 19) array
	"""
	types = tiles[:, 2].reshape(_view_size, _view_size)
	result = np.zeros((_view_size, _view_size, 19))
	for i in range(16):
		result[:, :, i] = 1*(types == i)
  
	for x in range(15):
		for y in range(15):
			result[x, y, 16] = 1 if types[x, y] in material.Impassible else 0
			result[x, y, 17] = 1 if types[x, y] in material.Habitable else 0
			result[x, y, 18] = 1 if types[x, y] in material.Harvestable else 0

	return result
