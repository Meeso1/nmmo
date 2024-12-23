from dataclasses import dataclass
import torch
from implementations.Observations import Observations, EntityData
from torch import Tensor
import numpy as np
from nmmo.core.config import Config, Resource, Progression
#from nmmo import material


_map_size_x = 128
_map_size_y = 128
_view_radius = 7
_view_size = 2 * _view_radius + 1


def observations_to_network_inputs(obs: Observations, device: torch.device) \
	-> tuple[Tensor, Tensor, Tensor, Tensor]:
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


def _encode_id(single_id: int) -> np.ndarray:
	"""
	Encode agent ID into some constant representations - (6,) array
	"""
	encoded = np.array([
		(single_id >> i) & 1 for i in range(6)
	])

	return encoded


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
			id=entity_data.id[entity_id],
			npc_type=entity_data.npc_type[entity_id],
			row=entity_data.row[entity_id],
			col=entity_data.col[entity_id],
			damage=entity_data.damage[entity_id],
			time_alive=entity_data.time_alive[entity_id],
			freeze=entity_data.freeze[entity_id],
			item_level=entity_data.item_level[entity_id],
			attacker_id=entity_data.attacker_id[entity_id],
			latest_combat_tick=entity_data.latest_combat_tick[entity_id],
			message=entity_data.message[entity_id],
			gold=entity_data.gold[entity_id],
			health=entity_data.health[entity_id],
			food=entity_data.food[entity_id],
			water=entity_data.water[entity_id],
			melee_level=entity_data.melee_level[entity_id],
			melee_exp=entity_data.melee_exp[entity_id],
			range_level=entity_data.range_level[entity_id],
			range_exp=entity_data.range_exp[entity_id],
			mage_level=entity_data.mage_level[entity_id],
			mage_exp=entity_data.mage_exp[entity_id],
			fishing_level=entity_data.fishing_level[entity_id],
			fishing_exp=entity_data.fishing_exp[entity_id],
			herbalism_level=entity_data.herbalism_level[entity_id],
			herbalism_exp=entity_data.herbalism_exp[entity_id],
			prospecting_level=entity_data.prospecting_level[entity_id],
			prospecting_exp=entity_data.prospecting_exp[entity_id],
			carving_level=entity_data.carving_level[entity_id],
			carving_exp=entity_data.carving_exp[entity_id],
			alchemy_level=entity_data.alchemy_level[entity_id],
			alchemy_exp=entity_data.alchemy_exp[entity_id]
		)

	def to_input_array(self) -> np.ndarray:
		"""
		Get (19,) input array for the entity
  		"""
		return np.array([
			1 if self.id < 0 else 0, # is NPC
			1 if self.id > 0 else 0, # is player
			self.npc_type,
			self.damage / 100,
			np.log(self.time_alive + 1),
			self.freeze / 3,
			self.item_level / Progression.PROGRESSION_LEVEL_MAX,
			np.log(self.latest_combat_tick + 1),
			self.health / Config.PLAYER_BASE_HEALTH,
			self.food / Resource.RESOURCE_BASE,
			self.water / Resource.RESOURCE_BASE,
			self.melee_level / Progression.PROGRESSION_LEVEL_MAX,
			self.range_level / Progression.PROGRESSION_LEVEL_MAX,
			self.mage_level / Progression.PROGRESSION_LEVEL_MAX,
			self.fishing_level / Progression.PROGRESSION_LEVEL_MAX,
			self.herbalism_level / Progression.PROGRESSION_LEVEL_MAX,
			self.prospecting_level / Progression.PROGRESSION_LEVEL_MAX,
			self.carving_level / Progression.PROGRESSION_LEVEL_MAX,
			self.alchemy_level / Progression.PROGRESSION_LEVEL_MAX
		])


def _get_entity_data(obs: Observations) -> tuple[np.ndarray, np.ndarray]:
	"""
	Get entity data for the CNN (of shape (15, 15, 21)) and self data (of shape (21,))
	"""
	if np.all(obs.entities.id != obs.agent_id):
		print(f"[{obs.current_tick:4d}] Agent {obs.agent_id} not found in entities ({np.sum(obs.entities.id != 0)} entities seen)")
		return np.zeros((_view_size, _view_size, 21)), np.zeros((21,))

	me_idx = np.where(obs.entities.id == obs.agent_id)[0][0]
	me = SingleEntity.from_entity_data(obs.entities, me_idx)

	other_entites = [SingleEntity.from_entity_data(obs.entities, i)
				  	for i, a_id in enumerate(obs.entities.id)
				   	if a_id != obs.agent_id and a_id != 0]

	cnn_data = np.zeros((_view_size, _view_size, 21))
	for entity in other_entites:
		cnn_data[entity.row - me.row + _view_radius, entity.col - me.col + _view_radius, :] = \
			np.concatenate([
				entity.to_input_array(),
				np.array([
					obs.action_targets.attack_target[me_idx],
					np.sqrt((entity.row - me.row) ** 2 + (entity.col - me.col) ** 2) / np.sqrt(_view_radius ** 2 + _view_radius ** 2),
			  	]),
			])

	my_data = SingleEntity.from_entity_data(obs.entities, me_idx)
	return cnn_data, np.concatenate([
			my_data.to_input_array(),
			np.array([
				my_data.col / _map_size_x,
				my_data.row / _map_size_y
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
	Encode tiles into a (15, 15, 16) array
	"""
	types = tiles[:, :, 2].reshape(_view_size, _view_size, 1),
	result = np.zeros((_view_size, _view_size, 16))
	for i in range(16):
		result[:, :, i] = (types == i).astype(np.float32)	

	return result
