from abc import ABC, abstractmethod
import numpy as np

from implementations.Observations import Observations


class CustomRewardBase(ABC):
    @abstractmethod
    def get_rewards(
        self, 
        step: int,
        observations_per_agent: dict[int, Observations], 
        rewards: dict[int, float],
        terminations: dict[int, bool], 
        truncations: dict[int, bool]
        ) -> dict[int, float]:
        pass
    
    @abstractmethod
    def reset(self) -> None:
        pass
    

class LifetimeReward(CustomRewardBase):
    def __init__(self, max_lifetime: int) -> None:
        self.max_lifetime = max_lifetime
    
    def get_rewards(
        self, 
        step: int,
        observations_per_agent: dict[int, Observations], 
        rewards: dict[int, float],
        terminations: dict[int, bool], 
        truncations: dict[int, bool]
        ) -> dict[int, float]:
        return {
            agent_id: (0 if (terminations[agent_id] or truncations[agent_id]) else 1 / self.max_lifetime)
            for agent_id in rewards.keys()
        }
    
    def reset(self) -> None:
        pass
    
    
class ResourcesReward(CustomRewardBase):
    def __init__(self, max_lifetime: int) -> None:
        self.max_lifetime = max_lifetime
    
    def _get_resources(self, agent_id, observations: Observations) -> float:
        agent_idx = next(iter(i for i in range(len(observations.entities.id)) 
                              if observations.entities.id[i] == agent_id), None)
        hp = observations.entities.health[agent_idx]
        water = observations.entities.water[agent_idx]
        food = observations.entities.food[agent_idx]
        
        return 0 if hp == 0 else (hp+water+food) / 300
    
    def get_rewards(
        self, 
        step: int,
        observations_per_agent: dict[int, Observations], 
        rewards: dict[int, float],
        terminations: dict[int, bool], 
        truncations: dict[int, bool]
        ) -> dict[int, float]:
        return {
            agent_id: (0 if (terminations[agent_id] or truncations[agent_id]) 
                       else self._get_resources(agent_id, observations_per_agent[agent_id]) / self.max_lifetime)
            for agent_id in rewards.keys()
        }
    
    def reset(self) -> None:
        pass


class ResourcesAndGatheringReward(CustomRewardBase):
    def __init__(self, max_lifetime: int, gathering_bonus: int = 10) -> None:
        self.max_lifetime = max_lifetime
        self.gathering_bonus = gathering_bonus
        self.last_water = {}
        self.last_food = {}
    
    def _get_resources_reward(self, agent_id, observations: Observations) -> float:
        agent_idx = next(iter(i for i in range(len(observations.entities.id)) 
                              if observations.entities.id[i] == agent_id), None)
        hp = observations.entities.health[agent_idx]
        water = observations.entities.water[agent_idx]
        food = observations.entities.food[agent_idx]
        
        resources_reward = (hp+water+food) / 300
        
        if agent_id not in self.last_water:
            self.last_water[agent_id] = water
            self.last_food[agent_id] = food
            
        gathering_reward = 0
        if water > self.last_water[agent_id] and self.last_water[agent_id] < self.last_food[agent_id]:
            gathering_reward = self.gathering_bonus
            
        if food > self.last_food[agent_id] and self.last_food[agent_id] < self.last_water[agent_id]:
            gathering_reward = self.gathering_bonus
            
        self.last_water[agent_id] = water
        self.last_food[agent_id] = food
        
        reward = 0 if hp == 0 else max(gathering_reward, resources_reward)
        
        # Normalize so that max total reward is ~1 (assuming that ideally agent gets bonus every 3 steps)
        return reward / (1 + (self.gathering_bonus - 1) / 3) if self.gathering_bonus > 1 else reward
    
    def get_rewards(
        self, 
        step: int,
        observations_per_agent: dict[int, Observations], 
        rewards: dict[int, float],
        terminations: dict[int, bool], 
        truncations: dict[int, bool]
        ) -> dict[int, float]:
        return {
            agent_id: (0 if (terminations[agent_id] or truncations[agent_id]) 
                       else self._get_resources_reward(agent_id, observations_per_agent[agent_id]) / self.max_lifetime)
            for agent_id in rewards.keys()
        }
    
    def reset(self) -> None:
        self.last_water = {}
        self.last_food = {}


class ExplorationReward(CustomRewardBase):
    border_size = 16
    
    def __init__(self, max_lifetime: int, map_size: int = 128, view_radius: int = 7) -> None:
        self.max_lifetime = max_lifetime
        self.map_size = map_size
        self.view_radius = view_radius
        self.seen_tiles: dict[int, np.ndarray] = {}
    
    def get_rewards(
        self, 
        step: int,
        observations_per_agent: dict[int, Observations], 
        rewards: dict[int, float],
        terminations: dict[int, bool], 
        truncations: dict[int, bool]
        ) -> dict[int, float]:
        if step == 0:
            for agent_id in observations_per_agent.keys():
                self.seen_tiles[agent_id] = np.zeros((self.map_size, self.map_size), dtype=bool)
                
        exploration_rewards = {}
                
        for agent_id in rewards.keys():
            if agent_id not in observations_per_agent:
                exploration_rewards[agent_id] = 0
                continue
            
            observations = observations_per_agent[agent_id]
            self_data = observations.entities.get_by_id(agent_id)
            if self_data is None:
                print(f"Agent {agent_id} not found in observations")
                continue
            
            x, y = self_data.row - ExplorationReward.border_size, self_data.col - ExplorationReward.border_size
            
            # Mark all tiles in view radius as seen
            min_x = max(0, x - self.view_radius) 
            max_x = min(self.map_size, x + self.view_radius + 1)
            min_y = max(0, y - self.view_radius)
            max_y = min(self.map_size, y + self.view_radius + 1)
            
            previous_seen_tiles = self.seen_tiles[agent_id][min_x:max_x, min_y:max_y].copy()
            self.seen_tiles[agent_id][min_x:max_x, min_y:max_y] = True
            
            # Calculate exploration reward as a number of new tiles seen
            reward = np.sum(~previous_seen_tiles)
            
            # Normalize so that max total reward is ~1
            exploration_rewards[agent_id] = reward / self.max_lifetime / (self.view_radius * 2 + 1)
        
        return exploration_rewards

    def reset(self) -> None:
        self.seen_tiles = {}


class WeightedReward(CustomRewardBase):
    def __init__(self, rewards_with_weights: dict[CustomRewardBase, float]) -> None:
        self.rewards_with_weights = rewards_with_weights
        
    def get_rewards(
        self, 
        step: int,
        observations_per_agent: dict[int, Observations], 
        rewards: dict[int, float],
        terminations: dict[int, bool], 
        truncations: dict[int, bool]
        ) -> dict[int, float]:
        total_rewards = {agent_id: 0 for agent_id in rewards.keys()}
        
        for reward, weight in self.rewards_with_weights.items():
            rewards = reward.get_rewards(step, observations_per_agent, rewards, terminations, truncations)
            for agent_id in rewards.keys():
                total_rewards[agent_id] += rewards[agent_id] * weight
                
        weights_sum = sum(self.rewards_with_weights.values())
        for agent_id in total_rewards.keys():
            total_rewards[agent_id] /= weights_sum
        
        return total_rewards
