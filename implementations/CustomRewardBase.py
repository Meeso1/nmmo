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
    def clear_episode(self) -> None:
        """
        Clear any episode-specific state that the reward might have
        """
        pass
    
    def advance_episode(self) -> None:
        """
        Change reward state to the next episode (not called during evaluation)
        """
        pass
    
    def reset(self) -> None:
        """
        Reset the reward to the initial state
        """
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
    
    def clear_episode(self) -> None:
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
    
    def clear_episode(self) -> None:
        pass


class ResourcesAndGatheringReward(CustomRewardBase):
    def __init__(
        self, 
        max_lifetime: int, 
        gathering_bonus: int = 10, 
        scale_with_resource_change: bool = False
    ) -> None:
        self.max_lifetime = max_lifetime
        self.gathering_bonus = gathering_bonus
        self.scale_with_resource_change = scale_with_resource_change
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
            if self.scale_with_resource_change:
                gathering_reward += (water - self.last_water[agent_id]) / 100 * self.gathering_bonus
            else:
                gathering_reward = self.gathering_bonus
            
        if food > self.last_food[agent_id] and self.last_food[agent_id] < self.last_water[agent_id]:
            if self.scale_with_resource_change:
                gathering_reward += (food - self.last_food[agent_id]) / 100 * self.gathering_bonus
            else:
                gathering_reward = self.gathering_bonus
            
        self.last_water[agent_id] = water
        self.last_food[agent_id] = food
        
        reward = 0 if hp == 0 else max(gathering_reward, resources_reward)
        
        # Normalize so that max total reward is ~1 (assuming that ideally agent gets bonus every 3 steps)
        if not self.scale_with_resource_change:
            normalization_factor = 1 + (self.gathering_bonus - 1) / 3 if self.gathering_bonus > 1 else 1
        else:
            water_loss_per_step = 5
            food_loss_per_step = 5
            normalization_factor = 1 + (water_loss_per_step + food_loss_per_step) / 100 * self.gathering_bonus

        return reward / normalization_factor
    
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
    
    def clear_episode(self) -> None:
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

    def clear_episode(self) -> None:
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
    
    def clear_episode(self) -> None:
        for reward in self.rewards_with_weights.keys():
            reward.clear_episode()


class ShiftingReward(CustomRewardBase):
    def __init__(self, initial_reward: CustomRewardBase, final_reward: CustomRewardBase, shift_episodes: int) -> None:
        self.initial_reward = initial_reward
        self.final_reward = final_reward
        self.shift_episodes = shift_episodes
        self.episode = 0
        
    def get_rewards(
        self, 
        step: int,
        observations_per_agent: dict[int, Observations], 
        rewards: dict[int, float],
        terminations: dict[int, bool], 
        truncations: dict[int, bool]
    ) -> dict[int, float]:
        # Linearly shift from initial to final reward over shift_episodes episodes
        alpha = min(1, self.episode / self.shift_episodes)
        return {
            agent_id: (1 - alpha) * self.initial_reward.get_rewards(step, observations_per_agent, rewards, terminations, truncations)[agent_id] +
                       alpha * self.final_reward.get_rewards(step, observations_per_agent, rewards, terminations, truncations)[agent_id]
            for agent_id in rewards.keys()
        }
    
    def clear_episode(self) -> None:
        self.initial_reward.clear_episode()
        self.final_reward.clear_episode()
        
    def advance_episode(self) -> None:
        self.episode += 1
        
    def reset(self) -> None:
        self.episode = 0
