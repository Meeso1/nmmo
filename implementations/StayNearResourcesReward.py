import numpy as np
from nmmo import material

from implementations.CustomRewardBase import CustomRewardBase
from implementations.Observations import Observations


class StayNearResourcesReward(CustomRewardBase):
    def __init__(self, max_lifetime: int, target_distance: int = 3, view_radius: int = 7) -> None:
        self.max_lifetime = max_lifetime
        self.target_distance = target_distance
        self.view_radius = view_radius
    
    def get_rewards(
        self, 
        step: int,
        observations_per_agent: dict[int, Observations], 
        rewards: dict[int, float],
        terminations: dict[int, bool], 
        truncations: dict[int, bool]
        ) -> dict[int, float]:
        new_rewards = {}
        
        for agent_id in rewards.keys():
            if agent_id not in observations_per_agent:
                print(f"Agent {agent_id} not found in observations")
                new_rewards[agent_id] = 0
                continue
            
            tile_ids = observations_per_agent[agent_id].tiles[:, 2].reshape(2 * self.view_radius + 1, 2 * self.view_radius + 1)
            food_id = material.Foilage.index
            water_id = material.Water.index
            
            food_reward = 0
            water_reward = 0
            
            for distance in range(self.target_distance, self.view_radius + 1):
                square = tile_ids[
                    self.view_radius - distance : self.view_radius + distance + 1,
                    self.view_radius - distance : self.view_radius + distance + 1
                ]
                
                if np.any(square == food_id) and food_reward == 0:
                    food_reward = 1 - (distance - self.target_distance) / (self.view_radius + 1 - self.target_distance)
                    
                if np.any(square == water_id) and water_reward == 0:
                    water_reward = 1 - (distance - self.target_distance) / (self.view_radius + 1 - self.target_distance)
                    
                if food_reward > 0 and water_reward > 0:
                    break
                
            new_rewards[agent_id] = (food_reward + water_reward) / 2 / self.max_lifetime 
        
        return new_rewards              
            
    def reset(self) -> None:
        pass
