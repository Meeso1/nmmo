from abc import ABC, abstractmethod

from implementations.Observations import Observations


class CustomRewardBase(ABC):
    @abstractmethod
    def get_rewards(
        self, 
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
    def __init__(self, max_lifetime: int) -> None:
        self.max_lifetime = max_lifetime
        self.last_water = {}
        self.last_food = {}
    
    def _get_resources(self, agent_id, observations: Observations) -> float:
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
        if water > self.last_water[agent_id]:
            gathering_reward = 1
            
        if food > self.last_food[agent_id]:
            gathering_reward = 1
            
        self.last_water[agent_id] = water
        self.last_food[agent_id] = food
        
        return 0 if hp == 0 else max(gathering_reward, resources_reward)
    
    def get_rewards(
        self, 
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
        self.last_water = {}
        self.last_food = {}
