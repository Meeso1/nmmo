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
