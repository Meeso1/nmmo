from abc import ABC, abstractmethod
from typing import Any


class EvaluationCallback(ABC):
    @abstractmethod
    def step(
        self,
        observations_per_agent: dict[int, Any], 
        actions_per_agent: dict[int, dict[str, dict[str, int]]], 
        episode: int, 
        step: int) -> None:
        pass
    
    @abstractmethod
    def episode_start(self, episode: int) -> None:
        pass
    
    @abstractmethod
    def episode_end(
        self, 
        episode: int, 
        rewards_per_agent: dict[int, float],
        losses: tuple[list[float], list[float], list[float]]) -> None:
        pass