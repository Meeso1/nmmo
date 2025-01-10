from abc import ABC, abstractmethod
from typing import Any
from implementations.ActionData import ActionData


class EvaluationCallback(ABC):
    @abstractmethod
    def step(
        self,
        observations_per_agent: dict[int, Any], 
        actions_per_agent: dict[int, ActionData],
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
        losses: tuple[list[float], list[float], list[float]],
        eval_rewards: list[dict[int, float]] | None
    ) -> None:
        pass