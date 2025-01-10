import matplotlib.pyplot as plt
import numpy as np

from implementations.ActionData import ActionData
from implementations.EvaluationCallback import EvaluationCallback
from implementations.Observations import Observations


class PathTrackingCallback(EvaluationCallback):
    def __init__(self) -> None:
        self.paths_per_episode: dict[int, dict[int, list[tuple[int, int]]]] = {}
        self.current_episode_paths: dict[int, list[tuple[int, int]]] = {}
        
    def step(
        self,
        observations_per_agent: dict[int, Observations], 
        actions_per_agent: dict[int, ActionData],
        episode: int, 
        step: int) -> None:
        if step == 0:
            for agent_id in observations_per_agent.keys():
                self.current_episode_paths[agent_id] = []
                
        for agent_id, observations in observations_per_agent.items():
            self_data = observations.entities.get_by_id(agent_id)
            if self_data is None:
                print(f'Agent {agent_id} not found in its observations')
                continue
            
            self.current_episode_paths[agent_id].append((self_data.row, self_data.col))
    
    def episode_start(self, episode: int) -> None:
        pass
    
    def episode_end(
        self, 
        episode: int, 
        rewards_per_agent: dict[int, float],
        losses: tuple[list[float], list[float], list[float]],
        eval_rewards: list[dict[int, float]] | None
    ) -> None:
        self.paths_per_episode[episode] = self.current_episode_paths
        self.current_episode_paths = {}
        
    def plot_paths(self, episode: int) -> None:
        plt.figure(figsize=(10, 10))
        plt.grid(True)
        plt.xlim(0, 160)
        plt.ylim(0, 160)

        paths = self.paths_per_episode[episode]
        colors = plt.cm.rainbow(np.linspace(0, 1, len(paths)))
        for path, color in zip(paths.values(), colors):
            for pos in path:
                plt.fill([pos[1], pos[1]+1, pos[1]+1, pos[1]], 
                         [pos[0], pos[0], pos[0]+1, pos[0]+1],
                         color=color, 
                         alpha=0.3)

        plt.title(f'Agent Paths - Episode {episode}')
        plt.show()
