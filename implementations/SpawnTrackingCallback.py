import matplotlib.pyplot as plt
import numpy as np

from implementations.ActionData import ActionData
from implementations.EvaluationCallback import EvaluationCallback
from implementations.Observations import Observations


class SpawnTrackingCallback(EvaluationCallback):
    def __init__(self) -> None:
        self.spawns_and_rewards_per_episode: dict[int, dict[int, tuple[tuple[int, int], float]]] = {}
        self.current_episode_spawns: dict[int, tuple[int, int]] = {}
        
    def step(
        self,
        observations_per_agent: dict[int, Observations], 
        actions_per_agent: dict[int, ActionData],
        episode: int, 
        step: int) -> None:
        if step != 0:
            return
        
        for agent_id, observations in observations_per_agent.items():
            self_data = observations.entities.get_by_id(agent_id)
            if self_data is None:
                print(f'Agent {agent_id} not found in its observations')
                continue
            
            self.current_episode_spawns[agent_id] = (self_data.row, self_data.col)
    
    def episode_start(self, episode: int) -> None:
        pass
    
    def episode_end(
        self, 
        episode: int, 
        rewards_per_agent: dict[int, float],
        losses: tuple[list[float], list[float], list[float]],
        eval_rewards: list[dict[int, float]] | None
    ) -> None:
        self.spawns_and_rewards_per_episode[episode] = {
            agent_id: (spawn, rewards_per_agent[agent_id])
            for agent_id, spawn in self.current_episode_spawns.items()
        }
        self.current_episode_spawns = {}
        
    def _calculate_density_and_rewards(self, max_distance: float, smoothing: float) -> tuple[list[float], list[float]]:
        density_values = []
        rewards = []
        
        for episode_data in self.spawns_and_rewards_per_episode.values():
            spawn_positions = [(pos[0], pos[1]) for pos, _ in episode_data.values()]
            episode_rewards = [reward for _, reward in episode_data.values()]
            
            # Calculate density value for each agent based on nearby agents
            for i, ((row, col), reward) in enumerate(episode_data.values()):
                other_positions = spawn_positions[:i] + spawn_positions[i+1:]
                if not other_positions:
                    continue
                    
                # Calculate inverse distance sum for nearby agents
                density = 0.0
                for other_row, other_col in other_positions:
                    distance = np.sqrt((row - other_row)**2 + (col - other_col)**2)
                    if distance <= max_distance:
                        density += 1.0 / (distance ** smoothing)
                
                density_values.append(density)
                rewards.append(reward)
                
        return density_values, rewards

    def plot_density_reward_correlation(self, max_distance: float = 25.0, smoothing: float = 2.0, save_as: str | None = None) -> None:
        density_values, rewards = self._calculate_density_and_rewards(max_distance, smoothing)
        
        if not density_values:
            print("No data to plot")
            return
            
        plt.figure(figsize=(10, 6))
        plt.scatter(density_values, rewards, alpha=0.5)
        plt.xlabel('Agent Density (Inverse Distance Sum)')
        plt.ylabel('Episode Reward')
        plt.title('Agent Density vs Reward Correlation')
        
        # Add trend line
        z = np.polyfit(density_values, rewards, 1)
        p = np.poly1d(z)
        plt.plot(density_values, p(density_values), "r--", alpha=0.8)
        
        # Add correlation coefficient
        correlation = np.corrcoef(density_values, rewards)[0, 1]
        plt.text(0.05, 0.95, f'Correlation: {correlation:.2f}', 
                transform=plt.gca().transAxes)
        
        plt.grid(True, alpha=0.3)
        
        if save_as is not None:
            plt.savefig(f'plots/plots/{save_as}')
            plt.close()
        else:
            plt.show()
