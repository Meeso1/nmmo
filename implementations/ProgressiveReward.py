import numpy as np
from implementations.CustomRewardBase import CustomRewardBase
from implementations.Observations import Observations


class ProgressiveReward(CustomRewardBase):
    def __init__(
        self, 
        rewards: list[CustomRewardBase], 
        window_size: int = 100,
        alpha: float = 0.1
    ) -> None:
        self.rewards = rewards
        self.weights = np.zeros(len(rewards))
        self.weights[0] = 1.0
        self.progress_threshold = 0.6
        self.variance_threshold = 0.1
        self.window_size = window_size
        self.reward_history = {i: [] for i in range(len(rewards))}
        self.current_level = 0
        self.alpha = alpha

    def update_progress_threshold(self):
        recent_performance = np.mean(self.reward_history[self.current_level][-self.window_size:])
        self.progress_threshold = min(0.8, max(0.4, recent_performance + 0.1))

    def should_progress(self) -> bool:
        if len(self.reward_history[self.current_level]) < self.window_size:
            return False
        avg_reward = np.mean(self.reward_history[self.current_level][-self.window_size:])
        reward_variance = np.var(self.reward_history[self.current_level][-self.window_size:])
        return (avg_reward > self.progress_threshold and 
                reward_variance < self.variance_threshold)

    def update_weights(self):
        if self.should_progress() and self.current_level < len(self.rewards) - 1:
            self.current_level += 1
            self.update_progress_threshold()
        
        target_weights = np.zeros(len(self.rewards))
        target_weights[self.current_level] = 1.0
        self.weights = (1 - self.alpha) * self.weights + self.alpha * target_weights

    def get_rewards(
        self,
        step: int,
        observations_per_agent: dict[int, Observations],
        rewards: dict[int, float],
        terminations: dict[int, bool],
        truncations: dict[int, bool]
    ) -> dict[int, float]:
        total_rewards = {agent_id: 0.0 for agent_id in rewards.keys()}
        
        # Get individual rewards from each reward component
        rewards_per_component = []
        for reward_component in self.rewards:
            component_rewards = reward_component.get_rewards(
                step, observations_per_agent, rewards, terminations, truncations)
            rewards_per_component.append(component_rewards)

        # Calculate average reward for the current step (for progression tracking)
        for i, component_rewards in enumerate(rewards_per_component):
            avg_reward = sum(component_rewards.values()) / len(component_rewards)
            self.reward_history[i].append(avg_reward)
            if len(self.reward_history[i]) > self.window_size:
                self.reward_history[i].pop(0)

        for agent_id in rewards.keys():
            for i, component_rewards in enumerate(rewards_per_component):
                total_rewards[agent_id] += self.weights[i] * component_rewards[agent_id]

        return total_rewards

    def episode_end(self) -> None:
        """Called after each episode"""
        self.update_weights()
        
        for reward in self.rewards:
            reward.episode_end()

    def reset(self) -> None:
        """Reset progress and history"""
        self.episode_end()
        self.weights = np.zeros(len(self.rewards))
        self.weights[0] = 1.0
        self.reward_history = {i: [] for i in range(len(self.rewards))}
        self.current_level = 0
