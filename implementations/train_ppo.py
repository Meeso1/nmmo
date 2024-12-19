from abc import ABC, abstractmethod
from pettingzoo import ParallelEnv

from implementations.PpoAgent import AgentBase, PPOAgent
from implementations.Observations import Observations, to_observations


class EvaluationCallback(ABC):
    @abstractmethod
    def step(
        self,
        observations_per_agent: dict[int, Observations], 
        actions_per_agent: dict[int, dict[str, dict[str, int]]], 
        episode: int, 
        step: int) -> None:
        pass
    
    @abstractmethod
    def episode_start(self, episode: int) -> None:
        pass
    
    @abstractmethod
    def episode_end(self, episode: int, rewards_per_agent: dict[int, float]) -> None:
        pass


def train_ppo(
    env: ParallelEnv,
    *,
    episodes: int = 1000,
    print_every: int | None = 1,
    save_every: int | None = 100,
    agent_name: str | None = None,
    start_state_name: str | None = None,
    callbacks: list[EvaluationCallback] | None = None
) -> None:
    if callbacks is None:
        callbacks = []
    
    agent_name = agent_name or 'ppo_agent'
    agent = PPOAgent.load(start_state_name) if start_state_name is not None else PPOAgent()
    avg_rewards = []

    for episode in range(1, episodes+1):
        for callback in callbacks:
            callback.episode_start(episode)
        
        states, _ = env.reset()
        episode_data = {
            agent_id: {
                'states': [],
                'actions': [],
                'rewards': [],
                'log_probs': [],
                'dones': []
            }
            for agent_id in env.agents
        }

        total_rewards: dict[int, float] = {agent_id: 0.0 for agent_id in env.agents}
        
        step = 0
        while env.agents:  # While there are active agents
            observations = {
                agent_id: to_observations(agent_state)
                for agent_id, agent_state in states.items()
            }
            action_data = agent.get_actions({
                agent_id: obs 
                for agent_id, obs in observations.items()
                if agent_id in env.agents
            })

            env_actions = {
                agent_id: action_data[agent_id][0]
                for agent_id in env.agents
            }

            states, rewards, terminations, truncations, _ = env.step(env_actions)
            
            for callback in callbacks:
                callback.step(observations, env_actions, episode, step)

            for agent_id in env.agents:
                episode_data[agent_id]['states'].append(observations[agent_id])
                episode_data[agent_id]['actions'].append(action_data[agent_id][1])
                episode_data[agent_id]['rewards'].append(rewards[agent_id])
                episode_data[agent_id]['log_probs'].append(action_data[agent_id][2])
                episode_data[agent_id]['dones'].append(
                    terminations[agent_id] or truncations[agent_id]
                )
                total_rewards[agent_id] += rewards[agent_id]

            step += 1
            
        for callback in callbacks:
            callback.episode_end(episode, total_rewards)

        agent.update(
            {aid: data['states'] for aid, data in episode_data.items()},
            {aid: data['actions'] for aid, data in episode_data.items()},
            {aid: data['rewards'] for aid, data in episode_data.items()},
            {aid: data['log_probs'] for aid, data in episode_data.items()},
            {aid: data['dones'] for aid, data in episode_data.items()}
        )

        avg_reward = sum(total_rewards.values()) / len(total_rewards)
        avg_rewards.append(avg_reward)

        if print_every is not None and (episode % print_every == 0 or episode == episodes):
            print(f"Episode {episode}, Average Reward: {avg_reward:.2f}")

        if save_every is not None and (episode % save_every == 0 or episode == episodes):
            agent.save(f"{agent_name}_at_ep{episode}")


def evaluate_agent(
    env: ParallelEnv, 
    agent: AgentBase, 
    *,
    episodes=10,
    callbacks: list[EvaluationCallback] | None = None
) -> None:
    if callbacks is None:
        callbacks = []
    
    avg_rewards = []

    for episode in range(1, episodes+1):
        states, _ = env.reset()
        total_rewards: dict[int, float] = {agent_id: 0.0 for agent_id in env.agents}

        for callback in callbacks:
            callback.episode_start(episode)
        
        step = 0
        while env.agents:
            observations = { 
                agent_id: to_observations(states) 
                for agent_id, states in states.items()
            }
            
            action_data = agent.get_actions({agent_id: obs 
                 for agent_id, obs in observations.items() 
                 if agent_id in env.agents})

            env_actions = {
                agent_id: action_data[agent_id][0]
                for agent_id in env.agents
            }

            states, rewards, _, _, _ = env.step(env_actions)
            
            for callback in callbacks:
                callback.step(observations, env_actions, episode, step)

            for agent_id in env.agents:
                total_rewards[agent_id] += rewards[agent_id]
                
            step += 1
            
        for callback in callbacks:
            callback.episode_end(episode, total_rewards)

        avg_reward = sum(total_rewards.values()) / len(total_rewards)
        avg_rewards.append(avg_reward)
        print(f"Episode {episode}, Average Reward: {avg_reward:.2f}")
