from pettingzoo import ParallelEnv

from implementations.CustomRewardBase import CustomRewardBase
from implementations.PpoAgent import AgentBase
from implementations.EvaluationCallback import EvaluationCallback


def train_ppo(
    env: ParallelEnv,
    agent: AgentBase,
    *,
    episodes: int = 1000,
    print_every: int | None = 1,
    save_every: int | None = 100,
    agent_name: str | None = None,
    start_episode: int = 1,
    custom_reward: CustomRewardBase | None = None,
    callbacks: list[EvaluationCallback] | None = None
) -> None:
    if callbacks is None:
        callbacks = []
    
    agent_name = agent_name or 'ppo_agent'
    avg_rewards = []
    last_print_episode = start_episode-1

    for episode in range(start_episode, episodes+start_episode):
        for callback in callbacks:
            callback.episode_start(episode)
        
        if custom_reward is not None:
            custom_reward.reset()

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
                agent_id: agent.get_observations_from_state(agent_state)
                for agent_id, agent_state in states.items()
                if agent_id in env.agents # Filter out inactive agents
            }
            
            action_data = agent.get_actions(observations)
            env_actions = {
                agent_id: action_data[agent_id].action_dict
                for agent_id in env.agents
            }

            states, rewards, terminations, truncations, _ = env.step(env_actions)
            if custom_reward is not None:
                rewards = custom_reward.get_rewards(observations, rewards, terminations, truncations)
            
            for callback in callbacks:
                callback.step(observations, env_actions, episode, step)

            for agent_id in env.agents:
                episode_data[agent_id]['states'].append(observations[agent_id])
                episode_data[agent_id]['actions'].append(action_data[agent_id].sampled_actions)
                episode_data[agent_id]['rewards'].append(rewards[agent_id])
                episode_data[agent_id]['log_probs'].append(action_data[agent_id].log_probs)
                episode_data[agent_id]['dones'].append(
                    terminations[agent_id] or truncations[agent_id]
                )
                total_rewards[agent_id] += rewards[agent_id]

            step += 1
            
        episode_losses = agent.update(
            {aid: data['states'] for aid, data in episode_data.items()},
            {aid: data['actions'] for aid, data in episode_data.items()},
            {aid: data['rewards'] for aid, data in episode_data.items()},
            {aid: data['log_probs'] for aid, data in episode_data.items()},
            {aid: data['dones'] for aid, data in episode_data.items()}
        )
        
        for callback in callbacks:
            callback.episode_end(episode, total_rewards, episode_losses)

        avg_reward = sum(total_rewards.values()) / len(total_rewards)
        avg_rewards.append(avg_reward)
        
        if episode == 1:
            print(f"{len(total_rewards)} agents in the environment")

        if print_every is not None and (episode % print_every == 0 or episode == episodes+start_episode-1):
            avg_reward_since_last_print = sum(avg_rewards[(last_print_episode-start_episode+1):]) / (episode - last_print_episode)
            print(f"Episode {episode}, Average Reward: {avg_reward_since_last_print:.4f} (Episode Reward: {avg_reward:.4f})")
            last_print_episode = episode

        if save_every is not None and (episode % save_every == 0 or episode == episodes+start_episode-1):
            agent.save(f"{agent_name}_at_ep{episode}")


def evaluate_agent(
    env: ParallelEnv, 
    agent: AgentBase, 
    *,
    episodes: int = 10,
    custom_reward: CustomRewardBase | None = None,
    callbacks: list[EvaluationCallback] | None = None,
    quiet: bool = False
) -> None:
    if callbacks is None:
        callbacks = []
    
    avg_rewards = []

    for episode in range(1, episodes+1):
        if custom_reward is not None:
            custom_reward.reset()
        
        states, _ = env.reset()
        total_rewards: dict[int, float] = {agent_id: 0.0 for agent_id in env.agents}

        for callback in callbacks:
            callback.episode_start(episode)
        
        step = 0
        while env.agents:
            observations = { 
                agent_id: agent.get_observations_from_state(states) 
                for agent_id, states in states.items()
                if agent_id in env.agents
            }
            
            action_data = agent.get_actions(observations)
            env_actions = {
                agent_id: action_data[agent_id][0]
                for agent_id in env.agents
            }

            states, rewards, terminations, truncations, _ = env.step(env_actions)
            if custom_reward is not None:
                rewards = custom_reward.get_rewards(observations, rewards, terminations, truncations)
            
            for callback in callbacks:
                callback.step(observations, env_actions, episode, step)

            for agent_id in env.agents:
                total_rewards[agent_id] += rewards[agent_id]
                
            step += 1
            
        for callback in callbacks:
            callback.episode_end(episode, total_rewards, ([], [], []))

        avg_reward = sum(total_rewards.values()) / len(total_rewards)
        avg_rewards.append(avg_reward)
        
        if not quiet:
            print(f"Episode {episode}, Average Reward: {avg_reward:.4f}")
