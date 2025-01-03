
"""
Similar to train_ppo.py but for the SACAgent. Uses on-policy data for demonstration;
in a real scenario, use replay buffer.
"""

from pettingzoo import ParallelEnv
from typing import Optional, List
from implementations.SACAgent import SACAgent
from implementations.EvaluationCallback import EvaluationCallback
# ...existing code for Observations, custom rewards, etc. if needed...


def train_sac(
    env: ParallelEnv,
    agent: SACAgent,
    *,
    episodes: int = 1000,
    print_every: Optional[int] = 1,
    save_every: Optional[int] = 100,
    agent_name: Optional[str] = None,
    start_episode: int = 1,
    callbacks: Optional[List[EvaluationCallback]] = None
) -> None:
    """
    Trains a SAC agent for a specified number of episodes using on-policy data.
    In practice, a replay buffer is recommended for SAC, but this example
    mimics train_ppo structure for demonstration.
    """
    if callbacks is None:
        callbacks = []

    agent_name = agent_name or 'sac_agent'
    last_print_episode = start_episode - 1

    for episode in range(start_episode, episodes + start_episode):
        for cb in callbacks:
            cb.episode_start(episode)

        states, _ = env.reset()
        # Episode data
        step_data = {
            agent_id: {
                'states': [],
                'actions': [],
                'rewards': [],
                'dones': []
            }
            for agent_id in env.agents
        }

        total_rewards = {agent_id: 0.0 for agent_id in env.agents}
        step = 0
        while env.agents:
            # Ask agent for actions
            action_dict = agent.select_actions(states)

            # Convert to environment-compatible actions
            env_actions = {}
            for agent_id in env.agents:
                # Adapt the agent's continuous action to discrete environment if needed
                env_actions[agent_id] = {}  # fill with environment's action format
                # Example: env_actions[agent_id]["Move"] = ...
                # For demonstration, storing the direct action tensor
                env_actions[agent_id] = action_dict[agent_id]["action_tensor"]

            next_states, rewards, terminations, truncations, _ = env.step(env_actions)

            for cb in callbacks:
                cb.step(states, env_actions, episode, step)

            for agent_id in env.agents:
                step_data[agent_id]['states'].append(states[agent_id])
                step_data[agent_id]['actions'].append(action_dict[agent_id])
                step_data[agent_id]['rewards'].append(rewards[agent_id])
                step_data[agent_id]['dones'].append(terminations[agent_id] or truncations[agent_id])
                total_rewards[agent_id] += rewards[agent_id]

            states = next_states
            step += 1

        # Train the agent once per episode (on-policy demonstration)
        agent.train_step(
            {aid: data['states'] for aid, data in step_data.items()},
            {aid: data['actions'] for aid, data in step_data.items()},
            {aid: data['rewards'] for aid, data in step_data.items()},
            {aid: data['dones'] for aid, data in step_data.items()}
        )

        for cb in callbacks:
            cb.episode_end(episode, total_rewards, ([], [], []))

        avg_reward = sum(total_rewards.values()) / len(total_rewards)

        if episode == start_episode:
            print(f"{len(total_rewards)} agents in the environment")

        if print_every is not None and (episode % print_every == 0 or episode == episodes + start_episode - 1):
            ep_range = (episode - last_print_episode)
            print(f"Episode {episode}, Average Reward: {avg_reward:.4f} over last {ep_range} episodes")
            last_print_episode = episode

        if save_every is not None and (episode % save_every == 0 or episode == episodes + start_episode - 1):
            agent.save(f"{agent_name}_ep{episode}")