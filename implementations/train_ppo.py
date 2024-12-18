from pettingzoo import ParallelEnv

from implementations.PpoAgent import PPOAgent
from implementations.Observations import to_observations


def train_ppo(
    env: ParallelEnv,
    *,
    episodes: int = 1000,
    print_every: int | None = 1,
    save_every: int | None = 100,
    agent_name: str | None = None,
    start_state_name: str | None = None
) -> None:
    agent_name = agent_name or 'ppo_agent'
    agent = PPOAgent.load(start_state_name) if start_state_name is not None else PPOAgent()
    avg_rewards = []

    for episode in range(1, episodes+1):
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
        while env.agents:  # While there are active agents
            observations = {
                agent_id: to_observations(agent_state)
                for agent_id, agent_state in states.items()
                if agent_id in env.agents
            }
            action_data = agent.get_actions(observations)

            env_actions = {
                agent_id: action_data[agent_id][0]
                for agent_id in env.agents
            }

            next_states, rewards, terminations, truncations, _ = env.step(env_actions)

            for agent_id in env.agents:
                episode_data[agent_id]['states'].append(observations[agent_id])
                episode_data[agent_id]['actions'].append(action_data[agent_id][1])
                episode_data[agent_id]['rewards'].append(rewards[agent_id])
                episode_data[agent_id]['log_probs'].append(action_data[agent_id][2])
                episode_data[agent_id]['dones'].append(
                    terminations[agent_id] or truncations[agent_id]
                )
                total_rewards[agent_id] += rewards[agent_id]

            states = next_states

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
