import torch
from torch import optim
import numpy as np
from torch import Tensor
from pettingzoo import ParallelEnv
from dataclasses import dataclass

from implementations.PpoAgent import PPOAgent
from implementations.Observations import Observations, ActionTargets


def to_observations(obs: dict[str]) -> Observations:
    return Observations(
        agent_id=obs["AgentId"],
        current_tick=obs["CurrentTick"],
        inventory=obs["Inventory"],
        tiles=obs["Tile"],
        entities=obs["Entity"],
        action_targets=ActionTargets(
            move_direction=obs["ActionTargets"]["Move"]["Direction"],
            attack_style=obs["ActionTargets"]["Attack"]["Style"],
            attack_target=obs["ActionTargets"]["Attack"]["Target"],
            use_inventory_item=obs["ActionTargets"]["Use"]["InventoryItem"],
            destroy_inventory_item=obs["ActionTargets"]["Destroy"]["InventoryItem"]
        )
    )


# TODO:
# - Try if current implementation works with multi-output PPONetwork
# - Move many things from train_ppo to PPOAgent
# - Try multi-input PPONetwork

def train_ppo(
    env: ParallelEnv,
    episodes: int = 1000,
) -> None:
    agent = PPOAgent()
    avg_rewards = []
    
    for episode in range(episodes):
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
        
        if episode % 1 == 0:
            print(f"Episode {episode}, Average Reward: {avg_reward:.2f}")
