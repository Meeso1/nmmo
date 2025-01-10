from typing import Any
import numpy as np
from torch import Tensor
import nmmo
from nmmo import config
from implementations.CustomRewardBase import CustomRewardBase
from implementations.Observations import Observations
from implementations.PpoAgent import AgentBase
from implementations.ActionData import ActionData
from implementations.EvaluationCallback import EvaluationCallback
from implementations.train_ppo import evaluate_agent


class RandomAgent(AgentBase):
    def __init__(self) -> None:
        self.action_dims: dict[str, int] = {"Move": 5,
                                       "Attack style": 3,
                                       "Attack target": 101,
                                       "Use": 13,
                                       "Destroy": 13}

    def get_actions(
        self,
        states: dict[int, Observations],
        return_most_probable: bool = False
    ) -> dict[int, ActionData]:
        actions = {}
        for agent_id, obs in states.items():
            masks = {
                "Move": obs.action_targets.move_direction,
                "Attack style": obs.action_targets.attack_style,
                "Attack target": obs.action_targets.attack_target,
                "Use": obs.action_targets.use_inventory_item,
                "Destroy": obs.action_targets.destroy_inventory_item
            }
            
            # for each mask, choose a random index where the mask is 1
            items = {
                key: np.random.choice(np.where(mask == 1)[0]) 
                for key, mask in masks.items()
            }
                        
            actions[agent_id] = ActionData({
                "Move": {
                    "Direction": items["Move"]
                },
                "Attack": {
                    "Style": items["Attack style"],
                    "Target": items["Attack target"]
                },
                "Use": {
                    "InventoryItem": items["Use"]
                },
                "Destroy": {
                    "InventoryItem": items["Destroy"]
                }
            }, {}, {}, {})
        return actions


def get_avg_lifetime_for_random_agent(config: config.Default, *, retries: int = 5) -> tuple[float, list[float]]:
    class Callback(EvaluationCallback):
        def __init__(self):
            self.avg_lifetimes = []
            self.current_lifetimes = {}
            
        def step(self, observations_per_agent: dict[int, Any], actions_per_agent: dict[int, ActionData], episode: int, step: int) -> None:
            if step == 0:
                for agent_id in observations_per_agent.keys():
                    self.current_lifetimes[agent_id] = 0
                    
            for agent_id in observations_per_agent.keys():
                self.current_lifetimes[agent_id] += 1
                
        def episode_end(
            self, 
            episode: int, 
            rewards_per_agent: dict[int, float], 
            losses: tuple[list[float], list[float], list[float]],
            eval_rewards: list[dict[int, float]] | None
        ) -> None:
            self.avg_lifetimes.append(np.mean(list(self.current_lifetimes.values())))
            self.current_lifetimes = {}
            
        def episode_start(self, episode: int) -> None:
            pass
        
    callback = Callback()
    evaluate_agent(
        nmmo.Env(config),
        agent=RandomAgent(),
        episodes=retries,
        callbacks=[callback],
        quiet=True
    )
    
    return np.mean(callback.avg_lifetimes), callback.avg_lifetimes
                

def get_avg_reward_for_random_agent(config: config.Default, *, reward: CustomRewardBase | None = None, retries: int = 5) -> tuple[float, list[float]]:
    avg_rewards, _ = evaluate_agent(
        nmmo.Env(config),
        agent=RandomAgent(),
        episodes=retries,
        custom_reward=reward,
        quiet=True
    )
    
    return np.mean(avg_rewards), avg_rewards
