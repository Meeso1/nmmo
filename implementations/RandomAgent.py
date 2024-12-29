import numpy as np
from torch import Tensor
from implementations.Observations import Observations
from implementations.PpoAgent import AgentBase


class RandomAgent(AgentBase):
    def __init__(self) -> None:
        self.action_dims: dict[str, int] = {"Move": 5,
                                       "Attack style": 3,
                                       "Attack target": 101,
                                       "Use": 13,
                                       "Destroy": 13}

    def get_actions(
        self,
        states: dict[int, Observations]
    ) -> dict[int, tuple[dict[str, dict[str, int]], dict[str, Tensor], dict[str, Tensor]]]:
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
                        
            actions[agent_id] = ({
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
            }, {}, {})
        return actions
