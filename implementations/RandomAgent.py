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
        for agent_id, _ in states.items():
            items = {
                key: np.random.randint(0, choices)
                for key, choices in self.action_dims.items()
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
