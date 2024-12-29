from implementations.Observations import ActionTargets, EntityData, Observations
from nmmo.entity.entity import EntityState


def to_observations(obs: dict[str]) -> Observations:
    return Observations(
        agent_id=obs["AgentId"],
        current_tick=obs["CurrentTick"],
        inventory=obs["Inventory"],
        tiles=obs["Tile"],
        entities=EntityData(
            **{feature: obs["Entity"][:, idx] 
            for feature, idx in EntityState.State.attr_name_to_col.items()}
        ),
        action_targets=ActionTargets(
            move_direction=obs["ActionTargets"]["Move"]["Direction"],
            attack_style=obs["ActionTargets"]["Attack"]["Style"],
            attack_target=obs["ActionTargets"]["Attack"]["Target"],
            use_inventory_item=obs["ActionTargets"]["Use"]["InventoryItem"],
            destroy_inventory_item=obs["ActionTargets"]["Destroy"]["InventoryItem"]
        )
    )
