from typing import Any
from implementations.train_ppo import EvaluationCallback
from implementations.jar import Jar
from implementations.ActionData import ActionData


class SavingCallback(EvaluationCallback):
    def __init__(
        self, 
        name: str, 
        saved_agent_ids: list[int] | None = None, 
        append_to_existing: bool = False,
        overwrite: bool = False,
        reward_config: dict | None = None
    ) -> None:
        self.name = name
        self.saved_agent_ids = saved_agent_ids if saved_agent_ids is not None else []
        self.append_to_existing = append_to_existing
        self.overwrite = overwrite
        self.reward_config = reward_config
        
        self.verified_existing = False
        self.episodes: list[tuple[
            list[tuple[dict[int, dict], dict[int, dict]]], # Observations per agent + actions per agent
            dict[int, float],                              # Rewards per agent
            tuple[list[float], list[float], list[float]],  # Losses
            dict[int, int],                                # Lifetimes
            list[dict[int, dict[str, float]]],             # Entropies (per step -> per agent -> per action type)
            list[list[dict[int, float]] | None]            # Evaluation rewards (None in episodes without evaluation)
            ]] = []
        self.current_episode_obs_and_actions: list[tuple[dict[int, dict], dict[int, dict]]] = []
        self.current_episode_entropies: list[dict[int, dict[str, float]]] = []
        self.lifetimes_per_agent = {}
        self.jar = Jar("saves")
        self.rewards_jar = Jar("saves/rewards")
        
    def _check_existing(self) -> None:
        exists = self.name in self.jar
        
        if exists:
            if self.append_to_existing:
                self.episodes = self.jar.get(self.name)
            elif not self.overwrite:
                raise ValueError(f"Save with name '{self.name}' already exists")
        else:
            if self.append_to_existing:
                print(f"Save with name '{self.name}' does not exist - creating new save")  
                
    def _save_rewards(self) -> None:
        if self.reward_config is not None:
            if self.overwrite:
                self.rewards_jar.remove_all(self.name)
                
            # Ignore append_to_existing - both configs will be saved
            self.rewards_jar.add(self.name, self.reward_config)         

    def step(
        self,
        observations_per_agent: dict[int, Any],
        actions_per_agent: dict[int, ActionData],
        episode: int,
        step: int
    ) -> None:
        if not self.verified_existing:
            self._check_existing()
            self.verified_existing = True
            self._save_rewards()
        
        if step == 0:
            self.lifetimes_per_agent = {agent_id: 0 for agent_id in observations_per_agent.keys()}
        
        self.current_episode_obs_and_actions.append((
            {agent_id: obs for agent_id, obs in observations_per_agent.items() if agent_id in self.saved_agent_ids},
            {agent_id: actions.action_dict for agent_id, actions in actions_per_agent.items() if agent_id in self.saved_agent_ids}
        ))
        
        for agent_id in observations_per_agent.keys():
            self.lifetimes_per_agent[agent_id] += 1
            
        self.current_episode_entropies.append(
            {agent_id: {
                    name: distribution.entropy().sum().item() 
                    for name, distribution in actions.distributions.items()
                } 
             for agent_id, actions in actions_per_agent.items()})

    def episode_start(self, episode: int) -> None:
        pass

    def episode_end(
        self, 
        episode: int, 
        rewards_per_agent: dict[int, float],
        losses: tuple[list[float], list[float], list[float]],
        eval_rewards: list[dict[int, float]] | None
    ) -> None:
        self.episodes.append((
            self.current_episode_obs_and_actions, 
            rewards_per_agent, 
            losses, 
            self.lifetimes_per_agent,
            self.current_episode_entropies,
            eval_rewards))
        self.current_episode_obs_and_actions = []
        self.current_episode_entropies = []
        self.lifetimes_per_agent = {}
        
        self.jar.add(self.name, self.episodes)
        self.jar.remove_all_but_latest(self.name)
    