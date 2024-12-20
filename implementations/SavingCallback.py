from implementations.train_ppo import EvaluationCallback
from implementations.Observations import Observations
from implementations.jar import Jar


class SavingCallback(EvaluationCallback):
    def __init__(self, name: str, saved_agent_ids: list[int]) -> None:
        self.name = name
        self.saved_agent_ids = saved_agent_ids
        self.episodes: list[tuple[list[tuple[dict[int, Observations], dict[int, dict]]], dict[int, float]]] = []
        self.current_episode: list[tuple[dict[int, Observations], dict[int, dict]]] = []
        self.jar = Jar("saves")

    def step(
        self,
        observations_per_agent: dict[int, Observations],
        actions_per_agent: dict[int, dict[str, dict[str, int]]],
        episode: int,
        step: int
    ) -> None:
        self.current_episode.append((
            {agent_id: obs for agent_id, obs in observations_per_agent.items() if agent_id in self.saved_agent_ids},
            {agent_id: actions for agent_id, actions in actions_per_agent.items() if agent_id in self.saved_agent_ids}
        ))

    def episode_start(self, episode: int) -> None:
        pass

    def episode_end(self, episode: int, rewards_per_agent: dict[int, float]) -> None:
        self.episodes.append((self.current_episode, rewards_per_agent))
        self.current_episode = []
        self.jar.add(self.name, self.episodes)
        self.jar.remove_all_but_latest(self.name)
    