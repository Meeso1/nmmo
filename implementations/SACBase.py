"""
Defines a base class for SAC-based agents, specifying required methods and expected input/output shapes.
"""

from abc import ABC, abstractmethod


class SACBase(ABC):
    """
    A base class for Soft Actor-Critic (SAC) style agents.
    """

    @abstractmethod
    def select_actions(
        self,
        states: dict[int, any]
    ) -> dict[int, any]:
        """
        Given a batch of states (indexed by agent_id), return chosen actions.

        Args:
            states: A dictionary keyed by agent_id. Each value is an observation object
                    or tensor representing state (shape depends on environment specifics).

        Returns:
            A dictionary keyed by agent_id. Each value should contain:
                1) The action in environment's expected format.
                2) Any additional policy outputs/log probabilities if needed.
        """
        pass

    @abstractmethod
    def train_step(
        self,
        states: dict[int, list[any]],
        actions: dict[int, list[any]],
        rewards: dict[int, list[float]],
        dones: dict[int, list[bool]]
    ) -> None:
        """
        Executes one training update using the collected experience.

        Args:
            states: Dictionary of lists of states. Each key is agent_id.
            actions: Dictionary of lists of actions taken by each agent.
            rewards: Dictionary of lists of scalar rewards.
            dones: Dictionary of lists of booleans indicating whether an episode ended.
        """
        pass

    @abstractmethod
    def save(self, name: str | None = None) -> None:
        """
        Saves agent state (e.g. network parameters) to disk.

        Args:
            name: Optional string specifying save file or identifier.
        """
        pass

    @classmethod
    @abstractmethod
    def load(cls, name: str | None = None) -> 'SACBase':
        """
        Loads agent state (e.g. network parameters) from disk into a new instance.

        Args:
            name: Optional string specifying load file or identifier.

        Returns:
            A new instance of SACBase with loaded parameters.
        """
        pass