"""Base agent interface for all Tri-Mem variants."""
from abc import ABC, abstractmethod
from utils.metrics import TurnMetric


class BaseAgent(ABC):
    """Every agent variant implements this interface."""

    name: str = "base"

    @abstractmethod
    def act(self, observation: str, turn: int) -> tuple[str, TurnMetric]:
        """
        Given an observation from the environment, return:
        - action string to execute
        - TurnMetric with telemetry
        """
        ...

    @abstractmethod
    def reset(self, goal: str):
        """Reset agent state for a new task."""
        ...
