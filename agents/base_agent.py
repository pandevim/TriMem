"""Base agent interface for all Tri-Mem variants."""
from abc import ABC, abstractmethod
from utils.metrics import TurnMetric


class BaseAgent(ABC):
    """Every agent variant implements this interface."""

    name: str = "base"

    @staticmethod
    def parse_action(raw: str) -> str:
        """Extract the actual action from raw LLM output.

        Handles chain-of-thought from models like Qwen/DeepSeek where:
        - Full ``<think>...</think>`` blocks may be present, OR
        - The opening ``<think>`` tag is stripped by the tokenizer
          (skip_special_tokens) but ``</think>`` and the thinking
          content remain.
        """
        # If a closing </think> tag exists, take everything after the
        # last one — this handles both <think>…</think> and bare …</think>
        if "</think>" in raw:
            raw = raw.rsplit("</think>", 1)[-1]
        text = raw.strip()
        # The action is typically the last non-empty line
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        return lines[-1] if lines else text

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
