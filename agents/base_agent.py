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
          content remain, OR
        - Both tags are stripped but thinking content remains.
        """
        import re

        # If a closing </think> tag exists, take everything after the
        # last one — this handles both <think>…</think> and bare …</think>
        if "</think>" in raw:
            raw = raw.rsplit("</think>", 1)[-1]

        text = raw.strip()
        lines = [l.strip() for l in text.splitlines() if l.strip()]

        if not lines:
            return text

        # If we successfully stripped </think>, last line is the action
        if "</think>" not in raw and len(lines) == 1:
            return lines[0]

        # Fallback: find the first line matching a valid ALFWorld action pattern
        action_pattern = re.compile(
            r"^(go to|take|put|open|close|clean|heat|cool|use|slice|examine)\s"
        )
        for line in lines:
            if action_pattern.match(line):
                return line

        # Last resort: return last non-empty line
        return lines[-1]

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
