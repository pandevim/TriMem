"""Base agent interface for all Tri-Mem variants."""
import re
from abc import ABC, abstractmethod
from utils.metrics import TurnMetric


class BaseAgent(ABC):
    """Every agent variant implements this interface."""

    name: str = "base"

    # Valid command verbs for the IT audit environment
    _CMD_VERBS = ("access", "query", "download", "upload", "revoke", "scan", "run")
    _CMD_PATTERN = re.compile(
        r"^(" + "|".join(_CMD_VERBS) + r")\s"
    )
    # Pattern to find a command embedded anywhere in reasoning text
    _EMBEDDED_CMD = re.compile(
        r"\b(" + "|".join(_CMD_VERBS) + r")"
        r"\s+[a-zA-Z]\S*"              # first argument (system/record name)
        r"(?:\s+(?:from|to|with)\s+\S+)?"  # optional preposition clause
    )

    @staticmethod
    def parse_action(raw: str) -> str:
        """Extract the actual action from raw LLM output.

        Handles chain-of-thought from models like Qwen/DeepSeek where:
        - Full ``<think>...</think>`` blocks may be present, OR
        - The opening ``<think>`` tag is stripped by the tokenizer
          (skip_special_tokens) but ``</think>`` and the thinking
          content remain, OR
        - Both tags are stripped AND the output was truncated at
          max_tokens, leaving pure reasoning with no command at all.
        """

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

        # Look for a line that starts with a valid command verb
        for line in lines:
            if BaseAgent._CMD_PATTERN.match(line):
                return line

        # ── Orphaned reasoning fallback ──
        # No clean command line found. This typically happens when the
        # model hit max_tokens mid-reasoning (<think> stripped, </think>
        # never generated). Try to extract the *last* embedded command
        # from the reasoning text — it's usually what the model was
        # building up to before truncation.
        matches = list(BaseAgent._EMBEDDED_CMD.finditer(text))
        if matches:
            return matches[-1].group(0)

        # Nothing extractable — return last line (will cause syntax error
        # but at least won't inject paragraphs of reasoning into history)
        return lines[-1].split(".")[0][:80]

    # ── Loop detection ──

    _REPEAT_THRESHOLD = 3   # flag after this many consecutive repeats
    _FAIL_WINDOW = 5        # look-back window for the failure-rate guard

    def _init_loop_guard(self):
        """Call from subclass __init__ or reset to initialise tracking state."""
        self._recent_actions: list[str] = []
        self._recent_success: list[bool] = []

    def _record_outcome(self, action: str, succeeded: bool):
        """Record an action and its outcome for loop detection."""
        self._recent_actions.append(action)
        self._recent_success.append(succeeded)

    def _loop_warning(self) -> str:
        """Return a warning string if the agent is stuck, else empty string.

        Detects two patterns:
          1. Same action repeated ≥ _REPEAT_THRESHOLD times in a row.
          2. Last _FAIL_WINDOW actions all failed (even if different).
        """
        actions = self._recent_actions
        success = self._recent_success

        parts: list[str] = []

        # Pattern 1: same action repeated
        if len(actions) >= self._REPEAT_THRESHOLD:
            tail = actions[-self._REPEAT_THRESHOLD:]
            if len(set(tail)) == 1:
                parts.append(
                    f"WARNING: You have repeated '{tail[0]}' {self._REPEAT_THRESHOLD} "
                    f"times and it keeps failing. Do NOT try this command again. "
                    f"Try a completely different approach."
                )

        # Pattern 2: everything in the window failed
        if len(success) >= self._FAIL_WINDOW:
            window = success[-self._FAIL_WINDOW:]
            if not any(window):
                parts.append(
                    f"WARNING: Your last {self._FAIL_WINDOW} actions ALL failed. "
                    f"Stop and reconsider your strategy. Try a different system "
                    f"or a different sequence of steps."
                )

        return "\n".join(parts)

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
