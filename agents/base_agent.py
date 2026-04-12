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
    _BLACKLIST_THRESHOLD = 3  # blacklist a command after this many total failures
    _FAIL_WINDOW = 5        # look-back window for the failure-rate guard

    def _init_loop_guard(self):
        """Call from subclass __init__ or reset to initialise tracking state."""
        self._recent_actions: list[str] = []
        self._recent_success: list[bool] = []
        self._failure_counts: dict[str, int] = {}   # per-command failure tally
        self._blacklisted: set[str] = set()          # commands that hit the threshold

    def _record_outcome(self, action: str, succeeded: bool):
        """Record an action and its outcome for loop detection."""
        self._recent_actions.append(action)
        self._recent_success.append(succeeded)
        if not succeeded:
            self._failure_counts[action] = self._failure_counts.get(action, 0) + 1
            if self._failure_counts[action] >= self._BLACKLIST_THRESHOLD:
                self._blacklisted.add(action)
        else:
            # A success clears the failure tally for that action
            self._failure_counts.pop(action, None)
            self._blacklisted.discard(action)

    def _loop_warning(self) -> str:
        """Return a warning string if the agent is stuck, else empty string.

        Detects two patterns:
          1. Same action repeated ≥ _REPEAT_THRESHOLD times in a row.
          2. Last _FAIL_WINDOW actions all failed (even if different).
        Also emits an explicit, named blacklist of commands that must not be retried.
        """
        actions = self._recent_actions
        success = self._recent_success

        parts: list[str] = []

        # Explicit blacklist — name the forbidden commands so the LLM can't miss them
        if self._blacklisted:
            cmd_list = "\n".join(f"  • {cmd}" for cmd in sorted(self._blacklisted))
            parts.append(
                f"FORBIDDEN COMMANDS — each has failed {self._BLACKLIST_THRESHOLD}+ times "
                f"and must NOT be issued again under any circumstances:\n{cmd_list}\n"
                f"Choose a completely different action."
            )

        # Pattern 1: same action repeated consecutively (catches early spirals before blacklist kicks in)
        if len(actions) >= self._REPEAT_THRESHOLD:
            tail = actions[-self._REPEAT_THRESHOLD:]
            if len(set(tail)) == 1 and tail[0] not in self._blacklisted:
                parts.append(
                    f"WARNING: '{tail[0]}' has been repeated {self._REPEAT_THRESHOLD} times "
                    f"consecutively and keeps failing. Do NOT issue it again."
                )

        # Pattern 2: everything in the window failed
        if len(success) >= self._FAIL_WINDOW:
            window = success[-self._FAIL_WINDOW:]
            if not any(window):
                parts.append(
                    f"WARNING: Your last {self._FAIL_WINDOW} actions ALL failed. "
                    f"Stop and reconsider your strategy entirely. Try a different "
                    f"system or a different sequence of steps."
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
