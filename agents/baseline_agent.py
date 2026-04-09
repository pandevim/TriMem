"""
Phase 1: Baseline Text-Only Agent.

Appends the full text history every turn. This is the control group.
It will suffer from context rot and escalating token costs as the
conversation grows — exactly what Tri-Mem is designed to fix.
"""
from agents.base_agent import BaseAgent
from utils.metrics import TurnMetric
from utils.llm import get_llm


SYSTEM_PROMPT = """\
You are an expert household robot operating in a simulated home environment.
You must complete tasks by issuing one action per turn.

CRITICAL RULES:
- Issue EXACTLY ONE action per response. Nothing else. No explanation.
- Use the EXACT object IDs you observe (e.g., "apple 1", "mug 1", "fridge 1").
- Valid actions: go to <object>, take <object> from <object>, put <object> in/on <object>,
  open <object>, close <object>, clean <object> with <object>, heat <object> with <object>,
  cool <object> with <object>, use <object>, slice <object> with <object>, examine <object>
- If you get "Nothing happens", your syntax was wrong. Fix it.
- Think about what you need to do step by step: find the object, pick it up,
  bring it to the right appliance, perform the action, then place it at the destination.
"""


class BaselineAgent(BaseAgent):
    """Full-text-history agent. No memory optimization."""

    name = "baseline"

    def __init__(self):
        self.llm = get_llm()
        self.history: list[dict] = []
        self.goal = ""

    def reset(self, goal: str):
        self.history = []
        self.goal = goal

    def act(self, observation: str, turn: int) -> tuple[str, TurnMetric]:
        # Append observation to history
        self.history.append({"role": "user", "content": observation})

        resp = self.llm.chat(
            system=SYSTEM_PROMPT + f"\n\nYour current task: {self.goal}",
            messages=self.history,
        )

        action = self.parse_action(resp.text)
        tokens_in = resp.tokens_in
        tokens_out = resp.tokens_out
        latency = resp.latency_ms

        # Append action to history
        self.history.append({"role": "assistant", "content": action})

        # Detect errors
        syntactic_error = "nothing happens" in observation.lower()
        spatial_hallucination = (
            "nothing happens" in observation.lower()
            and turn > 3
            and any(
                action.startswith(p)
                for p in ["go to ", "take ", "open "]
            )
        )

        metric = TurnMetric(
            turn=turn,
            action=action,
            observation=observation,
            success=True,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            memory_source="none",
            latency_ms=latency,
            syntactic_error=syntactic_error,
            spatial_hallucination=spatial_hallucination,
        )

        return action, metric
