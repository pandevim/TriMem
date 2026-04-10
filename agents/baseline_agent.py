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
You are an expert IT auditor operating in the NovaCorp corporate network environment.
You must complete audit tasks by issuing one terminal command per turn.

CRITICAL RULES:
- Issue EXACTLY ONE command per response. Nothing else. No explanation.
- Use the EXACT system and record IDs you observe (e.g., "procurement_db", "invoice_1").
- Valid commands:
    access <system>                  — connect to a system
    query <system>                   — list records in a system
    download <record> from <system>  — retrieve a record to your local workspace
    upload <record> to <system>      — send a record from your local workspace
    revoke <token> with <system>     — revoke credentials via a system
    scan <record> with <system>      — scan a record using a security tool
    run <script>                     — execute a script on the current system
- If you get "Syntax error" or "Command executed but returned no results", your syntax was wrong. Fix it.
- Follow prerequisites: you must access a system before downloading from it.
- Think step by step: access the source system, query if needed, download the record,
  access the destination system, then upload/act on the record.
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
        syntactic_error = "syntax error" in observation.lower()
        spatial_hallucination = (
            ("syntax error" in observation.lower() or "returned no results" in observation.lower())
            and turn > 3
            and any(
                action.startswith(p)
                for p in ["access ", "download ", "upload ", "query "]
            )
        )

        metric = TurnMetric(
            turn=turn,
            action=action,
            observation=observation,
            success=False,  # updated by runner after env.step()
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            memory_source="none",
            latency_ms=latency,
            syntactic_error=syntactic_error,
            spatial_hallucination=spatial_hallucination,
            llm_output=resp.text,
        )

        return action, metric
