"""
Phase 3.5: Visual Bus + RAG Combined Agent.

Tests the Syntactic Action Gap hypothesis: visual compression preserves the
episodic *narrative* (what happened, the flow of the session) while RAG
guarantees lossless recall of exact identifiers that OCR blurs.

Key difference from the Phase 3 VisualBusAgent (which already had RAG):
  - RAG queries are **guided by the visual summary**.  After OCR compresses
    the history, we extract entity mentions (system names, record IDs,
    alphanumeric keys) from the compressed text and run *targeted* RAG
    lookups for each one.  This bridges the gap: the visual summary tells
    the agent *what* to look up, and RAG provides the *exact* string.
  - Observations are stored in RAG with success/failure metadata so the
    agent can recall "access to X failed at turn N" — the failure signal
    that raw visual compression loses.

Token cost profile:
  Baseline     : grows linearly (full transcript every call)
  Visual Bus   : ~constant (compressed visual summary, generic RAG)
  This agent   : ~constant (compressed visual summary, targeted RAG)
                  — same token cost as Visual Bus, but higher-quality
                    RAG context should reduce hallucinations and loops
"""
from agents.base_agent import BaseAgent
from memory.visual_bus import VisualBus
from memory.rag_store import RAGStore
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
- If a command keeps failing, do NOT repeat it. Try a different approach.

MEMORY SYSTEM:
You have two complementary memory aids before each action:
  [COMPRESSED HISTORY] — a visual summary of all prior steps (what happened, in what order)
  [EXACT FACTS]        — precise record/system IDs and outcomes retrieved from a fact database
                         These IDs are EXACT — prefer them over anything in the compressed history.
If the compressed history mentions a record but the ID looks garbled, check [EXACT FACTS] for the correct string.
"""


class VisualBusRAGAgent(BaseAgent):
    """
    Combined Visual Bus + RAG agent with summary-guided retrieval.

    The LLM never receives the raw growing transcript. Instead it gets:
      1. A compressed visual summary (episodic narrative)
      2. Targeted RAG results driven by entities mentioned in that summary
    """

    name = "visual_bus_rag"

    def __init__(self):
        self.llm = get_llm()
        self.visual_bus = VisualBus()
        self.rag = RAGStore()
        # Raw history kept locally for image rendering only — never sent to LLM
        self._raw_history: list[dict] = []
        self.goal = ""
        self.current_location = "unknown"
        self._init_loop_guard()

    def reset(self, goal: str):
        self._raw_history = []
        self.goal = goal
        self.current_location = "unknown"
        self.rag.reset()
        self._init_loop_guard()
        task_slug = goal[:40].replace(" ", "_")
        self.visual_bus.reset(task_slug)

    def act(self, observation: str, turn: int) -> tuple[str, TurnMetric]:
        # --- Track current system from last access command ---
        last_action = ""
        if self._raw_history and self._raw_history[-1]["role"] == "assistant":
            last_action = self._raw_history[-1]["content"]
            if last_action.startswith("access "):
                self.current_location = last_action[7:]

        # --- Store observation in RAG with success/failure metadata ---
        succeeded = not any(
            s in observation.lower()
            for s in ("syntax error", "returned no results", "access denied")
        )
        self.rag.store_observation(turn, self.current_location, observation)
        # Also store the action→outcome pair so RAG can recall failures
        if last_action and turn > 0:
            outcome = "succeeded" if succeeded else "FAILED"
            self.rag.store_fact(
                f"Turn {turn - 1}: '{last_action}' {outcome}. Response: {observation[:200]}",
                {
                    "turn": turn - 1,
                    "location": self.current_location,
                    "type": "action_outcome",
                    "success": succeeded,
                },
            )

        # --- Visual Bus: compress raw history into a single image summary ---
        compressed_history = self.visual_bus.compress(self._raw_history)

        # --- RAG: summary-guided targeted retrieval ---
        rag_context = ""
        if turn > 0:
            # Extract entity mentions from the visual summary
            entities = RAGStore.extract_entities(compressed_history) if compressed_history else []
            # Build targeted queries from extracted entities
            targeted_queries = [e for e in entities[:8]]  # cap to avoid excessive queries
            # Always include a goal-oriented query
            targeted_queries.append(self.goal)
            # Include current observation context
            targeted_queries.append(observation[:200])

            facts = self.rag.query_multi(targeted_queries, top_k=2)
            if facts:
                # Cap total injected facts to control token cost
                facts = facts[:10]
                rag_context = "\n\n[EXACT FACTS]\n" + "\n".join(f"- {f}" for f in facts)

        # --- Build single compact message ---
        user_content_parts = []
        if compressed_history:
            user_content_parts.append(f"[COMPRESSED HISTORY]\n{compressed_history}")
        user_content_parts.append(f"[CURRENT OBSERVATION]\n{observation}")
        if rag_context:
            user_content_parts.append(rag_context)

        # Inject loop warning if the agent is stuck
        loop_warn = self._loop_warning()
        if loop_warn:
            user_content_parts.append(loop_warn)

        user_msg = "\n\n".join(user_content_parts)

        # LLM sees one message (compressed context + current obs + targeted facts)
        resp = self.llm.chat(
            system=SYSTEM_PROMPT + f"\n\nYour current task: {self.goal}",
            messages=[{"role": "user", "content": user_msg}],
        )

        action = self.parse_action(resp.text)

        # Record outcome for loop detection (succeeded already computed above)
        self._record_outcome(action, succeeded)

        # Append raw obs + action to local history for next turn's rendering
        self._raw_history.append({"role": "user", "content": observation})
        self._raw_history.append({"role": "assistant", "content": action})

        syntactic_error = "syntax error" in observation.lower()
        spatial_hallucination = (
            ("syntax error" in observation.lower() or "returned no results" in observation.lower())
            and turn > 3
            and any(action.startswith(p) for p in ["access ", "download ", "upload ", "query "])
        )

        metric = TurnMetric(
            turn=turn,
            action=action,
            observation=observation,
            success=False,
            tokens_in=resp.tokens_in,
            tokens_out=resp.tokens_out,
            memory_source="visual_bus_rag",
            latency_ms=resp.latency_ms,
            syntactic_error=syntactic_error,
            spatial_hallucination=spatial_hallucination,
            llm_output=resp.text,
        )

        return action, metric
