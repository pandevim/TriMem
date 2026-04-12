"""
Phase 3: Visual Bus Agent.

Replaces the growing raw-text history with a compressed visual timeline.
Each turn:
  1. Raw history → rendered to a colour-coded PNG
  2. DeepSeek-OCR-2 reads the image → compressed markdown summary
  3. Reasoning LLM (Qwen) sees: system prompt + compressed summary + current obs + RAG
  4. Action is executed; raw history is updated for next render cycle

Token cost profile:
  Baseline : grows linearly with turns (full transcript every call)
  This agent: ~constant per turn (compressed visual summary replaces transcript)

The agent also includes RAG for exact fact retrieval (object IDs, record names)
because visual compression is lossy for precise alphanumeric strings.
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

MEMORY SYSTEM:
You have two memory aids before each action:
  [COMPRESSED HISTORY] — a visual summary of all prior steps this session (rendered and OCR-compressed)
  [RELEVANT MEMORY]    — exact facts retrieved from a vector database (use these IDs verbatim)
"""


class VisualBusAgent(BaseAgent):
    """
    Agent using Visual Bus episodic memory + RAG for exact fact retrieval.

    The LLM never receives the raw growing transcript. Instead it gets a
    compressed visual summary produced by DeepSeek-OCR-2 each turn.
    """

    name = "visual_bus"

    def __init__(self):
        self.llm = get_llm()
        self.visual_bus = VisualBus()
        self.rag = RAGStore()
        # Raw history kept locally for image rendering only — never sent as-is to LLM
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
        # Give the visual bus a task ID so image filenames are organised
        task_slug = goal[:40].replace(" ", "_")
        self.visual_bus.reset(task_slug)

    def act(self, observation: str, turn: int) -> tuple[str, TurnMetric]:
        # Track which system we're connected to (for RAG metadata)
        if self._raw_history and self._raw_history[-1]["role"] == "assistant":
            last_action = self._raw_history[-1]["content"]
            if last_action.startswith("access "):
                self.current_location = last_action[7:]

        # Store observation in RAG for exact fact retrieval
        self.rag.store_observation(turn, self.current_location, observation)

        # --- Visual Bus: compress raw history into a single image summary ---
        compressed_history = self.visual_bus.compress(self._raw_history)

        # --- RAG: retrieve relevant exact facts ---
        rag_context = ""
        if turn > 0:
            query = f"What records and systems are available? {self.goal}"
            facts = self.rag.query(query, top_k=5)
            if facts:
                rag_context = "\n\n[RELEVANT MEMORY]\n" + "\n".join(f"- {f}" for f in facts)

        # Build a single compact message (not a growing list)
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

        # LLM only sees one message (compressed context + current obs).
        # This is what flattens the token cost curve.
        resp = self.llm.chat(
            system=SYSTEM_PROMPT + f"\n\nYour current task: {self.goal}",
            messages=[{"role": "user", "content": user_msg}],
        )

        action = self.parse_action(resp.text)

        # Record outcome for loop detection
        succeeded = not any(
            s in observation.lower()
            for s in ("syntax error", "returned no results", "access denied")
        )
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
            memory_source="visual_bus",
            latency_ms=resp.latency_ms,
            syntactic_error=syntactic_error,
            spatial_hallucination=spatial_hallucination,
            llm_output=resp.text,
        )

        return action, metric
