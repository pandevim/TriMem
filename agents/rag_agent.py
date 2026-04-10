"""
Phase 2: RAG-Augmented Agent.

Same as baseline but stores every observation in a vector DB.
Before acting, queries RAG for relevant facts about objects and locations.
This should reduce syntactic errors (exact IDs retrieved from RAG).
"""
from agents.base_agent import BaseAgent
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
You have access to a memory database. Before each action, you will be shown
relevant facts retrieved from your past observations. Use the EXACT system and
record IDs from these retrieved facts.
"""


class RAGAgent(BaseAgent):
    """Agent with RAG-augmented memory for exact fact retrieval."""

    name = "rag"

    def __init__(self):
        self.llm = get_llm()
        self.rag = RAGStore()
        self.history: list[dict] = []
        self.goal = ""
        self.current_location = "unknown"

    def reset(self, goal: str):
        self.history = []
        self.goal = goal
        self.current_location = "unknown"
        self.rag.reset()

    def act(self, observation: str, turn: int) -> tuple[str, TurnMetric]:
        # Track current system from last access command
        if self.history and self.history[-1]["role"] == "assistant":
            last_action = self.history[-1]["content"]
            if last_action.startswith("access "):
                self.current_location = last_action[7:]

        # Store observation in RAG
        self.rag.store_observation(turn, self.current_location, observation)

        # Query RAG for relevant context
        rag_context = ""
        if turn > 0:
            query = f"What records and systems are available? What system to access for: {self.goal}"
            facts = self.rag.query(query, top_k=5)
            if facts:
                rag_context = "\n\nRELEVANT MEMORY:\n" + "\n".join(f"- {f}" for f in facts)

        # Build message with RAG context injected
        user_msg = observation + rag_context
        self.history.append({"role": "user", "content": user_msg})

        resp = self.llm.chat(
            system=SYSTEM_PROMPT + f"\n\nYour current task: {self.goal}",
            messages=self.history,
        )

        action = self.parse_action(resp.text)
        tokens_in = resp.tokens_in
        tokens_out = resp.tokens_out
        latency = resp.latency_ms

        self.history.append({"role": "assistant", "content": action})

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
            success=True,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            memory_source="rag",
            latency_ms=latency,
            syntactic_error=syntactic_error,
            spatial_hallucination=spatial_hallucination,
        )

        return action, metric
