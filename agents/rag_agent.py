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
You are an expert household robot operating in a simulated home environment.
You must complete tasks by issuing one action per turn.

CRITICAL RULES:
- Issue EXACTLY ONE action per response. Nothing else. No explanation.
- Use the EXACT object IDs you observe (e.g., "apple 1", "mug 1", "fridge 1").
- Valid actions: go to <object>, take <object> from <object>, put <object> in/on <object>,
  open <object>, close <object>, clean <object> with <object>, heat <object> with <object>,
  cool <object> with <object>, use <object>, slice <object> with <object>, examine <object>
- If you get "Nothing happens", your syntax was wrong. Fix it.

MEMORY SYSTEM:
You have access to a memory database. Before each action, you will be shown
relevant facts retrieved from your past observations. Use the EXACT object IDs
and location names from these retrieved facts.
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
        # Track location from observation
        if "you see:" in observation.lower():
            # Try to extract location from previous action
            if self.history and self.history[-1]["role"] == "assistant":
                last_action = self.history[-1]["content"]
                if last_action.startswith("go to "):
                    self.current_location = last_action[6:]

        # Store observation in RAG
        self.rag.store_observation(turn, self.current_location, observation)

        # Query RAG for relevant context
        rag_context = ""
        if turn > 0:
            query = f"What objects are available? Where should I go for: {self.goal}"
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

        action = resp.text
        tokens_in = resp.tokens_in
        tokens_out = resp.tokens_out
        latency = resp.latency_ms

        self.history.append({"role": "assistant", "content": action})

        syntactic_error = "nothing happens" in observation.lower()
        spatial_hallucination = (
            "nothing happens" in observation.lower()
            and turn > 3
            and any(action.startswith(p) for p in ["go to ", "take ", "open "])
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
