"""
Phase 3.75: MSA Agent (simulated rulebook sparse-attention layer).

This is the "path 1 + path 2" agent from the MSA build plan:

  Path 1 — Functional simulation via prefix caching.
    The full NovaCorp IT Policy is embedded in a *byte-identical* system
    prompt every turn. vLLM's automatic prefix caching hashes the prompt
    prefix and reuses the pre-computed rulebook KV across turns, so the
    agent pays the rulebook compute exactly once per process even though
    it logically "re-reads" the rulebook every turn. Measured as the
    token-economic win MSA is supposed to deliver.

  Path 2 — Poor-man's MSA (chunk + route + inject).
    On each turn, the live query (goal + observation + last action) is
    routed against the chunked rulebook's embeddings. Top-k sections are
    pulled out and injected into the *user* message (not the system
    prompt — that would break the cache). This is the learned-router
    half of EverMind's design, swapped in as cosine similarity over
    mean-pooled chunk embeddings until the real sparse-attention layer
    lands.

When EverMind ships, the MSAStore query path becomes a sparse-attention
call and the frozen system prompt becomes a handle to a cached KV
tensor. Neither this agent nor the benchmark runner need to change.
"""
from agents.base_agent import BaseAgent
from memory.msa_store import MSAStore
from utils.metrics import TurnMetric
from utils.llm import get_llm
from configs.settings import (
    MSA_TOP_K,
    MSA_INJECT_FULL_RULEBOOK,
    MSA_INJECT_ROUTED_CHUNKS,
)


_STATIC_PREAMBLE = """\
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

MEMORY SYSTEM (MSA — Semantic Memory):
Your authoritative NovaCorp IT Policy & Procurement Guide is preloaded
into your semantic memory layer below. It contains the exact command
sequences for every standard audit SOP. When a user turn includes a
"ROUTED POLICY CONTEXT" block, those are the specific sections the
policy router has flagged as most relevant to the current turn — trust
them over your own recollection. If the routed context and the full
rulebook ever disagree, the routed context wins.

=== BEGIN NOVACORP IT POLICY (SEMANTIC MEMORY) ===

{rulebook}

=== END NOVACORP IT POLICY ===
"""


class MSAAgent(BaseAgent):
    """Semantic-memory agent: frozen rulebook in prompt + routed Top-k chunks per turn.

    Combines path 1 (prefix-cached static rulebook) and path 2 (learned
    router over chunked rulebook) from the Phase 3.75 plan.
    """

    name = "msa"

    def __init__(self):
        self.llm = get_llm()
        self.msa = MSAStore.shared()
        self.history: list[dict] = []
        self.goal = ""
        self.last_action = ""
        self._system_prompt = self._build_system_prompt()

    def _build_system_prompt(self) -> str:
        """Frozen per-process system prompt.

        Must be byte-identical across every call into ``self.llm.chat``
        so vLLM's prefix cache hits. We deliberately do NOT interpolate
        the task goal here — the goal goes in the first user message
        instead, which keeps the cached prefix reusable across tasks.
        """
        if MSA_INJECT_FULL_RULEBOOK:
            return _STATIC_PREAMBLE.format(rulebook=self.msa.full_rulebook_text)
        return _STATIC_PREAMBLE.format(rulebook="(rulebook access disabled)")

    def reset(self, goal: str):
        self.history = []
        self.goal = goal
        self.last_action = ""

    def act(self, observation: str, turn: int) -> tuple[str, TurnMetric]:
        routed_block = ""
        if MSA_INJECT_ROUTED_CHUNKS:
            query_text = (
                f"Task: {self.goal}\n"
                f"Current observation: {observation}\n"
                f"Last action: {self.last_action or '(none)'}"
            )
            chunks = self.msa.query(query_text, top_k=MSA_TOP_K)
            routed_block = MSAStore.format_routed_chunks(chunks)

        if turn == 0:
            user_msg = f"Your current task: {self.goal}\n\n{observation}"
        else:
            user_msg = observation
        if routed_block:
            user_msg = f"{user_msg}\n\n{routed_block}"

        self.history.append({"role": "user", "content": user_msg})

        resp = self.llm.chat(
            system=self._system_prompt,
            messages=self.history,
        )

        action = self.parse_action(resp.text)
        self.history.append({"role": "assistant", "content": action})
        self.last_action = action

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
            memory_source="msa",
            latency_ms=resp.latency_ms,
            syntactic_error=syntactic_error,
            spatial_hallucination=spatial_hallucination,
            llm_output=resp.text,
        )
        return action, metric
