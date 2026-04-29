"""
Phase 4: Tri-Mem Agent (entropy-routed three-layer memory).

This is the agent the README has been building toward. Each turn:

  1. PROBE  — one cheap LLM call with no memory injected (just goal +
              current observation + last action). We ask for K logprobs
              on the first generated token; the Shannon entropy of that
              top-K distribution is the model's intrinsic uncertainty.
  2. ROUTE  — `router.entropy.route()` maps the probe entropy to a band:
                low  → MSA only
                med  → MSA + Visual Bus
                high → MSA + Visual Bus + RAG
  3. ACT    — second LLM call with the chosen memory blocks injected.
              The frozen rulebook system prompt (Phase 3.75 path 1) is
              always present so vLLM's prefix cache stays warm; the
              expensive bits — routed MSA chunks, the OCR-compressed
              Visual Bus tile, and targeted RAG facts — are gated by the
              router's decision.

Why two passes? The whole point of entropy routing is to spend tokens
*proportionally to uncertainty*. A single-pass agent that always injects
everything is just Phase 3.5+3.75 stacked, which wastes tokens on easy
turns. The probe is small (1 generated token, prefix-cached prompt), and
on low-entropy turns we skip Visual Bus OCR + RAG entirely — that's
where the win shows up.

Outputs `memory_source` matching the band label so the dashboard's
modality breakdown reflects what the router actually picked.
"""
from __future__ import annotations

import re

from agents.base_agent import BaseAgent
from memory.msa_store import MSAStore
from memory.rag_store import RAGStore
from memory.visual_bus import VisualBus
from router.entropy import route, RouteDecision
from utils.llm import get_llm
from utils.metrics import TurnMetric
from configs.settings import (
    MSA_TOP_K,
    MSA_INJECT_FULL_RULEBOOK,
    ROUTER_ENABLED,
    ROUTER_PROBE_TOP_K,
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

MEMORY SYSTEM (Tri-Mem):
Your authoritative NovaCorp IT Policy & Procurement Guide is preloaded
into your semantic memory layer (MSA) below. On each turn, an entropy
router decides whether to additionally surface:
  - ROUTED POLICY CONTEXT  — Top-k policy sections relevant to this turn
  - COMPRESSED HISTORY     — visual summary of prior steps (Visual Bus)
  - EXACT FACTS            — exact alphanumeric IDs and outcomes (RAG)
When any of those blocks is present in the user message, trust them over
your own recollection.

=== BEGIN NOVACORP IT POLICY (SEMANTIC MEMORY) ===

{rulebook}

=== END NOVACORP IT POLICY ===
"""


_PROBE_SYSTEM = """\
You are an expert IT auditor operating in the NovaCorp corporate network.
Issue exactly one terminal command. Output only the command, nothing else.
Valid verbs: access, query, download, upload, revoke, scan, run.
"""


class TriMemAgent(BaseAgent):
    """Probe → route → generate. All three memory layers, gated by entropy."""

    name = "trimem"

    def __init__(self):
        self.llm = get_llm()
        self.msa = MSAStore.shared()
        self.visual_bus = VisualBus()
        self.rag = RAGStore()

        self.history: list[dict] = []          # full chat history for the act-pass
        self._raw_history: list[dict] = []     # obs/action pairs for visual rendering
        self.goal = ""
        self.last_action = ""
        self.current_location = "unknown"
        self._seen_entities: set[str] = set()

        self._system_prompt = self._build_system_prompt()
        self._init_loop_guard()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _build_system_prompt(self) -> str:
        """Frozen rulebook prompt — must be byte-identical across calls
        so vLLM's prefix cache holds the rulebook KV once per process."""
        if MSA_INJECT_FULL_RULEBOOK:
            return _STATIC_PREAMBLE.format(rulebook=self.msa.full_rulebook_text)
        return _STATIC_PREAMBLE.format(rulebook="(rulebook access disabled)")

    def reset(self, goal: str):
        self.history = []
        self._raw_history = []
        self.goal = goal
        self.last_action = ""
        self.current_location = "unknown"
        self._seen_entities = set()
        self.rag.reset()
        self._init_loop_guard()
        task_slug = goal[:40].replace(" ", "_")
        self.visual_bus.reset(task_slug)

    # ------------------------------------------------------------------
    # Probe + route
    # ------------------------------------------------------------------

    def _probe(self, observation: str) -> tuple[float, int, int, float]:
        """One-token probe with no memory injected. Returns
        (entropy_bits, tokens_in, tokens_out, latency_ms)."""
        probe_user = (
            f"Task: {self.goal}\n"
            f"Last action: {self.last_action or '(none)'}\n"
            f"Current observation: {observation}\n\n"
            f"Next command:"
        )
        resp = self.llm.chat(
            system=_PROBE_SYSTEM,
            messages=[{"role": "user", "content": probe_user}],
            max_tokens=1,
            temperature=0.0,
            logprobs=ROUTER_PROBE_TOP_K,
        )
        return resp.first_token_entropy, resp.tokens_in, resp.tokens_out, resp.latency_ms

    # ------------------------------------------------------------------
    # Memory assembly
    # ------------------------------------------------------------------

    def _update_seen_entities(self, observation: str, succeeded: bool, turn: int):
        if turn == 0 or succeeded:
            for e in RAGStore.extract_entities(observation):
                self._seen_entities.add(e.lower())
            if turn == 0:
                m = re.search(r'available systems[^:]*:\s*([^.]+)', observation.lower())
                if m:
                    for s in m.group(1).split(","):
                        self._seen_entities.add(s.strip().strip("."))

    def _is_spatial_hallucination(self, action: str) -> bool:
        if not self._seen_entities:
            return False
        targets = set(re.findall(r'[a-z][a-z0-9]*(?:_[a-z0-9]+)+', action.lower()))
        targets |= set(re.findall(r'(?<![a-z])[a-z]+[ ]\d+(?!\d)', action.lower()))
        return bool(targets - self._seen_entities)

    def _store_observation(self, observation: str, succeeded: bool, turn: int):
        """Mirror VisualBusRAGAgent's bookkeeping: store every observation
        and the previous action's outcome in RAG so the high-entropy band
        has something useful to retrieve."""
        obs_turn = turn - 1 if turn > 0 else 0
        self.rag.store_observation(obs_turn, self.current_location, observation)
        if self.last_action and turn > 0:
            outcome = "succeeded" if succeeded else "FAILED"
            self.rag.store_fact(
                f"Turn {turn - 1}: '{self.last_action}' {outcome}. "
                f"Response: {observation[:200]}",
                {
                    "turn": turn - 1,
                    "location": self.current_location,
                    "type": "action_outcome",
                    "success": succeeded,
                },
            )

    def _build_memory_blocks(
        self,
        decision: RouteDecision,
        observation: str,
        turn: int,
    ) -> list[str]:
        """Return a list of memory context blocks selected by the router."""
        blocks: list[str] = []

        # MSA routed Top-k chunks (Phase 3.75 path 2)
        if decision.use_msa_routed_chunks:
            query_text = (
                f"Task: {self.goal}\n"
                f"Current observation: {observation}\n"
                f"Last action: {self.last_action or '(none)'}"
            )
            chunks = self.msa.query(query_text, top_k=MSA_TOP_K)
            routed = MSAStore.format_routed_chunks(chunks)
            if routed:
                blocks.append(routed)

        # Visual Bus compressed timeline (Phase 3)
        if decision.use_visual_bus and self._raw_history:
            compressed = self.visual_bus.compress(self._raw_history)
            if compressed:
                blocks.append(f"[COMPRESSED HISTORY]\n{compressed}")

        # RAG targeted lookups (Phase 3.5 — summary-guided)
        if decision.use_rag and turn > 0:
            queries: list[str] = []
            # If we have a Visual Bus block, use entities mentioned in it
            # to drive targeted lookups (the Phase 3.5 trick).
            vbus_block = next(
                (b for b in blocks if b.startswith("[COMPRESSED HISTORY]")),
                "",
            )
            if vbus_block:
                queries.extend(RAGStore.extract_entities(vbus_block)[:8])
            queries.append(self.goal)
            queries.append(observation[:200])
            facts = self.rag.query_multi(queries, top_k=2)
            if facts:
                facts = facts[:10]
                blocks.append(
                    "[EXACT FACTS]\n" + "\n".join(f"- {f}" for f in facts)
                )

        return blocks

    # ------------------------------------------------------------------
    # Main act loop
    # ------------------------------------------------------------------

    def act(self, observation: str, turn: int) -> tuple[str, TurnMetric]:
        # Track current system from the previous turn's access command
        if self.last_action.startswith("access "):
            self.current_location = self.last_action[7:]

        # Bookkeeping that has to happen regardless of routing decision
        succeeded = not any(
            s in observation.lower()
            for s in ("syntax error", "returned no results", "access denied")
        )
        self._update_seen_entities(observation, succeeded, turn)
        self._store_observation(observation, succeeded, turn)

        # ── Step 1: probe ──
        probe_tokens_in = 0
        probe_tokens_out = 0
        probe_latency = 0.0
        if ROUTER_ENABLED:
            probe_entropy, probe_tokens_in, probe_tokens_out, probe_latency = (
                self._probe(observation)
            )
        else:
            probe_entropy = -1.0  # forces BAND_HIGH (full injection)

        decision = route(probe_entropy)

        # ── Step 2: build the act-pass user message ──
        memory_blocks = self._build_memory_blocks(decision, observation, turn)

        if turn == 0:
            obs_block = f"Your current task: {self.goal}\n\n{observation}"
        else:
            obs_block = f"[CURRENT OBSERVATION]\n{observation}"

        loop_warn = self._loop_warning()
        parts: list[str] = []
        if memory_blocks:
            parts.extend(memory_blocks)
        parts.append(obs_block)
        if loop_warn:
            parts.append(loop_warn)
        user_msg = "\n\n".join(parts)

        self.history.append({"role": "user", "content": user_msg})

        # ── Step 3: act ──
        resp = self.llm.chat(
            system=self._system_prompt,
            messages=self.history,
        )

        action = self.parse_action(resp.text)
        self.history.append({"role": "assistant", "content": action})

        # Update raw history for next turn's Visual Bus rendering
        self._raw_history.append({"role": "user", "content": observation})
        self._raw_history.append({"role": "assistant", "content": action})

        self._record_outcome(action, succeeded)
        self.last_action = action

        # Token + latency accounting: include the probe pass so the cost
        # metric reflects the true price of routed inference. The probe
        # is small (1 generated token, cache-warm prompt) but it isn't
        # free, and hiding it would understate the agent's overhead.
        syntactic_error = "syntax error" in observation.lower()
        spatial_hallucination = self._is_spatial_hallucination(action)

        metric = TurnMetric(
            turn=turn,
            action=action,
            observation=observation,
            success=False,  # overwritten by the runner after env.step()
            tokens_in=resp.tokens_in + probe_tokens_in,
            tokens_out=resp.tokens_out + probe_tokens_out,
            memory_source=decision.band,
            entropy_score=probe_entropy,
            latency_ms=resp.latency_ms + probe_latency,
            syntactic_error=syntactic_error,
            spatial_hallucination=spatial_hallucination,
            llm_output=resp.text,
        )
        return action, metric
