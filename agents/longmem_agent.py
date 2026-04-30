"""
Tri-Mem agent adapter for LongMemEval / LoCoMo.

The NovaCorp ``act(observation, turn)`` shape doesn't fit single-shot
memory Q&A: there's no environment loop, the haystack is per-task, and
the agent emits a free-text answer rather than a tool command. This
class exposes a different surface that the LongMemEval harness drives:

    agent.reset()
    agent.ingest(sessions, session_ids, dates)
    answer = agent.answer(question, question_date)

Memory layers reused from the NovaCorp stack:

    SessionMSA  — per-task semantic store over haystack sessions
                  (replaces the static-rulebook MSAStore)
    VisualBus   — compresses the session timeline (text-in → render →
                  GLM-OCR → text-out, same as today)
    RAGStore    — exact-fact lookup (dates, names, numbers in sessions)

The entropy router is wired the same way: a 1-token probe with no
memory injected gives a Shannon entropy score; ``router.entropy.route``
maps it to a band that gates which memory blocks get assembled into
the answer-pass prompt.

NOTE: VisualBus currently caps at MAX_VISUAL_TILES=4 sessions. That's
fine for short-haystack tasks but loses information on the medium / s
splits. Plumbing a per-task max_turns through VisualBus.compress() is
left for a follow-up so we can ship the harness end-to-end first.
"""
from __future__ import annotations

import os
import re

from memory.rag_store import RAGStore
from memory.session_msa import SessionMSA
from memory.visual_bus import VisualBus
from router.entropy import route, RouteDecision
from utils.llm import get_llm
from configs.settings import (
    MSA_TOP_K,
    ROUTER_ENABLED,
    ROUTER_PROBE_TOP_K,
)


_SYSTEM_PROMPT = """\
You are a conversational memory assistant. The user has had many prior
chat sessions with you over time; relevant excerpts are provided below
in tagged blocks ([ROUTED SESSIONS], [COMPRESSED HISTORY], [EXACT FACTS]).
Use them to answer the user's current question.

Output rules — STRICT:
- Reply with the final answer ONLY. No preamble, no "Thinking Process",
  no step-by-step explanation, no analysis sections, no markdown headers,
  no restating of the question.
- For factual questions, give just the fact (e.g. "Tomatoes",
  "Samsung Galaxy S22", "14 days").
- If the answer is not derivable from the blocks, reply exactly:
  I don't know.
- Trust the provided memory blocks over your own recall.
"""


_PROBE_SYSTEM = """\
You are a conversational memory assistant. Answer the user's question
in one short phrase. No preamble.
"""


def _strip_think(text: str) -> str:
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[-1]
    return text.strip()


class LongMemAgent:
    """Tri-Mem applied to single-shot conversational-memory Q&A."""

    name = "trimem_longmem"

    def __init__(self):
        self.llm = get_llm()
        self.rag = RAGStore()
        self.visual_bus = VisualBus()
        self.session_msa: SessionMSA | None = None
        self._visual_history: list[dict] = []
        self._ingested = False
        # Retained only for LONGMEM_DEBUG=1 introspection. Negligible
        # memory cost (a few KB of strings per task) and unused otherwise.
        self._debug_session_ids: list[str] = []
        self._debug_dates: list[str] = []
        self._debug_session_texts: list[str] = []

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def reset(self) -> None:
        if self.session_msa is not None:
            self.session_msa.teardown()
        self.session_msa = SessionMSA()
        self.rag.reset()
        self.visual_bus.reset("longmem")
        self._visual_history = []
        self._ingested = False
        self._debug_session_ids = []
        self._debug_dates = []
        self._debug_session_texts = []

    # ------------------------------------------------------------------
    # Ingest
    # ------------------------------------------------------------------

    def ingest(
        self,
        sessions: list[list[dict]],
        session_ids: list[str],
        dates: list[str],
    ) -> None:
        """Populate per-task memories from the haystack."""
        assert self.session_msa is not None, "call reset() before ingest()"

        # SessionMSA — one chunk per session, embedded for Top-k routing.
        self.session_msa.ingest_sessions(sessions, session_ids, dates)

        if os.environ.get("LONGMEM_DEBUG"):
            self._debug_session_ids = list(session_ids)
            self._debug_dates = list(dates)
            self._debug_session_texts = [
                self._render_session(s, d, sid)
                for s, d, sid in zip(sessions, dates, session_ids)
            ]

        # RAG — store each session as a fact so the high-entropy band
        # can pull verbatim text. Also store individual high-signal turns
        # (assistant-stated facts about the user) at finer granularity.
        for sid, date, session in zip(session_ids, dates, sessions):
            session_text = self._render_session(session, date, sid)
            self.rag.store_fact(
                session_text,
                {"session_id": sid, "date": date, "type": "session"},
            )
            for i, msg in enumerate(session):
                content = (msg.get("content") or "").strip()
                if not content:
                    continue
                # Keep individual turns retrievable too — short enough
                # to inject losslessly into the answer prompt.
                self.rag.store_fact(
                    f"[{date} · {sid} · turn {i} · {msg.get('role','user')}] "
                    f"{content[:500]}",
                    {
                        "session_id": sid,
                        "date": date,
                        "turn": i,
                        "role": msg.get("role", "user"),
                        "type": "turn",
                    },
                )

        # Visual Bus — feed sessions as a synthetic alternating history
        # so render_history's user/assistant colouring still works. The
        # MAX_VISUAL_TILES cap means only the most recent few sessions
        # show up; see module docstring TODO.
        for sid, date, session in zip(session_ids, dates, sessions):
            summary = self._summarize_session(session)
            self._visual_history.append(
                {"role": "user", "content": f"[{date} · {sid}] {summary}"}
            )
            self._visual_history.append(
                {"role": "assistant", "content": "(session end)"}
            )

        self._ingested = True

    @staticmethod
    def _render_session(session: list[dict], date: str, sid: str) -> str:
        lines = [f"[Session {sid} · {date}]"]
        for msg in session:
            role = msg.get("role", "user").upper()
            content = (msg.get("content") or "").strip()
            if content:
                lines.append(f"{role}: {content}")
        return "\n".join(lines)

    @staticmethod
    def _summarize_session(session: list[dict], max_chars: int = 400) -> str:
        """Cheap session summary: concatenate user turns, truncate.

        We deliberately avoid an LLM-call summarizer here so ingest is
        fast and deterministic. The Visual Bus's own GLM-OCR pass is the
        compression step; this just produces the visible text it OCRs.
        """
        user_turns = [
            (m.get("content") or "").strip()
            for m in session
            if m.get("role") == "user"
        ]
        joined = " | ".join(t for t in user_turns if t)
        if len(joined) > max_chars:
            joined = joined[: max_chars - 3] + "..."
        return joined or "(empty session)"

    # ------------------------------------------------------------------
    # Probe + route
    # ------------------------------------------------------------------

    def _probe(self, question: str, question_date: str) -> float:
        probe_user = (
            f"Date: {question_date or '(unknown)'}\n"
            f"Question: {question}\n\n"
            f"Answer:"
        )
        resp = self.llm.chat(
            system=_PROBE_SYSTEM,
            messages=[{"role": "user", "content": probe_user}],
            max_tokens=1,
            temperature=0.0,
            logprobs=ROUTER_PROBE_TOP_K,
        )
        return resp.first_token_entropy

    # ------------------------------------------------------------------
    # Memory assembly
    # ------------------------------------------------------------------

    def _build_memory_blocks(
        self,
        decision: RouteDecision,
        question: str,
        question_date: str,
    ) -> list[str]:
        blocks: list[str] = []
        debug = bool(os.environ.get("LONGMEM_DEBUG"))

        if decision.use_msa_routed_chunks and self.session_msa is not None:
            chunks = self.session_msa.query(question, top_k=MSA_TOP_K)
            routed = SessionMSA.format_chunks(chunks)
            if routed:
                blocks.append(routed)
            if debug:
                routed_ids = [c.session_id for c in chunks]
                print(f"[DEBUG] MSA routed (top_k={MSA_TOP_K}): {routed_ids}", flush=True)

        if decision.use_visual_bus and self._visual_history:
            compressed = self.visual_bus.compress(self._visual_history)
            if compressed:
                blocks.append(f"[COMPRESSED HISTORY]\n{compressed}")

        if decision.use_rag:
            queries = [question]
            if question_date:
                queries.append(question_date)
            # If we have a Visual Bus block, mine it for entities to drive
            # targeted lookups (the Phase 3.5 trick).
            vbus_block = next(
                (b for b in blocks if b.startswith("[COMPRESSED HISTORY]")),
                "",
            )
            if vbus_block:
                queries.extend(RAGStore.extract_entities(vbus_block)[:8])
            facts = self.rag.query_multi(queries, top_k=2)
            if facts:
                facts = facts[:10]
                blocks.append(
                    "[EXACT FACTS]\n" + "\n".join(f"- {f}" for f in facts)
                )
            if debug:
                print(
                    f"[DEBUG] RAG queries={queries[:5]}{'...' if len(queries) > 5 else ''}",
                    flush=True,
                )
                for i, f in enumerate(facts[:5] if facts else []):
                    print(f"[DEBUG] RAG fact[{i}]: {f[:200]!r}", flush=True)

        return blocks

    # ------------------------------------------------------------------
    # Answer
    # ------------------------------------------------------------------

    def answer(self, question: str, question_date: str = "") -> str:
        if not self._ingested:
            raise RuntimeError("ingest(...) must be called before answer(...)")

        if ROUTER_ENABLED:
            entropy = self._probe(question, question_date)
        else:
            entropy = -1.0  # forces full injection

        decision = route(entropy)

        if os.environ.get("LONGMEM_DEBUG"):
            print(f"[DEBUG] question: {question!r}", flush=True)
            print(f"[DEBUG] question_date: {question_date!r}", flush=True)
            print(
                f"[DEBUG] entropy={entropy:.4f}  decision="
                f"msa={decision.use_msa_routed_chunks} "
                f"vbus={decision.use_visual_bus} "
                f"rag={decision.use_rag}",
                flush=True,
            )
            print(
                f"[DEBUG] haystack ({len(self._debug_session_ids)} sessions): "
                f"{list(zip(self._debug_session_ids, self._debug_dates))}",
                flush=True,
            )
            grep = os.environ.get("LONGMEM_DEBUG_GREP", "").strip()
            if grep:
                needles = [n.strip().lower() for n in grep.split(",") if n.strip()]
                hits = []
                for sid, date, text in zip(
                    self._debug_session_ids,
                    self._debug_dates,
                    self._debug_session_texts,
                ):
                    lt = text.lower()
                    if any(n in lt for n in needles):
                        hits.append((sid, date))
                print(
                    f"[DEBUG] grep {needles!r} matched {len(hits)} sessions: {hits}",
                    flush=True,
                )

        blocks = self._build_memory_blocks(decision, question, question_date)

        date_line = f"Today's date: {question_date}\n" if question_date else ""
        user_msg_parts = list(blocks)
        user_msg_parts.append(f"{date_line}Question: {question}")
        user_msg = "\n\n".join(user_msg_parts)

        resp = self.llm.chat(
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_msg}],
        )
        return _strip_think(resp.text)
