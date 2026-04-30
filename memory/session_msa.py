"""
Per-task semantic memory for conversational benchmarks (LongMemEval, LoCoMo).

Mirrors MSAStore's Top-k chunked-retrieval API but is constructed per task
from a list of chat sessions rather than from a static rulebook on disk.
Each session is rendered into a single chunk (timestamp-prefixed) and
embedded with Chroma's default sentence-transformer encoder.

Why a separate class: MSAStore is bound to MSA_RULEBOOK_PATH and uses a
process-global Chroma collection (MSA_COLLECTION) so vLLM's prefix cache
can hold the rulebook KV across turns. LongMemEval's "rulebook" is a
different haystack per task, so we want per-instance lifetimes and a
fresh collection per task.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass

import chromadb
from chromadb.utils import embedding_functions


@dataclass
class SessionChunk:
    chunk_id: str
    session_id: str
    date: str
    text: str


def _render_session(session: list[dict], date: str, session_id: str) -> str:
    """Flatten a session (list of {role, content}) into a single string.

    Date prefix lets temporal-reasoning queries hit on the date itself.
    """
    lines = [f"[Session {session_id} · {date}]"]
    for msg in session:
        role = msg.get("role", "user").upper()
        content = (msg.get("content") or "").strip()
        if content:
            lines.append(f"{role}: {content}")
    return "\n".join(lines)


class SessionMSA:
    """Per-task semantic store over chat sessions."""

    def __init__(self, collection_prefix: str = "longmem_msa"):
        self.client = chromadb.Client()
        self._collection_name = f"{collection_prefix}_{uuid.uuid4().hex[:8]}"
        self._embedder = embedding_functions.DefaultEmbeddingFunction()
        self.collection = self.client.create_collection(
            name=self._collection_name,
            embedding_function=self._embedder,
            metadata={"hnsw:space": "cosine"},
        )
        self._chunks: list[SessionChunk] = []

    def ingest_sessions(
        self,
        sessions: list[list[dict]],
        session_ids: list[str] | None = None,
        dates: list[str] | None = None,
    ) -> int:
        """Ingest a list of sessions. Returns chunk count."""
        n = len(sessions)
        session_ids = session_ids or [f"s{i}" for i in range(n)]
        dates = dates or [""] * n

        docs, ids, metas = [], [], []
        for sid, date, session in zip(session_ids, dates, sessions):
            text = _render_session(session, date, sid)
            chunk_id = f"chunk_{len(self._chunks)}"
            self._chunks.append(SessionChunk(chunk_id, sid, date, text))
            docs.append(text)
            ids.append(chunk_id)
            metas.append({"session_id": sid, "date": date})

        if docs:
            self.collection.add(documents=docs, ids=ids, metadatas=metas)
        return len(docs)

    def query(self, text: str, top_k: int = 3) -> list[SessionChunk]:
        if not self._chunks:
            return []
        k = min(top_k, len(self._chunks))
        res = self.collection.query(query_texts=[text], n_results=k)
        ids = res.get("ids", [[]])[0]
        out = []
        by_id = {c.chunk_id: c for c in self._chunks}
        for cid in ids:
            chunk = by_id.get(cid)
            if chunk is not None:
                out.append(chunk)
        return out

    def teardown(self):
        """Drop the per-task collection. Safe to call multiple times."""
        try:
            self.client.delete_collection(self._collection_name)
        except Exception:
            pass
        self._chunks = []

    @staticmethod
    def format_chunks(chunks: list[SessionChunk]) -> str:
        if not chunks:
            return ""
        parts = ["[ROUTED SESSIONS]"]
        for c in chunks:
            parts.append(c.text)
        return "\n\n".join(parts)
