"""
Phase 3.75: MSA Rulebook Store (simulated).

Poor-man's MSA: chunk a static rulebook by section headings, embed each
chunk once, expose a Top-k routing query over mean-pooled section keys.

This is the research contribution the Tri-Mem paper needs *today*,
standing in for EverMind's "Coming Soon" sparse-attention layer:

  - Offline: load rulebook markdown → split on `## ` headings → one chunk
    per section → embed each chunk with Chroma's default encoder → store
    chunk text, section title, chunk id, and routing key (the embedding)
    entirely in CPU RAM.
  - Online: project the live query to the same embedding space → cosine
    Top-k against every stored key → return the Top-k (title, text) pairs
    to the caller, which injects them into the turn's user message.

When EverMind's real MSA lands, this file is the drop-in seam: replace
``query()`` with a call into the sparse-attention layer and leave every
caller untouched. The rulebook corpus, the agent, and the benchmark
runner never need to know.

The static rulebook system prompt itself is kept byte-identical across
turns so that vLLM's automatic prefix caching holds the rulebook KV in
GPU memory and the routing result goes into the dynamic (uncached) tail
of the prompt — this is what "path 1" of the 1+2 plan buys us.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass

import chromadb
from chromadb.utils import embedding_functions

from configs.settings import (
    MSA_RULEBOOK_PATH,
    MSA_COLLECTION,
    MSA_TOP_K,
)


@dataclass
class RulebookChunk:
    chunk_id: str
    title: str
    text: str


def _load_rulebook(path: str) -> str:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"MSA rulebook not found at {path}. "
            "Set MSA_RULEBOOK_PATH in configs/settings.py."
        )
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _chunk_by_section(markdown: str) -> list[RulebookChunk]:
    """Split a markdown rulebook on `## ` top-level section headings.

    Anything before the first `## ` is treated as a preamble chunk so no
    content is lost. Each emitted chunk is one full section: heading +
    body text. We intentionally chunk at the SOP granularity (not at
    paragraph granularity) because the whole point of MSA is to route to
    a coherent rulebook section, not to a fragment of one.
    """
    lines = markdown.splitlines()
    chunks: list[RulebookChunk] = []
    current_title: str | None = None
    current_body: list[str] = []
    preamble: list[str] = []

    def flush():
        nonlocal current_title, current_body
        if current_title is None:
            return
        body_text = "\n".join(current_body).strip()
        if body_text:
            chunk = RulebookChunk(
                chunk_id=f"chunk_{len(chunks):02d}",
                title=current_title.strip(),
                text=f"## {current_title.strip()}\n\n{body_text}",
            )
            chunks.append(chunk)
        current_title = None
        current_body = []

    for line in lines:
        if line.startswith("## "):
            flush()
            current_title = line[3:]
            current_body = []
        elif current_title is None:
            preamble.append(line)
        else:
            current_body.append(line)
    flush()

    preamble_text = "\n".join(preamble).strip()
    if preamble_text:
        chunks.insert(
            0,
            RulebookChunk(
                chunk_id="chunk_preamble",
                title="Preamble",
                text=preamble_text,
            ),
        )
    return chunks


class MSAStore:
    """Simulated MSA layer over a static rulebook.

    Routes a live query to the Top-k most relevant rulebook sections.
    The real EverMind MSA would return compressed KV states here; we
    return chunk text because we're sitting on top of a stock LLM and
    letting prefix caching + targeted context injection do the work.
    """

    _shared: "MSAStore | None" = None

    def __init__(self, rulebook_path: str = MSA_RULEBOOK_PATH):
        self.rulebook_path = rulebook_path
        self.chunks: list[RulebookChunk] = _chunk_by_section(
            _load_rulebook(rulebook_path)
        )
        if not self.chunks:
            raise ValueError(f"Rulebook at {rulebook_path} produced zero chunks.")

        self.client = chromadb.Client()
        try:
            self.client.delete_collection(MSA_COLLECTION)
        except Exception:
            pass
        self.collection = self.client.create_collection(
            name=MSA_COLLECTION,
            metadata={"hnsw:space": "cosine"},
            embedding_function=embedding_functions.DefaultEmbeddingFunction(),
        )
        self.collection.add(
            ids=[c.chunk_id for c in self.chunks],
            documents=[c.text for c in self.chunks],
            metadatas=[{"title": c.title} for c in self.chunks],
        )
        print(
            f"[MSA] Rulebook loaded: {len(self.chunks)} chunks from "
            f"{os.path.basename(rulebook_path)}",
            flush=True,
        )

    @classmethod
    def shared(cls) -> "MSAStore":
        """Process-wide singleton — the rulebook never changes across tasks."""
        if cls._shared is None:
            cls._shared = cls()
        return cls._shared

    def query(self, question: str, top_k: int = MSA_TOP_K) -> list[RulebookChunk]:
        """Route a live query to Top-k rulebook sections."""
        k = min(top_k, len(self.chunks))
        result = self.collection.query(query_texts=[question], n_results=k)
        ids = result["ids"][0] if result["ids"] else []
        by_id = {c.chunk_id: c for c in self.chunks}
        return [by_id[i] for i in ids if i in by_id]

    @property
    def full_rulebook_text(self) -> str:
        """Concatenated rulebook — used by the frozen system prompt so
        vLLM's prefix cache holds the rulebook KV once per process.
        Identical bytes across every turn is the whole point.
        """
        return "\n\n".join(c.text for c in self.chunks)

    @staticmethod
    def format_routed_chunks(chunks: list[RulebookChunk]) -> str:
        """Render routed chunks for injection into the user message."""
        if not chunks:
            return ""
        header = "ROUTED POLICY CONTEXT (most relevant rulebook sections for this turn):"
        body = "\n\n---\n\n".join(c.text for c in chunks)
        return f"{header}\n\n{body}"
