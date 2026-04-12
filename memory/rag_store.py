"""
Phase 2: RAG Memory Store.

Stores exact facts (object IDs, locations, syntax patterns) in a
vector database. The agent queries this when it needs an exact string
rather than relying on its context window.
"""
import json
import chromadb
from configs.settings import RAG_COLLECTION, RAG_TOP_K


class RAGStore:
    """ChromaDB-backed fact store for exact retrieval."""

    def __init__(self):
        self.client = chromadb.Client()  # in-memory for now
        self.collection = self.client.get_or_create_collection(
            name=RAG_COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )
        self._id_counter = 0

    def reset(self):
        """Clear all facts for a new task."""
        self.client.delete_collection(RAG_COLLECTION)
        self.collection = self.client.get_or_create_collection(
            name=RAG_COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )
        self._id_counter = 0

    def store_fact(self, text: str, metadata: dict = None):
        """Store an exact fact (e.g., 'apple 1 is in fridge 1')."""
        self._id_counter += 1
        self.collection.add(
            documents=[text],
            ids=[f"fact_{self._id_counter}"],
            metadatas=[metadata or {}],
        )

    def store_observation(self, turn: int, location: str, observation: str):
        """Parse and store facts from an environment observation."""
        # Store the raw observation with context
        self.store_fact(
            f"Turn {turn} at {location}: {observation}",
            {"turn": turn, "location": location, "type": "observation"},
        )

        # Extract object mentions (e.g., "apple 1", "mug 1")
        # Simple heuristic: look for "word number" patterns
        import re
        objects = re.findall(r'([a-z]+(?:\s[a-z]+)?\s\d+)', observation.lower())
        for obj in objects:
            self.store_fact(
                f"{obj} was seen at {location} on turn {turn}",
                {"turn": turn, "location": location, "type": "object_sighting", "object": obj},
            )

    def query(self, question: str, top_k: int = None) -> list[str]:
        """Retrieve the most relevant facts for a question."""
        k = top_k or RAG_TOP_K
        if self.collection.count() == 0:
            return []
        results = self.collection.query(
            query_texts=[question],
            n_results=min(k, self.collection.count()),
        )
        return results["documents"][0] if results["documents"] else []

    def query_multi(self, queries: list[str], top_k: int = None) -> list[str]:
        """Run multiple targeted queries and return de-duplicated results.

        Useful when the Visual Bus summary mentions several entities whose
        exact IDs may have been blurred by OCR compression.  Each query
        retrieves its own top-k; results are merged in retrieval order with
        duplicates removed.
        """
        if not queries or self.collection.count() == 0:
            return []
        seen: set[str] = set()
        merged: list[str] = []
        k = top_k or RAG_TOP_K
        for q in queries:
            for doc in self.query(q, top_k=k):
                if doc not in seen:
                    seen.add(doc)
                    merged.append(doc)
        return merged

    @staticmethod
    def extract_entities(text: str) -> list[str]:
        """Pull entity-like mentions from text (e.g. OCR-compressed summary).

        Targets the kinds of identifiers that visual compression blurs:
          - record/system IDs:  invoice_1, token_3, procurement_db
          - alphanumeric keys:  sk-NvC-4f8a2b1c, LIC-2024-NVC-00847
          - word-number pairs:  "vendor 3", "invoice 1"
        Returns a list of unique mention strings.
        """
        import re
        entities: list[str] = []
        seen: set[str] = set()

        # underscore-joined identifiers: invoice_1, procurement_db, credential_vault
        for m in re.findall(r'[a-zA-Z][a-zA-Z0-9]*(?:_[a-zA-Z0-9]+)+', text):
            low = m.lower()
            if low not in seen:
                seen.add(low)
                entities.append(m)

        # hyphen-joined keys: sk-NvC-4f8a2b1c, LIC-2024-NVC-00847
        for m in re.findall(r'[a-zA-Z][a-zA-Z0-9]*(?:-[a-zA-Z0-9]+){2,}', text):
            low = m.lower()
            if low not in seen:
                seen.add(low)
                entities.append(m)

        # word-number pairs: "invoice 1", "token 3" (space, not newline)
        # Exclude noise like "turn 1", "step 2", "on turn 12"
        _noise = {"turn", "step", "line", "page", "item", "on", "at", "in", "to", "of"}
        for m in re.findall(r'[a-z]+(?:[ ][a-z]+)?[ ]\d+', text.lower()):
            words = m.split()
            if not any(w in _noise for w in words[:-1]) and m not in seen:
                seen.add(m)
                entities.append(m)

        return entities
