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
