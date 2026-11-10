"""Semantic retrieval and context formatting module.

Orchestrates query embedding, vector search against VectorStore, and formats
evidence into structured citations and LLM context blocks.
"""

from typing import Any, Dict, List, Optional
try:
    from .embedder import Embedder, get_default_embedder
    from .vector_store import VectorStore
except ImportError:
    from embedder import Embedder, get_default_embedder
    from vector_store import VectorStore


class Retriever:
    """Orchestrates query embedding, semantic search, and context assembly."""

    def __init__(
        self,
        vector_store: Optional[VectorStore] = None,
        embedder: Optional[Embedder] = None,
    ):
        """Initialize Retriever.

        Args:
            vector_store: VectorStore instance (creates default if None).
            embedder: Embedder instance (uses default if None).
        """
        self.vector_store = vector_store or VectorStore()
        self.embedder = embedder or get_default_embedder()

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        course_filter: Optional[str] = None,
        lecture_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieve top-k semantically relevant chunks for a given query.

        Args:
            query: User's natural language question or topic.
            top_k: Number of nearest chunks to retrieve.
            course_filter: Optional filter by course metadata.
            lecture_filter: Optional filter by lecture metadata.

        Returns:
            List of result dicts sorted by relevance (lowest distance first).
        """
        cleaned_query = query.strip()
        if not cleaned_query:
            return []

        if self.vector_store.count() == 0:
            return []

        where_clause = {}
        if course_filter:
            where_clause["course"] = course_filter
        if lecture_filter:
            where_clause["lecture"] = lecture_filter
        where = where_clause if where_clause else None

        query_vec = self.embedder.embed_query(cleaned_query)
        raw_results = self.vector_store.query(
            query_embedding=query_vec,
            top_k=top_k,
            where=where,
        )

        results: List[Dict[str, Any]] = []

        ids = raw_results.get("ids", [[]])[0]
        documents = raw_results.get("documents", [[]])[0]
        metadatas = raw_results.get("metadatas", [[]])[0]
        distances = raw_results.get("distances", [[]])[0]

        for chunk_id, doc, meta, dist in zip(ids, documents, metadatas, distances):
            similarity = round(1.0 - dist, 4) if dist is not None else None
            results.append(
                {
                    "id": chunk_id,
                    "text": doc,
                    "metadata": meta,
                    "distance": round(dist, 4) if dist is not None else None,
                    "similarity": similarity,
                }
            )

        return results

    @staticmethod
    def format_context_for_llm(results: List[Dict[str, Any]]) -> str:
        """Format retrieved chunks into a clean prompt-ready context block.

        Args:
            results: List of retrieved chunk dictionaries.

        Returns:
            Formatted context string with clean source citations.
        """
        if not results:
            return "No relevant context found."

        blocks = []
        for i, item in enumerate(results, 1):
            meta = item.get("metadata", {})
            source = meta.get("source", "unknown")
            page = meta.get("page", "?")
            chunk_num = meta.get("chunk", "?")
            dist = item.get("distance", "N/A")

            header = f"[Evidence {i} | Source: {source} | Page: {page} | Chunk: {chunk_num} | Cosine Dist: {dist}]"
            # Ensure text is cleanly stripped
            clean_body = item["text"].strip()
            blocks.append(f"{header}\n{clean_body}")

        return "\n\n".join(blocks)
