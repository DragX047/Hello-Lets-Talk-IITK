"""Vector storage module using persistent ChromaDB.

Manages persistent on-disk vector index at storage/chroma with deterministic
chunk IDs, metadata tracking, and top-k vector similarity queries.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import chromadb
from chromadb.config import Settings
import numpy as np


class VectorStore:
    """Manages persistent ChromaDB collection for lecture chunks."""

    def __init__(
        self,
        persist_dir: str = "storage/chroma",
        collection_name: str = "hello_lets_talk_iitk_chunks_v1",
    ):
        """Initialize ChromaDB client and retrieve or create collection.

        Args:
            persist_dir: Relative or absolute path to persistent database folder.
            collection_name: Name of the Chroma collection.
        """
        self.persist_dir = str(Path(persist_dir).resolve())
        self.collection_name = collection_name
        Path(self.persist_dir).mkdir(parents=True, exist_ok=True)
        self._reconnect()

    def _reconnect(self) -> None:
        """Initialize or refresh persistent ChromaDB client and collection handle."""
        try:
          if hasattr(self, "client") and hasattr(self.client, "close"):
            self.client.close()
        except Exception:
          pass

        try:
          from chromadb.api.client import SharedSystemClient

          SharedSystemClient.clear_system_cache()
        except Exception:
          pass

        self.client = chromadb.PersistentClient(
            path=self.persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )
        # Migrate legacy collection if present
        existing_cols = [c.name for c in self.client.list_collections()]
        if self.collection_name not in existing_cols and "zombai_lecture_chunks_v1" in existing_cols:
            try:
                legacy_col = self.client.get_collection("zombai_lecture_chunks_v1")
                legacy_col.modify(name=self.collection_name)
            except Exception:
                pass

        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    @staticmethod
    def generate_chunk_id(chunk: Dict[str, Any]) -> str:
        """Create a deterministic unique ID for a chunk.

        Format: {course}_{lecture}_{source}_p{page}_c{chunk}
        """
        meta = chunk["metadata"]
        course = str(meta.get("course", "course")).replace(" ", "_")
        lecture = str(meta.get("lecture", "lec")).replace(" ", "_")
        source = Path(str(meta.get("source", "doc"))).stem.replace(" ", "_")
        page = meta.get("page", 0)
        c_num = meta.get("chunk", 0)

        return f"{course}_{lecture}_{source}_p{page}_c{c_num}"

    def add_chunks(
        self,
        chunks: List[Dict[str, Any]],
        embeddings: Union[np.ndarray, List[List[float]]],
    ) -> int:
        """Upsert chunks with precomputed embeddings into ChromaDB.

        Args:
            chunks: List of chunk dictionaries with 'text' and 'metadata'.
            embeddings: 2D array/list of vector embeddings (matching chunks).

        Returns:
            Number of chunks upserted.
        """
        if not chunks:
            return 0

        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Count mismatch: {len(chunks)} chunks vs {len(embeddings)} embeddings."
            )

        ids = [self.generate_chunk_id(c) for c in chunks]
        documents = [c["text"] for c in chunks]

        # Ensure all metadata values are Chroma-supported primitive types (str, int, float, bool)
        metadatas = []
        for c in chunks:
            clean_meta = {}
            for k, v in c["metadata"].items():
                if isinstance(v, (str, int, float, bool)):
                    clean_meta[k] = v
                else:
                    clean_meta[k] = str(v)
            metadatas.append(clean_meta)

        if isinstance(embeddings, np.ndarray):
            embeddings_list = embeddings.tolist()
        else:
            embeddings_list = embeddings

        self.collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings_list,
            metadatas=metadatas,
        )

        return len(chunks)

    def query(
        self,
        query_embedding: Union[np.ndarray, List[float]],
        top_k: int = 5,
        where: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Perform nearest-neighbor vector search.

        Args:
            query_embedding: 1D query vector.
            top_k: Number of results to return.
            where: Optional metadata filter dict.

        Returns:
            ChromaDB query results dictionary.
        """
        if isinstance(query_embedding, np.ndarray):
            q_emb = [query_embedding.tolist()]
        elif isinstance(query_embedding[0], (int, float)):
            q_emb = [query_embedding]
        else:
            q_emb = query_embedding

        try:
            results = self.collection.query(
                query_embeddings=q_emb,
                n_results=top_k,
                where=where,
                include=["documents", "metadatas", "distances"],
            )
        except Exception:
            # Stale handle, Rust index mismatch, or concurrent update: reconnect and retry
            self._reconnect()
            results = self.collection.query(
                query_embeddings=q_emb,
                n_results=top_k,
                where=where,
                include=["documents", "metadatas", "distances"],
            )
        return results

    def count(self) -> int:
        """Return total number of chunks currently stored in the collection."""
        try:
            return self.collection.count()
        except Exception:
            self._reconnect()
            return self.collection.count()

    def reset_collection(self) -> None:
        """Clear and recreate the collection."""
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

