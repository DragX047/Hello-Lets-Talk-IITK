"""Embedding module for ZombAI V1.

Generates 384-dimensional dense vector embeddings using all-MiniLM-L6-v2.
Supports both SentenceTransformer and ChromaDB's ONNX-based all-MiniLM-L6-v2
embedding function for robust offline and cross-platform execution.
"""

from typing import Any, Dict, List, Optional
import numpy as np


class Embedder:
    """Manages embedding model loading and inference for chunks and queries."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: Optional[str] = None,
    ):
        """Initialize embedding engine with fallback to local ONNX all-MiniLM-L6-v2.

        Args:
            model_name: Hugging Face model identifier.
            device: 'cuda', 'cpu', or None (auto-detect).
        """
        self.model_name = model_name
        self.device = device
        self.dimension = 384
        self._backend = None
        self._st_model = None
        self._chroma_ef = None

        # Attempt to load SentenceTransformer first (trying local cache first for speed);
        # if offline or Hub unreachable, fallback to Chroma's local ONNX all-MiniLM-L6-v2 engine.
        try:
            from sentence_transformers import SentenceTransformer

            try:
                self._st_model = SentenceTransformer(model_name, device=device, local_files_only=True)
            except Exception:
                self._st_model = SentenceTransformer(model_name, device=device)

            self._backend = "sentence_transformers"
            self.dimension = self._st_model.get_sentence_embedding_dimension()
        except Exception:
            try:
                import chromadb.utils.embedding_functions as ef

                self._chroma_ef = ef.DefaultEmbeddingFunction()
                self._backend = "chroma_onnx"
                self.dimension = 384
            except Exception as e:
                raise RuntimeError(
                    f"Failed to initialize embedding backend: {e}"
                )

    @property
    def backend(self) -> str:
        """Return the active backend name."""
        return self._backend or "unknown"

    def embed_chunks(
        self,
        chunks: List[Dict[str, Any]],
        batch_size: int = 32,
        show_progress_bar: bool = False,
    ) -> np.ndarray:
        """Generate dense embeddings for a list of chunk dictionaries.

        Args:
            chunks: List of chunk dictionaries, each containing a 'text' key.
            batch_size: Batch size for encoding.
            show_progress_bar: Whether to display progress bar.

        Returns:
            np.ndarray of shape (num_chunks, 384).
        """
        if not chunks:
            return np.empty((0, self.dimension), dtype=np.float32)

        texts = [chunk["text"] for chunk in chunks]

        if self._backend == "sentence_transformers" and self._st_model is not None:
            embeddings = self._st_model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=show_progress_bar,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
        else:
            raw_embeddings = self._chroma_ef(texts)
            embeddings = np.array(raw_embeddings, dtype=np.float32)

        if not np.all(np.isfinite(embeddings)):
            raise ValueError("Embedding computation yielded non-finite (NaN/Inf) values.")

        if embeddings.shape[1] != self.dimension:
            raise ValueError(
                f"Dimension mismatch: expected {self.dimension}, got {embeddings.shape[1]}"
            )

        return embeddings

    def embed_query(self, query: str) -> np.ndarray:
        """Generate dense embedding for a single user query.

        Args:
            query: Query string.

        Returns:
            np.ndarray 1D vector of shape (384,).
        """
        cleaned_query = query.strip()
        if not cleaned_query:
            raise ValueError("Query string cannot be empty.")

        if self._backend == "sentence_transformers" and self._st_model is not None:
            vec = self._st_model.encode(
                cleaned_query,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
        else:
            raw = self._chroma_ef([cleaned_query])[0]
            vec = np.array(raw, dtype=np.float32)

        if not np.all(np.isfinite(vec)):
            raise ValueError("Query embedding generated non-finite (NaN/Inf) values.")

        return vec


# Module-level singleton
_DEFAULT_EMBEDDER: Optional[Embedder] = None


def get_default_embedder() -> Embedder:
    """Retrieve or instantiate singleton Embedder."""
    global _DEFAULT_EMBEDDER
    if _DEFAULT_EMBEDDER is None:
        _DEFAULT_EMBEDDER = Embedder()
    return _DEFAULT_EMBEDDER


def embed_chunks(chunks: List[Dict[str, Any]], show_progress_bar: bool = False) -> np.ndarray:
    """Convenience function to embed chunks using default embedder."""
    return get_default_embedder().embed_chunks(chunks, show_progress_bar=show_progress_bar)


def embed_query(query: str) -> np.ndarray:
    """Convenience function to embed a query using default embedder."""
    return get_default_embedder().embed_query(query)
