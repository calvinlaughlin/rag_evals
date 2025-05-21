"""retriever.py
Real retrieval utilities for chunk similarity search.

Usage example
-------------
>>> from pdf_ingestion.retrieval.retriever import ChunkRetriever
>>> retriever = ChunkRetriever(model_name="all-MiniLM-L6-v2")
>>> retriever.index(chunks)  # ``chunks`` = List[Dict[str, Any]] with ``content`` key
>>> top_chunks = retriever.query("What is Reducto?", top_k=5)
"""
from __future__ import annotations

import logging
from typing import List, Dict, Any

import numpy as np
from sentence_transformers import SentenceTransformer

# -----------------------------------------------------------------------------
# Optional FAISS import
# -----------------------------------------------------------------------------
try:
    import faiss  # type: ignore
    _FAISS_AVAILABLE = True
except ImportError:  # pragma: no cover – handled gracefully below
    faiss = None  # type: ignore
    _FAISS_AVAILABLE = False


logger = logging.getLogger(__name__)


class ChunkRetriever:
    """Embeds document chunks and supports cosine-similarity retrieval via FAISS."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.model_name = model_name
        logger.info("Loading sentence-transformer model %s", model_name)
        self.model = SentenceTransformer(model_name)
        # Whether we can rely on FAISS for fast ANN search.  We require two
        # key callables – ``normalize_L2`` and the ``IndexFlatIP`` constructor.
        self._use_faiss = (
            _FAISS_AVAILABLE
            and callable(getattr(faiss, "normalize_L2", None))  # type: ignore[arg-type]
            and callable(getattr(faiss, "IndexFlatIP", None))  # type: ignore[arg-type]
        )

        # Runtime data structures (initialised on ``index``)
        self.faiss_index = None  # FAISS index object when available
        self._embeddings: np.ndarray | None = None  # fallback dense matrix when FAISS is not used
        self.chunks: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def index(self, chunks: List[Dict[str, Any]]) -> None:
        """Embed *chunks* and build an in-memory FAISS index."""
        if not chunks:
            raise ValueError("No chunks provided for indexing.")

        self.chunks = chunks

        # Compute embeddings (float32 numpy array of shape (n_chunks, dim))
        texts = [chunk["content"] for chunk in chunks]
        embeddings = self._embed(texts)

        # Normalise vectors so dot-product == cosine similarity.
        # We prefer FAISS for speed, but gracefully fall back to NumPy if any
        # call fails (e.g., broken/partial FAISS install).
        if self._use_faiss:
            try:
                faiss.normalize_L2(embeddings)  # type: ignore[arg-type]
                logger.info("Building FAISS index of size %d", embeddings.shape[0])
                index = faiss.IndexFlatIP(embeddings.shape[1])  # type: ignore[attr-defined]
                index.add(embeddings)
                self.faiss_index = index
                self._embeddings = None
            except Exception as exc:  # pragma: no cover
                logger.warning("FAISS operation failed – falling back to NumPy. (%s)", exc)
                self._use_faiss = False
        if not self._use_faiss:
            # Pure NumPy fallback – keep the normalised embedding matrix.
            logger.info("Using NumPy fallback for retrieval.")
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            self._embeddings = embeddings / norms
            self.faiss_index = None

    def query(self, query_text: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Return *top_k* most similar chunks to *query_text*."""
        if self.faiss_index is None and self._embeddings is None:
            raise RuntimeError("Index not built. Call `.index(chunks)` first.")

        query_emb = self._embed([query_text])

        if self._use_faiss and self.faiss_index is not None:
            faiss.normalize_L2(query_emb)  # type: ignore[arg-type]
            scores, indices = self.faiss_index.search(query_emb, top_k)
            indices = indices[0]
            scores = scores[0]
        else:
            # NumPy fallback – cosine similarity via dot product on L2-normalised vectors.
            norm = np.linalg.norm(query_emb, axis=1, keepdims=True)
            norm[norm == 0] = 1.0
            query_emb = query_emb / norm
            sims = np.dot(self._embeddings, query_emb.T).reshape(-1)  # type: ignore[arg-type]
            # Get top-k indices (descending order)
            indices = np.argsort(-sims)[:top_k]
            scores = sims[indices]

        # Collect chunks with similarity scores
        results: List[Dict[str, Any]] = []
        for idx, score in zip(indices, scores):
            if idx == -1 or idx >= len(self.chunks):
                continue
            chunk = self.chunks[idx].copy()
            chunk["similarity"] = float(score)
            results.append(chunk)
        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _embed(self, texts: List[str]) -> np.ndarray:
        """Return L2-normalised embeddings for *texts* as np.float32."""
        embeddings = self.model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        return embeddings.astype("float32") 