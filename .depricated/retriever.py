"""retriever.py
Real retrieval utilities for chunk similarity search.

Usage example
-------------
>>> from retriever import ChunkRetriever
>>> retriever = ChunkRetriever(model_name="all-MiniLM-L6-v2")
>>> retriever.index(chunks)  # ``chunks`` = List[Dict[str, Any]] with ``content`` key
>>> top_chunks = retriever.query("What is Reducto?", top_k=5)
"""
from __future__ import annotations

import logging
from typing import List, Dict, Any

import numpy as np
from sentence_transformers import SentenceTransformer

try:
    import faiss  # type: ignore
except ImportError as exc:
    raise ImportError(
        "faiss is required for vector search. Install with `pip install faiss-cpu`."
    ) from exc


logger = logging.getLogger(__name__)


class ChunkRetriever:
    """Embeds document chunks and supports cosine-similarity retrieval via FAISS."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.model_name = model_name
        logger.info("Loading sentence-transformer model %s", model_name)
        self.model = SentenceTransformer(model_name)
        self.index: faiss.IndexFlatIP | None = None
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

        # Normalize vectors to use inner-product as cosine similarity
        faiss.normalize_L2(embeddings)
        logger.info("Building FAISS index of size %d", embeddings.shape[0])
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)
        self.index = index

    def query(self, query_text: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Return *top_k* most similar chunks to *query_text*."""
        if self.index is None:
            raise RuntimeError("Index not built. Call `.index(chunks)` first.")

        query_emb = self._embed([query_text])
        faiss.normalize_L2(query_emb)
        scores, indices = self.index.search(query_emb, top_k)
        indices = indices[0]  # shape (top_k,)
        scores = scores[0]

        # Collect chunks with similarity scores
        results: List[Dict[str, Any]] = []
        for idx, score in zip(indices, scores):
            if idx == -1:
                continue  # faiss pads with -1 if fewer than top_k results
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