import os
import pytest

from pdf_ingestion.retrieval import ChunkRetriever

SAMPLE_CHUNKS = [
    {"content": "Reducto is a structured chunking system."},
    {"content": "Fixed-size rolling window divides text into equally-sized chunks."},
    {"content": "Sentence overlap chunking preserves sentence boundaries."},
]


@pytest.mark.integration
def test_retriever_real_embeddings():
    """Load real sentence-transformer model and perform similarity search."""
    model_name = os.getenv("ST_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    retr = ChunkRetriever(model_name=model_name)
    retr.index(SAMPLE_CHUNKS)

    # Query should bring the Reducto sentence to the top
    results = retr.query("What is Reducto?", top_k=1)
    assert results[0]["content"].startswith("Reducto") 