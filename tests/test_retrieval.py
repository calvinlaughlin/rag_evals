from pdf_ingestion.retrieval import ChunkRetriever

SAMPLE_CHUNKS = [
    {"content": "Reducto is a structured chunking system."},
    {"content": "Fixed-size rolling window divides text into equally-sized chunks."},
    {"content": "Sentence overlap chunking preserves sentence boundaries."},
]


def test_retriever_roundtrip():
    retr = ChunkRetriever(model_name="all-MiniLM-L6-v2")
    retr.index(SAMPLE_CHUNKS)
    results = retr.query("What is Reducto?", top_k=2)
    assert len(results) >= 1
    # The first result should mention Reducto
    assert "Reducto" in results[0]["content"] 