import os
import pytest

from pdf_ingestion.ingest import process_document
from pdf_ingestion.utils import load_document
from pdf_ingestion.retrieval import ChunkRetriever

OPENAI_KEY = os.getenv("OPENAI_API_KEY")

pytestmark = pytest.mark.integration


def _skip_if_no_key():
    if not OPENAI_KEY:
        pytest.skip("OPENAI_API_KEY not set, skipping live OpenAI test")


@pytest.mark.integration
def test_openai_answer_generation(sample_doc_path="sample_document.txt"):
    _skip_if_no_key()

    # Load and chunk
    text = load_document(sample_doc_path)
    chunks_by_strategy = process_document(text)
    strat, chunks = next(iter(chunks_by_strategy.items()))

    # Retrieval
    retr = ChunkRetriever()
    retr.index(chunks)
    results = retr.query("What is Reducto?", top_k=3)
    context = "\n\n".join(c["content"] for c in results)

    import openai

    resp = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0125",
        messages=[
            {"role": "system", "content": "Answer using the context strictly."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: What is Reducto?"},
        ],
        max_tokens=32,
    )
    answer = resp.choices[0].message.content.strip()
    assert answer 