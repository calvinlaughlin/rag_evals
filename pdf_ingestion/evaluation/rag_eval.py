"""rag_evaluation.py
End-to-end Retrieval-Augmented Generation evaluation over PDF/Text documents.

This script:
1. Loads a document (TXT or PDF).
2. Runs each ingestion strategy to obtain chunks.
3. Builds a FAISS index with Sentence-Transformer embeddings for each strategy.
4. For a set of user queries, retrieves the top-k chunks and asks OpenAI
   ChatCompletion to answer *using only the provided context*.
5. Writes answers to ``rag_results.json`` for later scoring/analysis.

Prerequisites
-------------
• environment variable ``OPENAI_API_KEY`` must be set.
• run ``pip install -r requirements.txt`` (requirements include ``openai``).

Example
-------
$ python rag_evaluation.py sample_document.txt "What is Reducto?" "How does fixed-size rolling window work?"
"""
from __future__ import annotations

import argparse
import json
import os
import textwrap
from typing import List, Dict, Any

from openai import OpenAI

from pdf_ingestion.ingest.strategies import get_all_strategies, process_document
from pdf_ingestion.utils.pdf_utils import load_document
from pdf_ingestion.retrieval.retriever import ChunkRetriever

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise EnvironmentError(
        "OPENAI_API_KEY environment variable not set. Export it before running."
    )

SYSTEM_MESSAGE = (
    "You are a meticulous assistant. Answer the user question ONLY with facts "
    "found in the provided context. If the context is insufficient, reply 'I don't "
    "know based on the given context.'"
)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def build_retrievers(chunks_by_strategy: Dict[str, List[Dict[str, Any]]]) -> Dict[str, ChunkRetriever]:
    """Return a ChunkRetriever instance per strategy."""
    retrievers: Dict[str, ChunkRetriever] = {}
    for name, chunks in chunks_by_strategy.items():
        retr = ChunkRetriever()
        retr.index(chunks)
        retrievers[name] = retr
    return retrievers


def answer_query(retriever: ChunkRetriever, query: str, top_k: int = 5) -> Dict[str, Any]:
    """Retrieve context and query OpenAI, returning answer and supporting data."""
    top_chunks = retriever.query(query, top_k=top_k)
    context = "\n\n---\n\n".join(
        [textwrap.dedent(chunk["content"]).strip() for chunk in top_chunks]
    )

    prompt_messages = [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {
            "role": "user",
            "content": (
                "Context:\n" + context + "\n\nQuestion: " + query + "\nAnswer:"),
        },
    ]

    # Initialize the OpenAI client
    client = OpenAI(api_key=api_key)
    
    # Call the OpenAI API using the new client-based approach
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo-0125",  # change as desired
        messages=prompt_messages,
        temperature=0.2,
        max_tokens=512,
    )

    # Extract the response content using the new API structure
    answer_text = completion.choices[0].message.content.strip()
    
    # Usage information is now accessed differently
    usage_dict = {}
    if hasattr(completion, "usage"):
        usage_dict = {
            "prompt_tokens": completion.usage.prompt_tokens,
            "completion_tokens": completion.usage.completion_tokens,
            "total_tokens": completion.usage.total_tokens
        }
        
    return {
        "answer": answer_text,
        "chunks": top_chunks,
        "usage": usage_dict,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Retrieval-Augmented Generation over a document and save answers."
    )
    parser.add_argument("document", help="Path to TXT or PDF document.")
    parser.add_argument("queries", nargs="+", help="Question(s) to ask.")
    parser.add_argument("--top-k", type=int, default=5, help="Number of chunks to retrieve.")
    parser.add_argument(
        "--output",
        default="rag_results.json",
        help="Output JSON file with answers (default: rag_results.json)",
    )

    args = parser.parse_args()

    # 1. Load and chunk document
    doc_text = load_document(args.document)
    chunks_by_strategy = process_document(doc_text)

    # 2. Build retrievers
    retrievers = build_retrievers(chunks_by_strategy)

    # 3. Answer queries
    all_results: Dict[str, Dict[str, Any]] = {}
    for strategy_name, retr in retrievers.items():
        strategy_results = {}
        for q in args.queries:
            strategy_results[q] = answer_query(retr, q, top_k=args.top_k)
        all_results[strategy_name] = strategy_results

    # 4. Save
    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Saved RAG answers to {args.output}")


if __name__ == "__main__":
    main() 