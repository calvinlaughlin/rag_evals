"""llm_evaluation.py
Real answer-quality evaluation for the ingestion strategies.

Workflow
--------
1. Load a document (TXT or PDF) and chunk it with all strategies.
2. Build a ChunkRetriever per strategy (Sentence-Transformer + FAISS).
3. For each test query, retrieve top-k chunks and ask an OpenAI chat
   model for an answer constrained to the context.
4. Score the answer with the same rubric used by ``rag_grader.grade_answer``
   (relevance, faithfulness, depth).
5. Print a table and optionally save JSON and a bar-chart PNG.

Run
---
$ export OPENAI_API_KEY=sk-...
$ python llm_evaluation.py sample_document.txt
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Any, List

import matplotlib.pyplot as plt

from pdf_ingestion.ingest.strategies import process_document
from pdf_ingestion.utils.pdf_utils import load_document
from pdf_ingestion.retrieval.retriever import ChunkRetriever
from pdf_ingestion.evaluation.rag_grade import grade_answer  # reuse grader helper

import openai

openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    raise EnvironmentError("OPENAI_API_KEY not set.")

TEST_QUERIES: List[str] = [
    "What is Reducto and how does it work?",
    "How does fixed-size rolling window chunking work?",
    "What evaluation metrics are used for PDF ingestion strategies?",
]

SYSTEM_PROMPT = (
    "You are a knowledgeable assistant. Answer ONLY with information present in the provided context. "
    "If the context is insufficient, say 'I don't know based on the context.'"
)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def build_retrievers(chunks_by_strategy: Dict[str, List[Dict[str, Any]]]) -> Dict[str, ChunkRetriever]:
    retrievers: Dict[str, ChunkRetriever] = {}
    for name, ch in chunks_by_strategy.items():
        r = ChunkRetriever()
        r.index(ch)
        retrievers[name] = r
    return retrievers


def answer_query(retr: ChunkRetriever, query: str, top_k: int = 5) -> Dict[str, Any]:
    top_chunks = retr.query(query, top_k=top_k)
    context = "\n\n---\n\n".join(c["content"] for c in top_chunks)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"},
    ]

    resp = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=messages,
    )

    answer = resp.choices[0].message.content.strip()
    return {
        "answer": answer,
        "chunks": top_chunks,
        "usage": resp.usage.to_dict() if hasattr(resp, "usage") else {},
    }


# ---------------------------------------------------------------------------
# Main evaluation routine
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate answer quality with real GPT answers.")
    ap.add_argument("document", help="Path to TXT or PDF document")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--output", default="answer_eval_results.json")
    args = ap.parse_args()

    # 1. Chunk document
    text = load_document(args.document)
    chunks_by_strategy = process_document(text)

    # 2. Build retrievers
    retrievers = build_retrievers(chunks_by_strategy)

    # 3. Generate answers & grade
    results: Dict[str, Any] = {}
    for strat, retr in retrievers.items():
        strat_res = {}
        for q in TEST_QUERIES:
            qa = answer_query(retr, q, top_k=args.top_k)
            scores = grade_answer(q, qa["answer"], "\n\n".join(c["content"] for c in qa["chunks"]))
            strat_res[q] = {**qa, "scores": scores}
        results[strat] = strat_res

    # 4. Save JSON
    with open(args.output, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"Saved detailed results to {args.output}")

    # 5. Plot overall quality per strategy
    quality_by_strat = {
        s: sum(r[q]["scores"]["relevance"] + r[q]["scores"]["faithfulness"] + r[q]["scores"]["depth"] for q in TEST_QUERIES) / (3 * len(TEST_QUERIES))
        for s, r in results.items()
    }

    plt.figure(figsize=(8, 5))
    strategies, scores = zip(*quality_by_strat.items())
    plt.bar(range(len(scores)), scores, color="seagreen")
    plt.ylabel("Average Quality (0-5)")
    plt.xticks(range(len(scores)), [s.split("(")[0].strip() for s in strategies], rotation=45, ha="right")
    plt.title("GPT-graded Answer Quality by Chunking Strategy")
    plt.tight_layout()
    png_path = "answer_quality_comparison.png"
    plt.savefig(png_path)
    print(f"Plot saved as {png_path}")


if __name__ == "__main__":
    main() 