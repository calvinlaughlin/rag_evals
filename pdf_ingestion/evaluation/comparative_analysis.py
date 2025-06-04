from __future__ import annotations

"""comparative_analysis.py
Run all chunking strategies on a document, compute IR metrics (precision, recall,
F1).  Acts as a thin wrapper combining:
• ingestion strategies (pdf_ingestion.ingest.strategies)
• retrieval (pdf_ingestion.retrieval.retriever)
• metric helpers (pdf_ingestion.evaluation.retrieval_metrics)

This module purposely avoids heavyweight LLM calls so it can execute offline and
quickly during development.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Any, List

from pdf_ingestion.ingest.strategies import process_document
from pdf_ingestion.retrieval.retriever import ChunkRetriever
from pdf_ingestion.evaluation.retrieval_metrics import (
    evaluate_chunking_strategy,
)
from pdf_ingestion.utils.pdf_utils import load_document

# ---------------------------------------------------------------------------
# Ground-truth helpers
# ---------------------------------------------------------------------------


def _load_ground_truth(gt_path: str | Path) -> Dict[str, List[int]]:
    """Return mapping query → relevant chunk indices (list[int])."""
    import json

    with open(gt_path) as fh:
        data = json.load(fh)
    return {q["query"]: q["relevant_chunk_indices"] for q in data["queries"]}


# ---------------------------------------------------------------------------
# Evaluation driver
# ---------------------------------------------------------------------------


def run_full_evaluation(
    document_path: str | Path,
    ground_truth_path: str | Path,
    top_k: int = 5,
) -> Dict[str, Any]:
    """Return dict with aggregate metrics for each strategy."""

    doc_text = load_document(document_path)
    chunks_by_strategy = process_document(doc_text)

    ground_truth = _load_ground_truth(ground_truth_path)

    results: Dict[str, Any] = {}
    for strat_name, chunks in chunks_by_strategy.items():
        retriever = ChunkRetriever()
        retriever.index(chunks)

        per_query_retrieved: Dict[str, List[int]] = {}
        for query, relevant_idx in ground_truth.items():
            top_chunks = retriever.query(query, top_k=top_k)
            # Use position in `chunks` list as ID – retriever retains original
            # ordering so we can resolve indices.
            retrieved_indices = [chunks.index(c) for c in top_chunks]
            per_query_retrieved[query] = retrieved_indices

        metrics = evaluate_chunking_strategy(strat_name, per_query_retrieved, ground_truth)
        results[strat_name] = metrics

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare chunking strategies on retrieval metrics.")
    ap.add_argument("document", help="Path to document (TXT or PDF)")
    ap.add_argument(
        "--ground-truth",
        default="data/ground_truth_queries.json",
        help="JSON with queries + relevant chunk indices",
    )
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--output", default="retrieval_comparison.json")
    args = ap.parse_args()

    stats = run_full_evaluation(args.document, args.ground_truth, top_k=args.top_k)

    with open(args.output, "w") as fh:
        json.dump(stats, fh, indent=2)
    print(f"Saved metrics to {args.output}")

    # Pretty print
    print("\nMacro-averaged metrics (precision/recall/F1):")
    print("Strategy                             | P    | R    | F1   |")
    print("-" * 60)
    for strat, m in stats.items():
        print(f"{strat.split('(')[0].strip():<35} | {m['precision']*100:5.1f} | {m['recall']*100:5.1f} | {m['f1']*100:5.1f}")


if __name__ == "__main__":
    main() 