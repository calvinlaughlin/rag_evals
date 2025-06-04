from __future__ import annotations

"""retrieval_metrics.py
Standard information-retrieval metrics (precision, recall, F1) and convenience
helpers for evaluating chunk retrieval strategies.
"""

from typing import List, Dict, Any, Iterable

__all__ = [
    "calculate_precision",
    "calculate_recall",
    "calculate_f1",
    "evaluate_query",
    "evaluate_chunking_strategy",
]


def _intersection_size(a: Iterable[int], b: Iterable[int]) -> int:
    """Return |a ∩ b| treating *a* and *b* as sets of integers."""
    return len(set(a).intersection(b))


def calculate_precision(retrieved: List[int], relevant: List[int]) -> float:
    """Precision = |relevant ∩ retrieved| / |retrieved|.

    Args:
        retrieved: List of retrieved chunk indices (or any hashable ids).
        relevant:  List of ground-truth relevant chunk indices.

    Returns
    -------
    float
        Precision in the range [0, 1].  Returns *0.0* if ``retrieved`` is
        empty (to avoid division-by-zero).
    """
    if not retrieved:
        return 0.0
    inter = _intersection_size(retrieved, relevant)
    return inter / len(retrieved)


def calculate_recall(retrieved: List[int], relevant: List[int]) -> float:
    """Recall = |relevant ∩ retrieved| / |relevant|.

    Returns 0.0 when *relevant* is empty.
    """
    if not relevant:
        return 0.0
    inter = _intersection_size(retrieved, relevant)
    return inter / len(relevant)


def calculate_f1(precision: float, recall: float) -> float:
    """F1 = harmonic mean of *precision* and *recall*.

    If both *precision* and *recall* are 0, returns 0 to avoid ``ZeroDivision``.
    """
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)


def evaluate_query(retrieved: List[int], relevant: List[int]) -> Dict[str, float]:
    """Return precision/recall/F1 for a single query."""
    p = calculate_precision(retrieved, relevant)
    r = calculate_recall(retrieved, relevant)
    f1 = calculate_f1(p, r)
    return {"precision": p, "recall": r, "f1": f1}


def evaluate_chunking_strategy(
    strategy_name: str,
    results_by_query: Dict[str, List[int]],
    ground_truth: Dict[str, List[int]],
) -> Dict[str, Any]:
    """Aggregate IR metrics for *strategy_name* across multiple queries.

    Parameters
    ----------
    strategy_name : str
        Name of the chunking strategy (for reference only).
    results_by_query : Dict[str, List[int]]
        Mapping *query* → list of **retrieved chunk indices** produced by the
        evaluated system.
    ground_truth : Dict[str, List[int]]
        Mapping *query* → list of **relevant chunk indices** (gold labels).

    Returns
    -------
    Dict[str, Any]
        ``{"precision": float, "recall": float, "f1": float, "per_query": {...}}``
        where *per_query* contains the individual query metrics.
    """
    per_query: Dict[str, Dict[str, float]] = {}
    precisions: List[float] = []
    recalls: List[float] = []
    f1s: List[float] = []

    for q, retrieved in results_by_query.items():
        relevant = ground_truth.get(q, [])
        metrics = evaluate_query(retrieved, relevant)
        per_query[q] = metrics
        precisions.append(metrics["precision"])
        recalls.append(metrics["recall"])
        f1s.append(metrics["f1"])

    macro_precision = sum(precisions) / len(precisions) if precisions else 0.0
    macro_recall = sum(recalls) / len(recalls) if recalls else 0.0
    macro_f1 = sum(f1s) / len(f1s) if f1s else 0.0

    return {
        "strategy": strategy_name,
        "precision": macro_precision,
        "recall": macro_recall,
        "f1": macro_f1,
        "per_query": per_query,
    } 