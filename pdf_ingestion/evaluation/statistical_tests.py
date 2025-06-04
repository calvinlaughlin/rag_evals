from __future__ import annotations

"""statistical_tests.py
Statistical significance testing utilities (paired t-test) for comparing
retrieval metrics between chunking strategies.
"""

from typing import Dict, List, Tuple

from scipy import stats

__all__ = [
    "test_significance",
    "generate_significance_report",
]


def _paired_metric_lists(
    results: Dict[str, Dict[str, float]], metric: str
) -> List[Tuple[float, float, float]]:
    """Return list of tuples (a_metric, b_metric) per strategy pair.

    Not used yet – placeholder if we extend beyond 3-way comparison.
    """
    raise NotImplementedError


def test_significance(
    structured_scores: List[float],
    fixed_scores: List[float],
    sentence_scores: List[float],
) -> Dict[str, float]:
    """Run paired t-tests between strategy score lists.

    Inputs are lists of per-query F1 (or any metric) scores for each strategy.
    Returns dict with p-values for comparisons.
    """
    res = {
        "structured_vs_fixed": stats.ttest_rel(structured_scores, fixed_scores).pvalue,
        "structured_vs_sentence": stats.ttest_rel(structured_scores, sentence_scores).pvalue,
        "fixed_vs_sentence": stats.ttest_rel(fixed_scores, sentence_scores).pvalue,
    }
    return res


def generate_significance_report() -> None:  # pragma: no cover – CLI entry point
    """Small CLI helper for quick local testing (not wired to anything yet)."""
    import json, argparse

    ap = argparse.ArgumentParser(description="Compute p-values between strategies from JSON comparison file")
    ap.add_argument("comparison_json", help="Path to retrieval_comparison.json produced by comparative_analysis.py")
    args = ap.parse_args()

    with open(args.comparison_json) as fh:
        data = json.load(fh)

    # Expect per-query metrics inside each strategy dict
    def _extract_per_query_f1(strategy_dict):
        return [m["f1"] for m in strategy_dict["per_query"].values()]

    struct_f1 = _extract_per_query_f1(data["Structured Chunking (Reducto-style)"])
    fixed_f1 = _extract_per_query_f1(next(k for k in data if k.startswith("Fixed-Size")))
    sent_f1 = _extract_per_query_f1(next(k for k in data if k.startswith("Sentence")))

    pvals = test_significance(struct_f1, fixed_f1, sent_f1)

    print("Paired t-test p-values:")
    for k, v in pvals.items():
        print(f"{k}: {v:.4f}") 