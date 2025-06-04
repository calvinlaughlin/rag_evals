import pytest

from pdf_ingestion.evaluation.retrieval_metrics import (
    calculate_precision,
    calculate_recall,
    calculate_f1,
    evaluate_query,
)


def test_basic_prf():
    retrieved = [1, 2, 3]
    relevant = [2, 3, 4, 5]

    p = calculate_precision(retrieved, relevant)
    r = calculate_recall(retrieved, relevant)
    f1 = calculate_f1(p, r)

    assert pytest.approx(p) == 2 / 3
    assert pytest.approx(r) == 2 / 4
    assert pytest.approx(f1) == 2 * p * r / (p + r)


def test_empty_sets():
    assert calculate_precision([], [1, 2]) == 0.0
    assert calculate_recall([1, 2], []) == 0.0
    assert calculate_f1(0.0, 0.0) == 0.0


def test_evaluate_query():
    metrics = evaluate_query([10, 20], [20, 30])
    assert set(metrics.keys()) == {"precision", "recall", "f1"} 