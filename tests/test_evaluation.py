import json
from pdf_ingestion.evaluation import grade_answer


def test_grade_answer_mock():
    scores = grade_answer("q", "Some answer", "Some context")
    # Mock returns dummy json so keys exist as ints
    assert set(scores.keys()) == {"relevance", "faithfulness", "depth"}
    assert all(isinstance(v, int) for v in scores.values()) 