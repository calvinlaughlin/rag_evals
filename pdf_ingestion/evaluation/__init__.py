"""
Evaluation tools for chunking strategies.
"""

from pdf_ingestion.evaluation.retrieval_metrics import (
    calculate_precision,
    calculate_recall,
    calculate_f1,
)

# ---------------------------------------------------------------------------
# Optional lazy import for `grade_answer` to avoid mandatory OpenAI key at
# *import time*.
# ---------------------------------------------------------------------------

import importlib
import os
from types import ModuleType
from typing import Any, Callable


def _lazy_grade_answer() -> Callable[..., Any]:
    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError("grade_answer requires OPENAI_API_KEY – set the variable before calling.")
    mod: ModuleType = importlib.import_module("pdf_ingestion.evaluation.rag_grade")
    return mod.grade_answer  # type: ignore[attr-defined]


# Expose a wrapper that resolves the real function on first call.

def grade_answer(*args, **kwargs):  # type: ignore[override]
    fn = _lazy_grade_answer()
    return fn(*args, **kwargs)
