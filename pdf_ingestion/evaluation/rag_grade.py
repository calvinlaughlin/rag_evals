"""rag_grader.py
Automatic grading of Retrieval-Augmented Generation results.

The grader expects the output JSON produced by ``rag_evaluation.py`` and
uses OpenAI ChatCompletion to assign 0-5 scores for three dimensions:
• relevance   – does the answer address the question?
• faithfulness – is every claim backed by the retrieved context?
• depth       – completeness / richness of the answer.

It prints a leaderboard and stores the enriched grading data to
``graded_results.json`` by default.

Example
-------
$ python rag_grader.py rag_results.json  --output graded_results.json
"""
from __future__ import annotations

import argparse
import json
import os
import time
from typing import Dict, Any

import openai

# ──────────────────────────────────────────────────────────────────────────────
# OpenAI setup
# ──────────────────────────────────────────────────────────────────────────────
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    raise EnvironmentError("Set OPENAI_API_KEY env var to use the grader.")

MODEL_NAME = "gpt-4o"  
MAX_RETRIES = 3

# ──────────────────────────────────────────────────────────────────────────────
# Prompt template
# ──────────────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are a strict evaluator of question-answer pairs. You will receive a "
    "user question, the assistant's answer, and the context passages that the "
    "assistant had access to. Your job is to grade the answer on three axes "
    "from 0 (very poor) to 5 (excellent). Return ONLY valid JSON with keys "
    "'relevance', 'faithfulness', 'depth' (integers). Do not output anything "
    "else.\n\n"
    "Guidelines:\n"
    "• relevance – how well the answer addresses the question.\n"
    "• faithfulness – are all statements supported by the context? Penalise "
    "  hallucinations.\n"
    "• depth – completeness, detail, and nuance given the context length."
)

USER_TEMPLATE = (
    "QUESTION:\n{question}\n\nANSWER:\n{answer}\n\nCONTEXT (source passages):\n{context}\n\n"
    "Respond with JSON only."
)

# ──────────────────────────────────────────────────────────────────────────────

def grade_answer(question: str, answer: str, context: str) -> Dict[str, int]:
    """Call OpenAI to grade *answer* and return metric dict."""
    user_content = USER_TEMPLATE.format(question=question, answer=answer, context=context)
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = openai.ChatCompletion.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
                temperature=0.0,
                max_tokens=128,
            )
            raw = resp.choices[0].message.content.strip()
            data = json.loads(raw)
            # basic validation
            assert all(k in data for k in ("relevance", "faithfulness", "depth"))
            return {k: int(data[k]) for k in ("relevance", "faithfulness", "depth")}
        except (json.JSONDecodeError, AssertionError):
            if attempt == MAX_RETRIES:
                raise
            # back-off and retry
            time.sleep(1.5 * attempt)
    # Unreachable
    raise RuntimeError("Failed to grade answer after retries")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description="Grade RAG answers with GPT")
    p.add_argument("results_json", help="Path to rag_results.json from rag_evaluation.py")
    p.add_argument("--output", default="graded_results.json", help="Output file path")
    p.add_argument("--sleep", type=float, default=0.0, help="Seconds to sleep between requests (optional rate limiting)")
    args = p.parse_args()

    with open(args.results_json) as fh:
        rag_results: Dict[str, Dict[str, Any]] = json.load(fh)

    graded: Dict[str, Any] = {}

    for strategy_name, q_dict in rag_results.items():
        strategy_scores = {"relevance": [], "faithfulness": [], "depth": []}
        graded[strategy_name] = {}
        for q, data in q_dict.items():
            answer = data["answer"]
            # join context passages (truncate to reasonable length)
            context_passages = [c["content"] for c in data["chunks"]]
            context = "\n\n---\n\n".join(context_passages)[:3000]

            scores = grade_answer(q, answer, context)
            graded[strategy_name][q] = {"scores": scores, "answer": answer}
            for k in strategy_scores:
                strategy_scores[k].append(scores[k])

            if args.sleep:
                time.sleep(args.sleep)

        # aggregate averages per strategy
        graded[strategy_name]["_avg"] = {
            k: sum(v) / len(v) if v else 0.0 for k, v in strategy_scores.items()
        }

    # save
    with open(args.output, "w") as fh:
        json.dump(graded, fh, indent=2)
    print(f"Saved graded results to {args.output}\n")

    # leaderboard printout
    print("\nAverage scores by strategy (0-5):")
    print("Strategy                           | Relevance | Faithful | Depth | Overall")
    print("-" * 75)
    for strat, data in graded.items():
        avg = data["_avg"]
        overall = (avg["relevance"] + avg["faithfulness"] + avg["depth"]) / 3
        print(f"{strat.split('(')[0].strip():<34} | {avg['relevance']:^9.2f} | {avg['faithfulness']:^8.2f} | {avg['depth']:^5.2f} | {overall:^7.2f}")


if __name__ == "__main__":
    main() 