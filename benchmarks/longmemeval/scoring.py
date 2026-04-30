"""
LongMemEval scoring.

Upstream uses a GPT-4o judge (``src/evaluation/evaluate_qa.py`` →
``autoeval_label``). We support two scorers:

  * ``exact_substring`` — cheap, deterministic. Marks a prediction
    correct if the gold answer string appears (case-insensitive) inside
    the prediction. Useful as a fast smoke metric while the LLM judge is
    still being wired up.
  * ``llm_judge``        — calls a judge LLM with the same rubric the
    upstream eval uses. Stubbed for now; raises until configured.

Use ``llm_judge`` for any number that ends up in the paper.
"""
from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class ScoreResult:
    correct: bool
    method: str          # "exact_substring" | "llm_judge"
    rationale: str = ""


def _normalize(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[\"'`]", "", text)
    return text


def score_exact_substring(prediction: str, gold: str) -> ScoreResult:
    p, g = _normalize(prediction), _normalize(gold)
    if not g:
        return ScoreResult(False, "exact_substring", "empty gold answer")
    return ScoreResult(g in p, "exact_substring")


def score_llm_judge(prediction: str, gold: str, question: str) -> ScoreResult:
    """Stub for the GPT-4o-style judge used by upstream LongMemEval.

    Plumbing TODO: route through utils.llm with a frontier model and the
    upstream rubric prompt.
    """
    raise NotImplementedError(
        "LLM judge not wired yet — use score_exact_substring for now."
    )


def score_prediction(
    prediction: str,
    gold: str,
    *,
    question: str = "",
    method: str = "exact_substring",
) -> ScoreResult:
    if method == "exact_substring":
        return score_exact_substring(prediction, gold)
    if method == "llm_judge":
        return score_llm_judge(prediction, gold, question)
    raise ValueError(f"Unknown scoring method: {method}")
