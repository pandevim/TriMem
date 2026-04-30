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


# Strips upstream's "(... ) is also acceptable" qualifier so the surrounding
# alternate answer can be split out as its own candidate.
_ACCEPTABLE_PHRASE = re.compile(
    r"\s*\([^)]*\)\s*is also acceptable\.?", flags=re.IGNORECASE
)


def _candidate_answers(gold: str) -> list[str]:
    """Extract acceptable answer candidates from a LongMemEval gold string.

    Upstream often writes gold like ``"7 days. 8 days (including the last
    day) is also acceptable."`` — both forms should match. We split on
    sentence/list boundaries, drop the qualifier phrase, and normalize.
    """
    cleaned = _ACCEPTABLE_PHRASE.sub("", gold).strip()
    parts = re.split(r"\s+or\s+|;|\.\s+", cleaned)
    cands: list[str] = []
    for part in parts:
        p = part.strip().rstrip(".,").strip()
        if not p:
            continue
        norm = _normalize(p)
        if norm and norm not in cands:
            cands.append(norm)
    full = _normalize(gold.rstrip(".,"))
    if full and full not in cands:
        cands.append(full)
    return cands


def score_exact_substring(prediction: str, gold: str) -> ScoreResult:
    p = _normalize(prediction)
    cands = _candidate_answers(gold)
    if not cands:
        return ScoreResult(False, "exact_substring", "empty gold answer")
    for c in cands:
        if c in p:
            return ScoreResult(True, "exact_substring", f"matched '{c}'")
    return ScoreResult(False, "exact_substring")


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
