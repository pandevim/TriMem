"""
LongMemEval scoring.

Upstream uses a GPT-4o judge (``src/evaluation/evaluate_qa.py`` →
``autoeval_label``). We support two scorers:

  * ``exact_substring`` — cheap, deterministic. Marks a prediction
    correct if any gold-answer candidate (the gold may list alternates
    like ``"7 days. 8 days (including the last day) is also acceptable."``)
    appears as a normalized substring of the prediction. Good for smoke
    runs; brittle on paraphrase.
  * ``llm_judge``        — same vLLM backend the agent uses, prompted
    with a strict yes/no rubric. Tolerates paraphrase, the dominant
    failure mode of ``exact_substring`` on free-form gold answers.

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


_JUDGE_SYSTEM = """\
You are a strict but fair grader for a long-term-memory QA benchmark.
Decide whether a model's predicted answer matches the gold answer for
the given question.

Rules:
- Accept paraphrases that convey the same fact (e.g. "GPS issue" matches
  "GPS system not functioning correctly"; "the bike" matches "bike";
  "Samsung Galaxy S22" matches "the Samsung Galaxy S22").
- For numeric / temporal answers, the gold may list alternates such as
  "7 days. 8 days (including the last day) is also acceptable." Either
  form counts as correct.
- Reject predictions that contradict the gold, omit the requested fact,
  or only restate the question without answering it.
- Ignore extraneous reasoning, preambles, or "Thinking Process" text in
  the prediction; judge only whether the final fact is present and right.

Output: a single token, either "yes" or "no". No punctuation, no
explanation, no other text.
"""


_JUDGE_USER_TMPL = """\
Question: {question}

Gold answer: {gold}

Predicted answer: {prediction}

Verdict (yes or no):"""


# Lazy-init: the judge shares the agent's vLLM singleton, so the model
# is loaded exactly once per process.
_judge_llm = None


def _get_judge_llm():
    global _judge_llm
    if _judge_llm is None:
        from utils.llm import get_llm
        _judge_llm = get_llm()
    return _judge_llm


def _parse_verdict(text: str) -> bool | None:
    """Return True for yes, False for no, None if unparseable."""
    if not text:
        return None
    # Strip any residual <think>…</think> block, whitespace, punctuation.
    t = text
    if "</think>" in t:
        t = t.rsplit("</think>", 1)[-1]
    t = t.strip().lower().lstrip("`*\"' \t\n\r.,:;!?")
    if t.startswith("yes"):
        return True
    if t.startswith("no"):
        return False
    return None


def score_llm_judge(prediction: str, gold: str, question: str) -> ScoreResult:
    """Judge by yes/no verdict from the same LLM the agent uses."""
    from configs.settings import (
        ENABLE_THINKING_JUDGE,
        JUDGE_MAX_TOKENS,
        JUDGE_TEMPERATURE,
    )

    if not gold.strip():
        return ScoreResult(False, "llm_judge", "empty gold answer")
    if not prediction.strip():
        return ScoreResult(False, "llm_judge", "empty prediction")

    user = _JUDGE_USER_TMPL.format(
        question=question.strip() or "(no question provided)",
        gold=gold.strip(),
        prediction=prediction.strip(),
    )
    resp = _get_judge_llm().chat(
        system=_JUDGE_SYSTEM,
        messages=[{"role": "user", "content": user}],
        max_tokens=JUDGE_MAX_TOKENS,
        temperature=JUDGE_TEMPERATURE,
        chat_template_kwargs={"enable_thinking": ENABLE_THINKING_JUDGE},
    )
    verdict = _parse_verdict(resp.text)
    if verdict is None:
        # Unparseable verdict — fall back to exact_substring so we don't
        # silently mark every borderline case correct or incorrect.
        sub = score_exact_substring(prediction, gold)
        return ScoreResult(
            sub.correct,
            "llm_judge",
            f"unparseable judge output {resp.text!r}; fell back to exact_substring",
        )
    return ScoreResult(verdict, "llm_judge", f"judge said {resp.text.strip()!r}")


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
