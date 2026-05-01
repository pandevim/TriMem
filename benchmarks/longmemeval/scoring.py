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


def _coerce(x) -> str:
    """LongMemEval gold answers are sometimes ints (count questions in the
    multi-session category, e.g. "How many times did I…"). Coerce to str
    once at every public entry point so downstream regex / .strip()
    code never sees a non-string.
    """
    if x is None:
        return ""
    return x if isinstance(x, str) else str(x)


def _normalize(text) -> str:
    text = _coerce(text)
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[\"'`]", "", text)
    return text


# Strips upstream's "(... ) is also acceptable" qualifier so the surrounding
# alternate answer can be split out as its own candidate.
_ACCEPTABLE_PHRASE = re.compile(
    r"\s*\([^)]*\)\s*is also acceptable\.?", flags=re.IGNORECASE
)


def _candidate_answers(gold) -> list[str]:
    """Extract acceptable answer candidates from a LongMemEval gold string.

    Upstream often writes gold like ``"7 days. 8 days (including the last
    day) is also acceptable."`` — both forms should match. We split on
    sentence/list boundaries, drop the qualifier phrase, and normalize.
    Accepts non-str input (some multi-session golds are ints).
    """
    if not isinstance(gold, str):
        gold = "" if gold is None else str(gold)
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


# --- Rubric judge (single-session-preference) ----------------------------
# Preference questions in LongMemEval don't have factual gold answers —
# they have *rubrics* describing what the user "would prefer" to receive.
# A factual-match judge fails on these even when the prediction is great
# (we measured 6.7% on n=30 with the strict judge). The rubric judge
# instead compares the recommendation against the preference criteria.
_RUBRIC_JUDGE_SYSTEM = """\
You are grading recommendations against a user-preference rubric for a
long-term-memory QA benchmark.

The gold answer is NOT a single factual target — it is a rubric describing
what the user "would prefer" to receive (and sometimes what they "would
not prefer"). Your job is to decide whether the predicted recommendation
satisfies that rubric.

Rules:
- "yes" if the prediction's recommendation aligns with the user's stated
  preferences in the rubric (compatible brand, topic, style, or features
  the rubric calls out).
- "yes" if the prediction names specific items consistent with the
  rubric's "would prefer" criteria, even if not every detail is covered.
- "no" if the prediction recommends something the rubric explicitly says
  the user "would not prefer", contradicts the user's known interests,
  or fails to make a recommendation at all (e.g. "I don't know").
- "no" if the prediction is purely a chain-of-thought with no concrete
  recommendation. Look at the *bottom-line recommendation*, ignoring
  preamble and "Thinking Process" text.
- Partial alignment counts as yes when the rubric is broad; only mark no
  when the recommendation clearly misses or contradicts the rubric.

Output: a single token, either "yes" or "no". No punctuation, no
explanation, no other text.
"""


_RUBRIC_JUDGE_USER_TMPL = """\
Question: {question}

Preference rubric (what the user would / would not prefer):
{gold}

Predicted recommendation:
{prediction}

Does the prediction align with the preference rubric? (yes or no):"""


# Categories whose gold answers are rubrics, not factual targets. Anything
# not in this set uses the strict factual judge.
_RUBRIC_CATEGORIES = frozenset({"single-session-preference"})


def _judge_call(system: str, user: str) -> tuple[bool | None, str]:
    """Run the judge LLM and return (verdict, raw_text)."""
    from configs.settings import (
        ENABLE_THINKING_JUDGE,
        JUDGE_MAX_TOKENS,
        JUDGE_TEMPERATURE,
    )

    resp = _get_judge_llm().chat(
        system=system,
        messages=[{"role": "user", "content": user}],
        max_tokens=JUDGE_MAX_TOKENS,
        temperature=JUDGE_TEMPERATURE,
        chat_template_kwargs={"enable_thinking": ENABLE_THINKING_JUDGE},
    )
    return _parse_verdict(resp.text), resp.text


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


def score_llm_judge(prediction, gold, question, question_type: str = "") -> ScoreResult:
    """Judge by yes/no verdict, dispatching on question_type.

    Most LongMemEval categories use the strict factual-match judge. The
    ``single-session-preference`` category gets a rubric-matching judge
    because its gold strings are preference rubrics, not factual targets
    (a strict judge scores ~6.7% on n=30; rubric judge recovers most of
    those). Accepts non-str inputs (some golds are ints).
    """
    prediction = _coerce(prediction)
    gold = _coerce(gold)
    question = _coerce(question)
    question_type = _coerce(question_type)

    if not gold.strip():
        return ScoreResult(False, "llm_judge", "empty gold answer")
    if not prediction.strip():
        return ScoreResult(False, "llm_judge", "empty prediction")

    if question_type in _RUBRIC_CATEGORIES:
        system = _RUBRIC_JUDGE_SYSTEM
        user = _RUBRIC_JUDGE_USER_TMPL.format(
            question=question.strip() or "(no question provided)",
            gold=gold.strip(),
            prediction=prediction.strip(),
        )
        method_label = "llm_judge_rubric"
    else:
        system = _JUDGE_SYSTEM
        user = _JUDGE_USER_TMPL.format(
            question=question.strip() or "(no question provided)",
            gold=gold.strip(),
            prediction=prediction.strip(),
        )
        method_label = "llm_judge"

    verdict, raw = _judge_call(system, user)
    if verdict is None:
        # Unparseable verdict — fall back to exact_substring so we don't
        # silently mark every borderline case correct or incorrect.
        sub = score_exact_substring(prediction, gold)
        return ScoreResult(
            sub.correct,
            method_label,
            f"unparseable judge output {raw!r}; fell back to exact_substring",
        )
    return ScoreResult(verdict, method_label, f"judge said {raw.strip()!r}")


def score_prediction(
    prediction,
    gold,
    *,
    question: str = "",
    question_type: str = "",
    method: str = "exact_substring",
) -> ScoreResult:
    prediction = _coerce(prediction)
    gold = _coerce(gold)
    question = _coerce(question)
    question_type = _coerce(question_type)
    if method == "exact_substring":
        return score_exact_substring(prediction, gold)
    if method == "llm_judge":
        return score_llm_judge(prediction, gold, question, question_type)
    raise ValueError(f"Unknown scoring method: {method}")
