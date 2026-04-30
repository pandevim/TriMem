"""
LongMemEval dataset loader.

Reads the upstream JSON files distributed at
    https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned

Three variants ship: ``longmemeval_oracle.json`` (only evidence sessions —
useful for sanity-checking the agent end-to-end), ``longmemeval_s_cleaned.json``
(small haystack, ~50 sessions per task), and ``longmemeval_m_cleaned.json``
(medium haystack).

Each instance has the schema documented in the upstream README:
    question_id, question_type, question, answer, question_date,
    haystack_session_ids, haystack_dates, haystack_sessions,
    answer_session_ids
``haystack_sessions`` is a list of sessions; each session is a list of
{role, content} turns.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class LongMemEvalTask:
    question_id: str
    question_type: str
    question: str
    answer: str
    question_date: str = ""
    haystack_session_ids: list[str] = field(default_factory=list)
    haystack_dates: list[str] = field(default_factory=list)
    haystack_sessions: list[list[dict]] = field(default_factory=list)
    answer_session_ids: list[str] = field(default_factory=list)

    @property
    def is_abstention(self) -> bool:
        return self.question_type.endswith("_abs")

    @property
    def num_sessions(self) -> int:
        return len(self.haystack_sessions)

    @property
    def num_turns(self) -> int:
        return sum(len(s) for s in self.haystack_sessions)


def load_tasks(
    path: str | Path,
    *,
    max_tasks: int | None = None,
    question_types: list[str] | None = None,
) -> list[LongMemEvalTask]:
    """Load LongMemEval tasks from a JSON file.

    Args:
        path: path to longmemeval_{oracle,s_cleaned,m_cleaned}.json
        max_tasks: take only the first N tasks (for smoke tests)
        question_types: keep only tasks whose question_type is in this list

    Raises:
        FileNotFoundError if the dataset isn't downloaded yet.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"LongMemEval data not found at {p}. Download from "
            f"https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned"
        )

    with p.open() as f:
        raw = json.load(f)

    tasks: list[LongMemEvalTask] = []
    for item in raw:
        task = LongMemEvalTask(
            question_id=item["question_id"],
            question_type=item["question_type"],
            question=item["question"],
            answer=item.get("answer", ""),
            question_date=item.get("question_date", ""),
            haystack_session_ids=item.get("haystack_session_ids", []),
            haystack_dates=item.get("haystack_dates", []),
            haystack_sessions=item.get("haystack_sessions", []),
            answer_session_ids=item.get("answer_session_ids", []),
        )
        if question_types and task.question_type not in question_types:
            continue
        tasks.append(task)
        if max_tasks and len(tasks) >= max_tasks:
            break
    return tasks
