"""
LongMemEval evaluation loop.

For each task:
    1. agent.reset()
    2. agent.ingest(sessions, session_ids, dates) — populate per-task memories
    3. prediction = agent.answer(question, question_date)
    4. score against the gold answer

Aggregates accuracy overall and per question_type. Token + latency
accounting is delegated to whatever the agent records on its
``last_metric`` (kept simple for now — we'll extend once the agent
plumbing for LongMemEval is hardened).
"""
from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Protocol

from benchmarks.longmemeval.loader import LongMemEvalTask
from benchmarks.longmemeval.scoring import score_prediction, ScoreResult


class LongMemAgentProto(Protocol):
    name: str

    def reset(self) -> None: ...
    def ingest(
        self,
        sessions: list[list[dict]],
        session_ids: list[str],
        dates: list[str],
    ) -> None: ...
    def answer(self, question: str, question_date: str = "") -> str: ...


@dataclass
class TaskOutcome:
    question_id: str
    question_type: str
    question: str
    gold: str
    prediction: str
    correct: bool
    latency_s: float


@dataclass
class HarnessResult:
    agent_name: str
    method: str
    outcomes: list[TaskOutcome] = field(default_factory=list)

    @property
    def accuracy(self) -> float:
        if not self.outcomes:
            return 0.0
        return sum(1 for o in self.outcomes if o.correct) / len(self.outcomes)

    def per_type(self) -> dict[str, dict[str, float]]:
        buckets: dict[str, list[TaskOutcome]] = defaultdict(list)
        for o in self.outcomes:
            buckets[o.question_type].append(o)
        return {
            qt: {
                "n": len(items),
                "accuracy": sum(1 for o in items if o.correct) / len(items),
            }
            for qt, items in buckets.items()
        }


def run_eval(
    agent: LongMemAgentProto,
    tasks: list[LongMemEvalTask],
    *,
    method: str = "exact_substring",
    verbose: bool = True,
) -> HarnessResult:
    result = HarnessResult(agent_name=agent.name, method=method)

    for i, task in enumerate(tasks, 1):
        if verbose:
            print(
                f"\n[{i}/{len(tasks)}] {task.question_id} "
                f"({task.question_type}, {task.num_sessions} sessions)",
                flush=True,
            )

        agent.reset()
        agent.ingest(
            task.haystack_sessions,
            task.haystack_session_ids,
            task.haystack_dates,
        )

        t0 = time.time()
        try:
            prediction = agent.answer(task.question, task.question_date)
        except Exception as e:
            prediction = f"[agent error: {e}]"
        latency = time.time() - t0

        score: ScoreResult = score_prediction(
            prediction, task.answer, question=task.question, method=method
        )
        outcome = TaskOutcome(
            question_id=task.question_id,
            question_type=task.question_type,
            question=task.question,
            gold=task.answer,
            prediction=prediction,
            correct=score.correct,
            latency_s=latency,
        )
        result.outcomes.append(outcome)

        if verbose:
            mark = "✓" if outcome.correct else "✗"
            print(
                f"  {mark} pred={prediction[:120]!r}  gold={task.answer[:80]!r}  "
                f"({latency:.1f}s)",
                flush=True,
            )

    return result
