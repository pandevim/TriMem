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

import json
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
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
    snapshot_path: str | None = None,
) -> HarnessResult:
    """Run the eval loop.

    ``snapshot_path``: if set, write a partial-results JSON after every
    task. Cheap (single-digit kB writes per task) and means a mid-run
    crash leaves N-1 graded outcomes on disk instead of zero.
    """
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

        try:
            score: ScoreResult = score_prediction(
                prediction,
                task.answer,
                question=task.question,
                question_type=task.question_type,
                method=method,
            )
            correct = score.correct
        except Exception as e:
            # Scoring failure shouldn't kill the whole run — record it as
            # incorrect with the exception text and continue. Without this
            # guard a single malformed gold answer ends 500-task jobs.
            print(f"  [scoring error: {e}] — marking task incorrect", flush=True)
            correct = False

        outcome = TaskOutcome(
            question_id=task.question_id,
            question_type=task.question_type,
            question=task.question,
            gold=str(task.answer),
            prediction=prediction,
            correct=correct,
            latency_s=latency,
        )
        result.outcomes.append(outcome)

        if verbose:
            mark = "✓" if outcome.correct else "✗"
            gold_disp = str(task.answer)
            print(
                f"  {mark} pred={prediction[:120]!r}  gold={gold_disp[:80]!r}  "
                f"({latency:.1f}s)",
                flush=True,
            )

        if snapshot_path:
            _write_snapshot(snapshot_path, result)

    return result


def _write_snapshot(path: str, result: HarnessResult) -> None:
    payload = {
        "agent": result.agent_name,
        "method": result.method,
        "accuracy": result.accuracy,
        "per_type": result.per_type(),
        "outcomes": [asdict(o) for o in result.outcomes],
        "_partial": True,
    }
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
    import os
    os.replace(tmp, path)
