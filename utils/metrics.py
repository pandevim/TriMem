"""Metrics collection and reporting for Tri-Mem experiments."""
import json
import time
from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class TurnMetric:
    turn: int
    action: str
    observation: str
    success: bool
    tokens_in: int = 0
    tokens_out: int = 0
    memory_source: str = "none"        # "msa" | "visual_bus" | "rag" | "none"
    entropy_score: float = -1.0        # -1 = not measured
    latency_ms: float = 0.0
    syntactic_error: bool = False      # env returned "Nothing happens"
    spatial_hallucination: bool = False # tried interacting with wrong object


@dataclass
class TaskMetric:
    task_id: str
    task_type: str
    agent_type: str
    success: bool = False
    total_turns: int = 0
    total_tokens_in: int = 0
    total_tokens_out: int = 0
    total_cost_usd: float = 0.0
    syntactic_errors: int = 0
    spatial_hallucinations: int = 0
    duration_s: float = 0.0
    turns: list = field(default_factory=list)
    failure_reason: str = ""

    def add_turn(self, t: TurnMetric):
        self.turns.append(asdict(t))
        self.total_turns += 1
        self.total_tokens_in += t.tokens_in
        self.total_tokens_out += t.tokens_out
        if t.syntactic_error:
            self.syntactic_errors += 1
        if t.spatial_hallucination:
            self.spatial_hallucinations += 1

    def finalize(self, success: bool, reason: str = ""):
        self.success = success
        self.failure_reason = reason
        # Local inference — no API cost. Track tokens for comparison only.
        self.total_cost_usd = 0.0


@dataclass
class BenchmarkResult:
    agent_type: str
    num_tasks: int = 0
    successes: int = 0
    tasks: list = field(default_factory=list)

    def add_task(self, t: TaskMetric):
        self.tasks.append(asdict(t))
        self.num_tasks += 1
        if t.success:
            self.successes += 1

    @property
    def success_rate(self):
        return self.successes / max(self.num_tasks, 1)

    @property
    def avg_tokens(self):
        if not self.tasks:
            return 0
        return sum(t["total_tokens_in"] + t["total_tokens_out"] for t in self.tasks) / len(self.tasks)

    @property
    def avg_turns(self):
        if not self.tasks:
            return 0
        return sum(t["total_turns"] for t in self.tasks) / len(self.tasks)

    @property
    def total_syntactic_errors(self):
        return sum(t["syntactic_errors"] for t in self.tasks)

    @property
    def total_spatial_hallucinations(self):
        return sum(t["spatial_hallucinations"] for t in self.tasks)

    @property
    def total_cost(self):
        return sum(t["total_cost_usd"] for t in self.tasks)

    def summary(self) -> dict:
        return {
            "agent_type": self.agent_type,
            "num_tasks": self.num_tasks,
            "success_rate": round(self.success_rate, 3),
            "avg_turns": round(self.avg_turns, 1),
            "avg_tokens": round(self.avg_tokens, 0),
            "total_syntactic_errors": self.total_syntactic_errors,
            "total_spatial_hallucinations": self.total_spatial_hallucinations,
            "total_cost_usd": round(self.total_cost, 4),
        }

    def save(self, path: str):
        data = {
            "summary": self.summary(),
            "tasks": self.tasks,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Results saved to {path}")
