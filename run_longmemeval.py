"""
LongMemEval runner.

Usage:
    python run_longmemeval.py --data path/to/longmemeval_oracle.json
    python run_longmemeval.py --data ... --max-tasks 5 --output results.json

Download the dataset from:
    https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict

sys.path.insert(0, os.path.dirname(__file__))

# Match run_benchmark.py: vLLM multiprocessing wants __spec__ on __main__
if not hasattr(sys.modules["__main__"], "__spec__"):
    sys.modules["__main__"].__spec__ = None

from benchmarks.longmemeval import load_tasks
from benchmarks.longmemeval.harness import run_eval


def make_agent(name: str):
    if name == "trimem":
        from agents.longmem_agent import LongMemAgent
        return LongMemAgent()
    raise ValueError(f"Unknown agent: {name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to longmemeval_*.json")
    parser.add_argument("--agent", default="trimem", choices=["trimem"])
    parser.add_argument("--max-tasks", type=int, default=None)
    parser.add_argument(
        "--question-types",
        nargs="+",
        default=None,
        help="Filter to these question_types (e.g. multi-session temporal-reasoning)",
    )
    parser.add_argument(
        "--method",
        default="exact_substring",
        choices=["exact_substring", "llm_judge"],
        help="Scoring method (llm_judge not yet wired)",
    )
    parser.add_argument("--output", default=None, help="Write JSON results here")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    tasks = load_tasks(
        args.data,
        max_tasks=args.max_tasks,
        question_types=args.question_types,
    )
    print(f"Loaded {len(tasks)} tasks from {args.data}", flush=True)
    if not tasks:
        print("No tasks matched the filter.", flush=True)
        return

    agent = make_agent(args.agent)
    result = run_eval(agent, tasks, method=args.method, verbose=not args.quiet)

    print("\n" + "=" * 60)
    print(f"Agent: {result.agent_name}  Method: {result.method}")
    print(f"Overall accuracy: {result.accuracy:.1%}  (n={len(result.outcomes)})")
    print("Per question_type:")
    for qt, stats in sorted(result.per_type().items()):
        print(f"  {qt:30s}  n={stats['n']:3d}  acc={stats['accuracy']:.1%}")
    print("=" * 60)

    if args.output:
        payload = {
            "agent": result.agent_name,
            "method": result.method,
            "accuracy": result.accuracy,
            "per_type": result.per_type(),
            "outcomes": [asdict(o) for o in result.outcomes],
        }
        with open(args.output, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"Wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
