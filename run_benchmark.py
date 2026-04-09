"""
Tri-Mem Benchmark Runner.

Usage:
    python run_benchmark.py --agent baseline --tasks 5
    python run_benchmark.py --agent rag --tasks 10
    python run_benchmark.py --agent all --tasks 5
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))

from benchmarks.alfworld_sim import ALFWorldSim, get_tasks
from utils.metrics import TaskMetric, BenchmarkResult, TurnMetric
from configs.settings import MAX_AGENT_TURNS


def make_agent(name: str):
    if name == "baseline":
        from agents.baseline_agent import BaselineAgent
        return BaselineAgent()
    elif name == "rag":
        from agents.rag_agent import RAGAgent
        return RAGAgent()
    else:
        raise ValueError(f"Unknown agent: {name}")


def run_task(agent, task: dict, verbose: bool = True) -> TaskMetric:
    """Run a single ALFWorld task with the given agent."""
    env = ALFWorldSim(task)
    agent.reset(task["goal"])

    task_metric = TaskMetric(
        task_id=task["run_id"],
        task_type=task["type"],
        agent_type=agent.name,
    )

    obs = env.reset()
    t_start = time.time()

    if verbose:
        print(f"\n{'='*60}")
        print(f"Task: {task['goal']}")
        print(f"Agent: {agent.name}")
        print(f"{'='*60}")
        print(f"[ENV] {obs[:120]}...")

    for turn in range(MAX_AGENT_TURNS):
        try:
            action, metric = agent.act(obs, turn)
        except Exception as e:
            task_metric.finalize(False, f"Agent error: {e}")
            if verbose:
                print(f"[ERROR] {e}")
            break

        obs, done, success = env.step(action)
        metric.observation = obs

        # Detect errors from environment response
        metric.syntactic_error = obs.strip().lower() == "nothing happens."
        task_metric.add_turn(metric)

        if verbose:
            status = "✓" if not metric.syntactic_error else "✗"
            print(f"  T{turn:02d} {status} {action:40s} → {obs[:60]}")

        if done:
            task_metric.finalize(success)
            break
    else:
        task_metric.finalize(False, "Max turns exceeded")

    task_metric.duration_s = time.time() - t_start

    if verbose:
        result = "SUCCESS" if task_metric.success else f"FAILED ({task_metric.failure_reason})"
        print(f"Result: {result} | Turns: {task_metric.total_turns} | "
              f"Tokens: {task_metric.total_tokens_in + task_metric.total_tokens_out} | "
              f"Syntax errors: {task_metric.syntactic_errors}")

    return task_metric


def run_benchmark(agent_name: str, num_tasks: int, verbose: bool = True) -> BenchmarkResult:
    """Run the full benchmark suite for one agent type."""
    agent = make_agent(agent_name)
    tasks = get_tasks(num_tasks)
    result = BenchmarkResult(agent_type=agent_name)

    print(f"\n{'#'*60}")
    print(f"  BENCHMARK: {agent_name.upper()} agent | {num_tasks} tasks")
    print(f"{'#'*60}")

    for task in tasks:
        task_metric = run_task(agent, task, verbose=verbose)
        result.add_task(task_metric)

    # Save results
    os.makedirs("logs", exist_ok=True)
    ts = int(time.time())
    path = f"logs/{agent_name}_{ts}.json"
    result.save(path)

    print(f"\n{'='*60}")
    print(f"  SUMMARY: {agent_name.upper()}")
    print(f"{'='*60}")
    for k, v in result.summary().items():
        print(f"  {k:30s}: {v}")
    print()

    return result


def main():
    parser = argparse.ArgumentParser(description="Tri-Mem Benchmark Runner")
    parser.add_argument("--agent", type=str, default="baseline",
                        choices=["baseline", "rag", "all"],
                        help="Which agent to benchmark")
    parser.add_argument("--tasks", type=int, default=5, help="Number of tasks")
    parser.add_argument("--quiet", action="store_true", help="Suppress per-turn output")
    args = parser.parse_args()

    agents = ["baseline", "rag"] if args.agent == "all" else [args.agent]
    results = {}

    for agent_name in agents:
        results[agent_name] = run_benchmark(agent_name, args.tasks, verbose=not args.quiet)

    if len(results) > 1:
        print(f"\n{'#'*60}")
        print(f"  COMPARISON")
        print(f"{'#'*60}")
        for name, r in results.items():
            s = r.summary()
            print(f"\n  {name.upper()}:")
            print(f"    Success rate:     {s['success_rate']:.1%}")
            print(f"    Avg turns:        {s['avg_turns']}")
            print(f"    Avg tokens:       {s['avg_tokens']:.0f}")
            print(f"    Syntax errors:    {s['total_syntactic_errors']}")
            print(f"    Cost:             ${s['total_cost_usd']:.4f}")


if __name__ == "__main__":
    main()
