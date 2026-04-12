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

# Fix for vLLM multiprocessing: __spec__ must exist on __main__
if not hasattr(sys.modules["__main__"], "__spec__"):
    sys.modules["__main__"].__spec__ = None

sys.path.insert(0, os.path.dirname(__file__))

from benchmarks.novacorp_audit_sim import NovaCorpAuditSim, get_tasks
from utils.metrics import TaskMetric, BenchmarkResult, TurnMetric
from configs.settings import MAX_AGENT_TURNS


def make_agent(name: str):
    if name == "baseline":
        from agents.baseline_agent import BaselineAgent
        return BaselineAgent()
    elif name == "rag":
        from agents.rag_agent import RAGAgent
        return RAGAgent()
    elif name == "visual_bus":
        from agents.visual_bus_agent import VisualBusAgent
        return VisualBusAgent()
    elif name == "visual_bus_rag":
        from agents.visual_bus_rag_agent import VisualBusRAGAgent
        return VisualBusRAGAgent()
    elif name == "msa":
        from agents.msa_agent import MSAAgent
        return MSAAgent()
    else:
        raise ValueError(f"Unknown agent: {name}")


def run_task(agent, task: dict, task_num: int = 0,
             total_tasks: int = 0, verbose: bool = True) -> TaskMetric:
    """Run a single audit task with the given agent."""
    env = NovaCorpAuditSim(task)
    agent.reset(task["goal"])

    task_metric = TaskMetric(
        task_id=task["run_id"],
        task_type=task["type"],
        agent_type=agent.name,
    )

    obs = env.reset()
    t_start = time.time()

    progress = f"[{task_num}/{total_tasks}]" if total_tasks else ""
    if verbose:
        print(f"\n{'='*60}", flush=True)
        print(f"Task {progress}: {task['goal']}", flush=True)
        print(f"Agent: {agent.name} | Type: {task['type']} | ID: {task['run_id']}", flush=True)
        print(f"{'='*60}", flush=True)
        print(f"[ENV] {obs[:120]}...", flush=True)

    for turn in range(MAX_AGENT_TURNS):
        if verbose:
            elapsed = time.time() - t_start
            print(f"  [Turn {turn:02d}/{MAX_AGENT_TURNS}] "
                  f"(elapsed: {elapsed:.1f}s) Thinking …",
                  end=" ", flush=True)

        try:
            action, metric = agent.act(obs, turn)
        except Exception as e:
            task_metric.finalize(False, f"Agent error: {e}")
            if verbose:
                print(f"\n[ERROR] {e}", flush=True)
            break

        obs, done, success = env.step(action)
        metric.observation = obs

        # Per-turn success: action advanced the solution (no error response)
        error_signals = ("syntax error", "returned no results", "access denied")
        metric.success = not any(s in obs.lower() for s in error_signals)
        metric.syntactic_error = "syntax error" in obs.lower()
        task_metric.add_turn(metric)

        if verbose:
            # Show raw LLM output (strip thinking block if present)
            raw = metric.llm_output
            if "</think>" in raw:
                raw = raw.rsplit("</think>", 1)[-1].strip()
            print(f"  [LLM] {raw[:200]}", flush=True)
            status = "✓" if metric.success else "✗"
            print(f"  {status} {action:40s} → {obs[:60]}", flush=True)

        if done:
            task_metric.finalize(success)
            break
    else:
        task_metric.finalize(False, "Max turns exceeded")

    task_metric.duration_s = time.time() - t_start

    if verbose:
        result = "SUCCESS" if task_metric.success else f"FAILED ({task_metric.failure_reason})"
        print(f"Result: {result} | Turns: {task_metric.total_turns} | "
              f"Duration: {task_metric.duration_s:.1f}s | "
              f"Tokens: {task_metric.total_tokens_in + task_metric.total_tokens_out} | "
              f"Syntax errors: {task_metric.syntactic_errors}", flush=True)

    return task_metric


def run_benchmark(agent_name: str, num_tasks: int, verbose: bool = True) -> BenchmarkResult:
    """Run the full benchmark suite for one agent type."""
    print(f"\n[INIT] Creating {agent_name} agent …", flush=True)
    agent = make_agent(agent_name)
    print(f"[INIT] Loading {num_tasks} tasks …", flush=True)
    tasks = get_tasks(num_tasks)
    result = BenchmarkResult(agent_type=agent_name)

    print(f"\n{'#'*60}", flush=True)
    print(f"  BENCHMARK: {agent_name.upper()} agent | {num_tasks} tasks", flush=True)
    print(f"{'#'*60}", flush=True)

    bench_start = time.time()
    for i, task in enumerate(tasks, 1):
        task_metric = run_task(agent, task, task_num=i,
                               total_tasks=num_tasks, verbose=verbose)
        result.add_task(task_metric)
        elapsed = time.time() - bench_start
        done_count = i
        avg_per_task = elapsed / done_count
        remaining = avg_per_task * (num_tasks - done_count)
        print(f"[PROGRESS] {done_count}/{num_tasks} tasks done | "
              f"Elapsed: {elapsed:.1f}s | "
              f"Est. remaining: {remaining:.1f}s", flush=True)

    # Save results
    os.makedirs("logs", exist_ok=True)
    ts = int(time.time())
    path = f"logs/{agent_name}_{ts}.json"
    result.save(path)
    print(f"[SAVE] Results written to {path}", flush=True)

    total_elapsed = time.time() - bench_start
    print(f"\n{'='*60}", flush=True)
    print(f"  SUMMARY: {agent_name.upper()} ({total_elapsed:.1f}s total)", flush=True)
    print(f"{'='*60}", flush=True)
    for k, v in result.summary().items():
        print(f"  {k:30s}: {v}", flush=True)
    print(flush=True)

    return result


def main():
    parser = argparse.ArgumentParser(description="Tri-Mem Benchmark Runner")
    parser.add_argument("--agent", type=str, default="baseline",
                        choices=["baseline", "rag", "visual_bus", "visual_bus_rag", "msa", "all"],
                        help="Which agent to benchmark")
    parser.add_argument("--tasks", type=int, default=5, help="Number of tasks")
    parser.add_argument("--quiet", action="store_true", help="Suppress per-turn output")
    args = parser.parse_args()

    agents = ["baseline", "rag", "visual_bus", "visual_bus_rag", "msa"] if args.agent == "all" else [args.agent]
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
