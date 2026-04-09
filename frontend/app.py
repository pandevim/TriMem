"""
Tri-Mem Dashboard API.

Serves benchmark results and provides a live agent execution endpoint.
"""
import os
import sys
import json
import glob
import time
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from benchmarks.novacorp_audit_sim import NovaCorpAuditSim, get_tasks, TASK_TEMPLATES
from configs.settings import FRONTEND_PORT

app = Flask(__name__)
CORS(app)

LOGS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")


@app.route("/api/results", methods=["GET"])
def get_results():
    """Return all benchmark result files."""
    os.makedirs(LOGS_DIR, exist_ok=True)
    files = sorted(glob.glob(os.path.join(LOGS_DIR, "*.json")), reverse=True)
    results = []
    for f in files[:20]:  # last 20
        with open(f) as fh:
            data = json.load(fh)
            data["filename"] = os.path.basename(f)
            results.append(data)
    return jsonify(results)


@app.route("/api/tasks", methods=["GET"])
def get_available_tasks():
    """Return available task templates."""
    return jsonify([
        {"id": t["id"], "type": t["type"], "goal": t["goal"],
         "num_steps": len(t["solution_steps"])}
        for t in TASK_TEMPLATES
    ])


@app.route("/api/run_task", methods=["POST"])
def run_single_task():
    """Run a single task with a specified agent (for vibe checks)."""
    data = request.json
    agent_name = data.get("agent", "baseline")
    task_id = data.get("task_id", "heat_apple")

    # Find task
    task = next((t for t in TASK_TEMPLATES if t["id"] == task_id), TASK_TEMPLATES[0])
    task["run_id"] = f"{task['id']}_live"

    try:
        from run_benchmark import make_agent, run_task
        agent = make_agent(agent_name)
        result = run_task(agent, task, verbose=False)
        return jsonify({
            "success": result.success,
            "turns": result.turns,
            "total_turns": result.total_turns,
            "total_tokens": result.total_tokens_in + result.total_tokens_out,
            "syntactic_errors": result.syntactic_errors,
            "spatial_hallucinations": result.spatial_hallucinations,
            "cost_usd": result.total_cost_usd,
            "duration_s": result.duration_s,
            "failure_reason": result.failure_reason,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    os.makedirs(LOGS_DIR, exist_ok=True)
    app.run(host="0.0.0.0", port=FRONTEND_PORT, debug=True)
