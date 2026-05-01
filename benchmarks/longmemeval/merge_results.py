"""Merge two LongMemEval result JSONs (e.g. oracle + s_cleaned) into a
single per-split + combined report for the paper.

Usage:
    python -m benchmarks.longmemeval.merge_results \
        --oracle logs/longmemeval_oracle_paper.json \
        --s logs/longmemeval_s_paper.json \
        --out logs/longmemeval_paper_combined.json

The combined view keeps each split's per_type breakdown intact and
adds a 'combined' block that aggregates across both. The split-level
numbers are what the paper cites; combined is for sanity.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict


def _summarize(outcomes: list[dict]) -> dict:
    if not outcomes:
        return {"n": 0, "accuracy": 0.0, "per_type": {}}
    by_type: dict[str, list[dict]] = defaultdict(list)
    for o in outcomes:
        by_type[o["question_type"]].append(o)
    per_type = {
        qt: {
            "n": len(items),
            "accuracy": sum(1 for o in items if o["correct"]) / len(items),
        }
        for qt, items in by_type.items()
    }
    return {
        "n": len(outcomes),
        "accuracy": sum(1 for o in outcomes if o["correct"]) / len(outcomes),
        "per_type": per_type,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oracle", required=True, help="Path to oracle result JSON")
    ap.add_argument("--s", required=True, help="Path to s_cleaned result JSON")
    ap.add_argument("--out", required=True, help="Output combined JSON")
    args = ap.parse_args()

    oracle = json.load(open(args.oracle))
    s = json.load(open(args.s))

    combined_outcomes = oracle["outcomes"] + s["outcomes"]
    payload = {
        "agent": oracle.get("agent") or s.get("agent"),
        "method": oracle.get("method") or s.get("method"),
        "splits": {
            "oracle": _summarize(oracle["outcomes"]),
            "s_cleaned": _summarize(s["outcomes"]),
        },
        "combined": _summarize(combined_outcomes),
    }
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"Wrote {args.out}")
    print()
    for split in ("oracle", "s_cleaned", "combined"):
        block = payload[split] if split == "combined" else payload["splits"][split]
        print(f"=== {split} (n={block['n']}, acc={block['accuracy']:.1%}) ===")
        for qt, stats in sorted(block["per_type"].items()):
            print(f"  {qt:30s}  n={stats['n']:4d}  acc={stats['accuracy']:.1%}")
        print()


if __name__ == "__main__":
    main()
