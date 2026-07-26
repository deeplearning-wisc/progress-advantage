"""Combine task-sharded ``runners.merged_bon`` summaries into one summary.

Every metric is a mean over tasks, so disjoint task shards combine by a
task-count-weighted average, per (method, alpha).
"""

from __future__ import annotations

import argparse
import glob
import json
import os


def _wavg(values: list[float], weights: list[int]) -> float:
    tot = sum(weights)
    return sum(v * w for v, w in zip(values, weights)) / tot if tot else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard-glob", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    paths = sorted(glob.glob(args.shard_glob))
    if not paths:
        raise SystemExit(f"no shard summaries matching {args.shard_glob}")
    shards = [json.load(open(p)) for p in paths]
    w = [s["n_tasks"] for s in shards]

    base = dict(shards[0])
    base["n_tasks"] = sum(w)
    base["n_shards_combined"] = len(shards)
    base["pass_at_1_mean_of_n"] = _wavg([s["pass_at_1_mean_of_n"] for s in shards], w)
    base["pass_at_n_oracle"] = _wavg([s["pass_at_n_oracle"] for s in shards], w)

    methods = shards[0]["results"].keys()
    combined: dict[str, dict] = {}
    for m in methods:
        alphas = shards[0]["results"][m].keys()
        combined[m] = {
            a: {
                "bon_success_rate": _wavg([s["results"][m][a]["bon_success_rate"] for s in shards], w),
                "n_tasks": sum(w),
            }
            for a in alphas
        }
    base["results"] = combined

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(base, f, indent=2)
    print(f"combined {len(shards)} shards ({sum(w)} tasks) -> {args.out}")


if __name__ == "__main__":
    main()
