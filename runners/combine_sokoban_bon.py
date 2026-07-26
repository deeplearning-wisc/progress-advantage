"""Combine task-sharded ``runners.sokoban_bon`` summaries into one summary.

Every reported metric is a mean over tasks, so disjoint task shards combine by a
task-count-weighted average. Produces a JSON identical in shape to an unsharded
run, so the report generator is agnostic to whether scoring was sharded.
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
    ap.add_argument("--shard-glob", required=True, help="glob of shard summary JSONs")
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

    methods = shards[0]["bon"].keys()
    base["bon"] = {}
    for m in methods:
        base["bon"][m] = {
            "bon_success_rate": _wavg([s["bon"][m]["bon_success_rate"] for s in shards], w),
            "bon_mean_reward": _wavg([s["bon"][m]["bon_mean_reward"] for s in shards], w),
        }

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(base, f, indent=2)
    print(
        f"combined {len(shards)} shards ({sum(w)} tasks) -> {args.out}  "
        f"PA@N={base['bon']['progress_advantage']['bon_success_rate']:.3f} "
        f"pass@1={base['pass_at_1_mean_of_n']:.3f} oracle={base['pass_at_n_oracle']:.3f}"
    )


if __name__ == "__main__":
    main()
