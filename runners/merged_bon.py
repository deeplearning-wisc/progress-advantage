"""Best-of-N selection scored by progress advantage with a MERGED reference.

The reference policy is interpolated toward the behavior policy in weight space,
    pi_alpha = base + alpha * (pi_theta - base),
optionally sparsified / sign-elected per ``pa.merging`` (linear / ties /
dare_linear / dare_ties / emr). This sweeps alpha in [0.1, 0.9] for each method
and reports how best-of-N success moves as the reference is sharpened from the
base model (alpha->0) toward the policy (alpha->1).

Efficiency: the behavior policy's per-token action log-probs are identical across
every reference, so they are computed ONCE and cached; only the reference is
re-scored per (method, alpha). ``MergedReference.set_alpha`` rewrites the resident
model in place, so each method loads its checkpoints only once. Uses the a-priori
(token=mean, step=mean) aggregation — the same headline metric as the main runs.

Task-shardable (round-robin over sorted task_ids) so a run fans across GPUs;
combine shard outputs with ``runners.combine_merged_bon``.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from collections import defaultdict

import numpy as np

from pa.aggregations import aggregate_steps, aggregate_tokens
from pa.merging import MERGE_METHODS, MergedReference
from pa.scoring import LogprobScorer, StepCache
from pa.trajectory import render_tau2_messages

logger = logging.getLogger(__name__)


def _action_gathered(cache: StepCache) -> list[np.ndarray]:
    return [cache.gathered[i] for i, is_act in enumerate(cache.is_action) if is_act]


def _pa_mean_mean(pol_actions: list[np.ndarray], ref_actions: list[np.ndarray]) -> float:
    rewards: list[float] = []
    for i, p in enumerate(pol_actions):
        p_agg = aggregate_tokens(p, "mean") if p.size > 0 else 0.0
        r_agg = 0.0
        if i < len(ref_actions) and ref_actions[i].size > 0:
            r_agg = aggregate_tokens(ref_actions[i], "mean")
        rewards.append(p_agg - r_agg)
    if not rewards:
        return float("-inf")
    return aggregate_steps(rewards, "mean")


def _bon_success(by_task: dict[str, list[tuple[float, float]]]) -> float:
    picked = [max(items, key=lambda x: x[0])[1] for items in by_task.values()]
    return float(np.mean([r >= 1.0 for r in picked])) if picked else 0.0


def run(
    sims_path: str,
    out_path: str,
    policy_path: str,
    base_path: str,
    methods: list[str],
    alphas: list[float],
    density: float = 0.2,
    dtype: str = "bfloat16",
    device_map: str = "cuda:0",
    max_length: int = 8192,
    task_shard_index: int = 0,
    task_num_shards: int = 1,
) -> dict:
    with open(sims_path) as f:
        sims = json.load(f)["simulations"]
    if task_num_shards > 1:
        task_ids = sorted({s.get("task_id", "") for s in sims})
        keep = {t for i, t in enumerate(task_ids) if i % task_num_shards == task_shard_index}
        sims = [s for s in sims if s.get("task_id", "") in keep]
    logger.info(f"shard {task_shard_index}/{task_num_shards}: {len(sims)} simulations")

    rendered = []
    for s in sims:
        texts, is_action = render_tau2_messages(s.get("messages", []))
        label = float(s.get("reward_info", {}).get("reward") or 0.0)
        rendered.append((texts, is_action, label, s.get("task_id", "")))

    labels_by_task: dict[str, list[float]] = defaultdict(list)
    for _, _, label, tid in rendered:
        labels_by_task[tid].append(label)
    pass1 = float(np.mean([np.mean(v) for v in labels_by_task.values()])) if labels_by_task else 0.0
    oracle = float(np.mean([max(v) >= 1.0 for v in labels_by_task.values()])) if labels_by_task else 0.0
    n_tasks = len(labels_by_task)

    # Cache policy action log-probs once (identical across all references).
    t0 = time.time()
    logger.info(f"caching policy log-probs: {policy_path}")
    pol = LogprobScorer(policy_path, dtype, device_map, top_k=20, max_length=max_length)
    pol_cache = [_action_gathered(pol.score(t, a, compute_topk=False)) for t, a, _, _ in rendered]
    pol.free()
    logger.info(f"cached {len(pol_cache)} policy trajectories in {time.time() - t0:.0f}s")

    results: dict[str, dict] = {}
    for method in methods:
        tm = time.time()
        logger.info(f"[{method}] building merged reference (base={base_path})")
        ref = MergedReference(
            policy_model=policy_path,
            reference_model=base_path,
            method=method,
            density=density,
            dtype=dtype,
            device_map=device_map,
            max_length=max_length,
        )
        per_alpha: dict[str, dict] = {}
        for alpha in alphas:
            ref.set_alpha(alpha)
            by_task: dict[str, list[tuple[float, float]]] = defaultdict(list)
            for idx, (texts, is_action, label, tid) in enumerate(rendered):
                rc = ref.score(texts, is_action, compute_topk=False)
                score = _pa_mean_mean(pol_cache[idx], _action_gathered(rc))
                by_task[tid].append((score, label))
            per_alpha[f"{alpha:.2f}"] = {"bon_success_rate": _bon_success(by_task), "n_tasks": len(by_task)}
            logger.info(f"[{method}] alpha={alpha:.2f} BoN@N={per_alpha[f'{alpha:.2f}']['bon_success_rate']:.3f}")
        ref.free()
        results[method] = per_alpha
        logger.info(f"[{method}] done in {time.time() - tm:.0f}s")

    summary = {
        "sims_path": sims_path,
        "policy_path": policy_path,
        "base_path": base_path,
        "density": density,
        "alphas": alphas,
        "aggregation": {"token_agg": "mean", "step_agg": "mean"},
        "task_shard_index": task_shard_index,
        "task_num_shards": task_num_shards,
        "n_tasks": n_tasks,
        "pass_at_1_mean_of_n": pass1,
        "pass_at_n_oracle": oracle,
        "results": results,
    }
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"wrote {out_path}")
    return summary


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sims", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--policy-path", required=True)
    ap.add_argument("--base-path", required=True, help="merge anchor = the base reference (Qwen2.5-3B)")
    ap.add_argument("--methods", default=",".join(MERGE_METHODS), help="comma-separated subset of MERGE_METHODS")
    ap.add_argument("--alphas", default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9")
    ap.add_argument("--density", type=float, default=0.2)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--device-map", default="cuda:0")
    ap.add_argument("--max-length", type=int, default=8192)
    ap.add_argument("--task-shard-index", type=int, default=0)
    ap.add_argument("--task-num-shards", type=int, default=1)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    alphas = [float(a) for a in args.alphas.split(",") if a.strip()]
    run(
        sims_path=args.sims,
        out_path=args.out,
        policy_path=args.policy_path,
        base_path=args.base_path,
        methods=methods,
        alphas=alphas,
        density=args.density,
        dtype=args.dtype,
        device_map=args.device_map,
        max_length=args.max_length,
        task_shard_index=args.task_shard_index,
        task_num_shards=args.task_num_shards,
    )


if __name__ == "__main__":
    main()
