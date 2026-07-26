"""Best-of-N selection on Sokoban rollouts, scored by progress advantage.

Unlike ``runners.tau2_bon`` this takes arbitrary policy / reference checkpoint
*paths* (Sokoban RL checkpoints are not in ``pa.models.MODEL_PAIRS``) and a single
simulations file produced by ``agent_r1.eval.run_bon`` /``merge_bon``:

    {"simulations": [{"task_id": "seed_1000000",
                      "messages": [{role, content}, ...],
                      "reward_info": {"reward": 1.0|0.0}}, ...]}

Each trajectory is scored once per model; from the cached per-token log-probs we
evaluate several selection methods (progress advantage over an aggregation grid,
policy log-prob, self-certainty, DeepConf-tail) and, per method, keep the
highest-scoring of the N rollouts of each task. Reported alongside pass@1
(mean-of-N) and the oracle pass@N upper bound.
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
from pa.scoring import LogprobScorer, StepCache
from pa.trajectory import render_tau2_messages

logger = logging.getLogger(__name__)

# Aggregation grid explored for progress advantage. The headline number uses the
# a-priori (mean, mean) cell (length-normalised, no oracle tuning); the rest are
# reported for sensitivity only.
TOKEN_AGGS = ["mean", "min", "max", "last"]
STEP_AGGS = ["mean", "min", "last", "max"]
PRIMARY_TOKEN_AGG = "mean"
PRIMARY_STEP_AGG = "mean"


def _action_token_arrays(cache: StepCache) -> list[np.ndarray]:
    """Per-action gathered log-prob arrays (only scorable action spans)."""
    return [cache.gathered[i] for i, is_act in enumerate(cache.is_action) if is_act]


def _action_topk_arrays(cache: StepCache) -> list[np.ndarray]:
    return [cache.topk_mean[i] for i, is_act in enumerate(cache.is_action) if is_act]


def _pa_score(
    pol_actions: list[np.ndarray],
    ref_actions: list[np.ndarray],
    token_agg: str,
    step_agg: str,
) -> float:
    rewards: list[float] = []
    for i, p in enumerate(pol_actions):
        p_agg = aggregate_tokens(p, token_agg) if p.size > 0 else 0.0
        r_agg = 0.0
        if i < len(ref_actions) and ref_actions[i].size > 0:
            r_agg = aggregate_tokens(ref_actions[i], token_agg)
        rewards.append(p_agg - r_agg)
    if not rewards:
        return float("-inf")
    return aggregate_steps(rewards, step_agg)


def _policy_logprob_score(pol_actions: list[np.ndarray], token_agg: str, step_agg: str) -> float:
    rewards = [aggregate_tokens(p, token_agg) if p.size > 0 else 0.0 for p in pol_actions]
    if not rewards:
        return float("-inf")
    return aggregate_steps(rewards, step_agg)


def _certainty_score(topk_actions: list[np.ndarray], token_agg: str, step_agg: str) -> float:
    rewards = [aggregate_tokens(c, token_agg) for c in topk_actions if c.size > 0]
    if not rewards:
        return float("-inf")
    return aggregate_steps(rewards, step_agg)


def _all_method_scores(pol: StepCache, ref: StepCache) -> dict[str, float]:
    """All selection-method scores for one trajectory from cached log-probs."""
    pol_actions = _action_token_arrays(pol)
    ref_actions = _action_token_arrays(ref)
    topk_actions = _action_topk_arrays(pol)

    scores: dict[str, float] = {}
    for t in TOKEN_AGGS:
        for s in STEP_AGGS:
            scores[f"pa_{t}_{s}"] = _pa_score(pol_actions, ref_actions, t, s)
    scores["progress_advantage"] = scores[f"pa_{PRIMARY_TOKEN_AGG}_{PRIMARY_STEP_AGG}"]
    scores["policy_logprob"] = _policy_logprob_score(pol_actions, "mean", "mean")
    scores["self_certainty"] = _certainty_score(topk_actions, "mean", "mean")
    scores["deepconf_tail"] = _certainty_score(topk_actions, "mean", "last")
    return scores


def _select_best(
    by_task: dict[str, list[tuple[dict[str, float], float]]], method: str
) -> tuple[float, float]:
    """Best-of-N by ``method``: (success_rate, mean_reward_of_selected) over tasks."""
    picked_rewards: list[float] = []
    for _tid, items in by_task.items():
        best = max(items, key=lambda x: x[0][method])
        picked_rewards.append(best[1])
    if not picked_rewards:
        return 0.0, 0.0
    success = float(np.mean([r >= 1.0 for r in picked_rewards]))
    return success, float(np.mean(picked_rewards))


def run(
    sims_path: str,
    out_path: str,
    policy_path: str,
    reference_path: str,
    policy_label: str,
    reference_label: str,
    dtype: str = "bfloat16",
    device_map: str = "cuda:0",
    max_length: int = 16384,
    top_k: int = 20,
    task_shard_index: int = 0,
    task_num_shards: int = 1,
) -> dict:
    with open(sims_path) as f:
        sims = json.load(f)["simulations"]
    if task_num_shards > 1:
        task_ids = sorted({s.get("task_id", "") for s in sims})
        keep = {t for i, t in enumerate(task_ids) if i % task_num_shards == task_shard_index}
        sims = [s for s in sims if s.get("task_id", "") in keep]
        logger.info(f"task shard {task_shard_index}/{task_num_shards}: {len(keep)} tasks kept")
    logger.info(f"loaded {len(sims)} simulations from {sims_path}")

    t0 = time.time()
    logger.info(f"loading pi_theta ({policy_label}) = {policy_path}")
    pol = LogprobScorer(policy_path, dtype, device_map, top_k, max_length)
    logger.info(f"loading pi_ref ({reference_label}) = {reference_path}")
    ref = LogprobScorer(reference_path, dtype, device_map, top_k, max_length)
    logger.info(f"loaded both in {time.time() - t0:.1f}s")

    by_task: dict[str, list[tuple[dict[str, float], float]]] = defaultdict(list)
    labels_by_task: dict[str, list[float]] = defaultdict(list)

    t1 = time.time()
    for n, sim in enumerate(sims):
        label = float(sim.get("reward_info", {}).get("reward") or 0.0)
        texts, is_action = render_tau2_messages(sim.get("messages", []))
        p = pol.score(texts, is_action)
        r = ref.score(texts, is_action)
        scores = _all_method_scores(p, r)
        tid = sim.get("task_id", "")
        by_task[tid].append((scores, label))
        labels_by_task[tid].append(label)
        if (n + 1) % 500 == 0:
            logger.info(f"  scored {n + 1}/{len(sims)} ({time.time() - t1:.0f}s)")
    logger.info(f"scored all {len(sims)} trajectories in {time.time() - t1:.0f}s")

    # Reference-free context metrics.
    n_tasks = len(by_task)
    pass_at_n = float(np.mean([max(v) >= 1.0 for v in labels_by_task.values()])) if labels_by_task else 0.0
    mean_at_n = float(np.mean([np.mean(v) for v in labels_by_task.values()])) if labels_by_task else 0.0
    cand_per_task = int(np.median([len(v) for v in labels_by_task.values()])) if labels_by_task else 0

    method_keys = ["progress_advantage", "policy_logprob", "self_certainty", "deepconf_tail"]
    method_keys += [f"pa_{t}_{s}" for t in TOKEN_AGGS for s in STEP_AGGS]
    bon = {}
    for m in method_keys:
        succ, meanr = _select_best(by_task, m)
        bon[m] = {"bon_success_rate": succ, "bon_mean_reward": meanr}

    summary = {
        "policy_label": policy_label,
        "policy_path": policy_path,
        "reference_label": reference_label,
        "reference_path": reference_path,
        "sims_path": sims_path,
        "n_tasks": n_tasks,
        "candidates_per_task": cand_per_task,
        "pass_at_1_mean_of_n": mean_at_n,
        "pass_at_n_oracle": pass_at_n,
        "primary_method": "progress_advantage",
        "primary_agg": {"token_agg": PRIMARY_TOKEN_AGG, "step_agg": PRIMARY_STEP_AGG},
        "bon": bon,
    }

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    pa = bon["progress_advantage"]["bon_success_rate"]
    logger.info(
        f"  [{policy_label} | ref={reference_label}] "
        f"PA BoN@N={pa:.3f}  pass@1={mean_at_n:.3f}  pass@N(oracle)={pass_at_n:.3f}  "
        f"({n_tasks} tasks x ~{cand_per_task} cand)"
    )
    logger.info(f"wrote {out_path}")

    pol.free()
    ref.free()
    return summary


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sims", required=True, help="merged simulations JSON from agent_r1.eval.merge_bon")
    ap.add_argument("--out", required=True, help="output JSON path")
    ap.add_argument("--policy-path", required=True)
    ap.add_argument("--reference-path", required=True)
    ap.add_argument("--policy-label", default="policy")
    ap.add_argument("--reference-label", default="reference")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--device-map", default="cuda:0")
    ap.add_argument("--max-length", type=int, default=16384)
    ap.add_argument("--top-k", type=int, default=20)
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
    run(
        sims_path=args.sims,
        out_path=args.out,
        policy_path=args.policy_path,
        reference_path=args.reference_path,
        policy_label=args.policy_label,
        reference_label=args.reference_label,
        dtype=args.dtype,
        device_map=args.device_map,
        max_length=args.max_length,
        top_k=args.top_k,
        task_shard_index=args.task_shard_index,
        task_num_shards=args.task_num_shards,
    )


if __name__ == "__main__":
    main()
