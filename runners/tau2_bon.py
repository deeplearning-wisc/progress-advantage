"""Best-of-N selection on tau2-bench from pre-generated rollouts."""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np

from pa.aggregations import aggregate_steps, aggregate_tokens
from pa.models import MODEL_PAIRS
from pa.scoring import LogprobScorer, StepCache
from pa.trajectory import render_tau2_messages

logger = logging.getLogger(__name__)


SCORING_METHODS = ["progress_advantage", "self_certainty", "deepconf_tail",
                   "deepconf_b10", "policy_logprob"]


def _load_trial(trials_dir: str, trial: int, domain: str) -> List[Dict]:
    fpath = os.path.join(trials_dir, f"trial_{trial}", f"{domain}.json")
    if not os.path.exists(fpath):
        return []
    with open(fpath) as f:
        return json.load(f).get("simulations", [])


def _score_one(
    method: str, policy: StepCache, reference: Optional[StepCache],
    token_agg: str, step_agg: str, beta: float,
) -> float:
    rewards: List[float] = []
    if method in ("self_certainty", "deepconf_tail", "deepconf_b10"):
        if method == "deepconf_tail":
            step_agg = "last"
        elif method == "deepconf_b10":
            step_agg = "bottom10"
        for i, is_act in enumerate(policy.is_action):
            if not is_act:
                continue
            c = policy.topk_mean[i]
            if c.size == 0:
                continue
            rewards.append(aggregate_tokens(c, token_agg))
    elif method == "policy_logprob":
        for i, is_act in enumerate(policy.is_action):
            if not is_act:
                continue
            p = policy.gathered[i]
            p_agg = aggregate_tokens(p, token_agg) if p.size > 0 else 0.0
            rewards.append(beta * p_agg)
    elif method == "progress_advantage":
        for i, is_act in enumerate(policy.is_action):
            if not is_act:
                continue
            p = policy.gathered[i]
            p_agg = aggregate_tokens(p, token_agg) if p.size > 0 else 0.0
            r_agg = 0.0
            if reference is not None and i < len(reference.gathered) and reference.gathered[i].size > 0:
                r_agg = aggregate_tokens(reference.gathered[i], token_agg)
            rewards.append(beta * (p_agg - r_agg))
    else:
        raise ValueError(method)
    if not rewards:
        return float("-inf")
    return aggregate_steps(rewards, step_agg)


def run(
    trials_dir: str,
    out_path: str,
    policy_pair: str,
    n_trials: int = 8,
    domain: str = "airline",
    method: str = "progress_advantage",
    token_agg: str = "mean",
    step_agg: str = "mean",
    dtype: str = "bfloat16",
    device_map: str = "cuda:0",
    max_length: int = 16384,
    top_k: int = 20,
    beta: float = 1.0,
) -> Dict:
    pair = MODEL_PAIRS[policy_pair]
    needs_ref = method == "progress_advantage"

    t0 = time.time()
    logger.info(f"loading π_θ = {pair['policy']}")
    pol = LogprobScorer(pair["policy"], dtype, device_map, top_k, max_length)
    ref = None
    if needs_ref:
        logger.info(f"loading π_ref = {pair['reference']}")
        ref = LogprobScorer(pair["reference"], dtype, device_map, top_k, max_length)
    logger.info(f"loaded in {time.time() - t0:.1f}s")

    by_task: Dict[str, List[Tuple[int, Dict, float, float]]] = defaultdict(list)
    mean_at_n_acc: Dict[str, List[float]] = defaultdict(list)

    for trial in range(n_trials):
        sims = _load_trial(trials_dir, trial, domain)
        if not sims:
            logger.warning(f"  trial_{trial}: no rollouts for {domain}")
            continue
        logger.info(f"--- trial {trial}: scoring {len(sims)} simulations ---")
        for sim in sims:
            label = float(sim.get("reward_info", {}).get("reward") or 0.0)
            texts, is_action = render_tau2_messages(sim.get("messages", []))
            p = pol.score(texts, is_action)
            r = ref.score(texts, is_action) if ref is not None else None
            score = _score_one(method, p, r, token_agg, step_agg, beta)
            tid = sim.get("task_id", sim.get("id", ""))
            by_task[tid].append((trial, sim, score, label))
            mean_at_n_acc[tid].append(label)

    selected: List[Dict] = []
    for tid, items in by_task.items():
        items.sort(key=lambda x: x[2], reverse=True)
        best_trial, best_sim, best_score, best_label = items[0]
        selected.append({
            "task_id": tid,
            "selected_trial": best_trial,
            "score": best_score,
            "reward": best_label,
            "n_candidates": len(items),
        })

    n = len(selected)
    # success = reward >= 1.0; for binary-reward tau2 this equals mean reward,
    # for continuous-reward WebShop it recovers the exact-match success rate.
    rewards = [s["reward"] for s in selected]
    bon = float(np.mean([r >= 1.0 for r in rewards])) if rewards else 0.0
    bon_mean_reward = float(np.mean(rewards)) if rewards else 0.0
    # oracle Pass@N: fraction of tasks with at least one successful rollout
    pass_n = float(np.mean([max(v) >= 1.0 for v in mean_at_n_acc.values()])) if mean_at_n_acc else 0.0
    mean_n = float(np.mean([np.mean(v) for v in mean_at_n_acc.values()])) if mean_at_n_acc else 0.0

    summary = {
        "policy_pair": policy_pair,
        "domain": domain,
        "method": method,
        "token_agg": token_agg,
        "step_agg": step_agg,
        "n_trials": n_trials,
        "n_tasks": n,
        "bon_success_rate": bon,
        "bon_mean_reward": bon_mean_reward,
        "pass_at_n_oracle": pass_n,
        "mean_at_n": mean_n,
        "selected": selected,
    }

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(
        f"  BoN(success)={bon:.3f}  BoN(meanR)={bon_mean_reward:.3f}  "
        f"Pass@N(oracle)={pass_n:.3f}  Mean@N={mean_n:.3f}  "
        f"({n} tasks, {n_trials} trials)"
    )
    logger.info(f"wrote {out_path}")

    pol.free()
    if ref is not None:
        ref.free()
    return summary


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials-dir", required=True,
                    help="directory containing trial_<i>/<domain>.json")
    ap.add_argument("--out", required=True, help="output JSON path")
    ap.add_argument("--policy-pair", required=True, choices=list(MODEL_PAIRS.keys()))
    ap.add_argument("--n-trials", type=int, default=8)
    ap.add_argument("--domain", default="airline")
    ap.add_argument("--method", default="progress_advantage", choices=SCORING_METHODS)
    ap.add_argument("--token-agg", default="mean")
    ap.add_argument("--step-agg", default="mean")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--device-map", default="cuda:0")
    ap.add_argument("--max-length", type=int, default=16384)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--beta", type=float, default=1.0)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                        datefmt="%H:%M:%S")
    run(
        trials_dir=args.trials_dir,
        out_path=args.out,
        policy_pair=args.policy_pair,
        n_trials=args.n_trials,
        domain=args.domain,
        method=args.method,
        token_agg=args.token_agg,
        step_agg=args.step_agg,
        dtype=args.dtype,
        device_map=args.device_map,
        max_length=args.max_length,
        top_k=args.top_k,
        beta=args.beta,
    )


if __name__ == "__main__":
    main()
