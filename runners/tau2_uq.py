"""Uncertainty quantification on tau2-bench (Airline / Retail)."""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from scipy import stats
from sklearn.metrics import roc_auc_score

from pa.aggregations import aggregate_steps, aggregate_tokens
from pa.models import MODEL_PAIRS
from pa.scoring import LogprobScorer, StepCache
from pa.trajectory import render_tau2_messages

logger = logging.getLogger(__name__)


SCORING_METHODS = ["progress_advantage", "self_certainty", "deepconf_tail", "deepconf_b10",
                   "policy_logprob", "reference_logprob"]


def _load_simulations(traj_dir: str, domain: str) -> List[Dict]:
    fpath = os.path.join(traj_dir, f"{domain}.json")
    if not os.path.exists(fpath):
        return []
    with open(fpath) as f:
        return json.load(f).get("simulations", [])


def build_caches(
    sims: List[Dict],
    policy_scorer: LogprobScorer,
    ref_scorer: Optional[LogprobScorer],
) -> List[Dict]:
    cache: List[Dict] = []
    for i, sim in enumerate(sims):
        msgs = sim.get("messages", [])
        reward = sim.get("reward_info", {}).get("reward")
        label = float(reward) if reward is not None else None
        texts, is_action = render_tau2_messages(msgs)
        p = policy_scorer.score(texts, is_action)
        r = ref_scorer.score(texts, is_action) if ref_scorer is not None else None
        cache.append({"id": sim.get("id", ""), "label": label, "policy": p, "reference": r})
        if (i + 1) % 10 == 0 or (i + 1) == len(sims):
            logger.info(f"  cached {i + 1}/{len(sims)}")
    return cache


def _score_trajectory(
    method: str,
    policy: StepCache,
    reference: Optional[StepCache],
    token_agg: str,
    step_agg: str,
    beta: float,
) -> Optional[float]:
    """Per-method trajectory scoring.

    Progress Advantage / policy / reference: an action position with an
    empty span (truncated by max_length) still contributes a 0.0 step
    reward. Self-Certainty / DeepConf skip empty positions, since the
    top-K mean is undefined for zero tokens.
    """
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
    elif method == "reference_logprob":
        if reference is None:
            return None
        for i, is_act in enumerate(policy.is_action):
            if not is_act:
                continue
            r = reference.gathered[i] if i < len(reference.gathered) else None
            r_agg = aggregate_tokens(r, token_agg) if r is not None and r.size > 0 else 0.0
            rewards.append(beta * r_agg)
    elif method == "progress_advantage":
        for i, is_act in enumerate(policy.is_action):
            if not is_act:
                continue
            p = policy.gathered[i]
            p_agg = aggregate_tokens(p, token_agg) if p.size > 0 else 0.0
            if reference is not None and i < len(reference.gathered):
                r = reference.gathered[i]
                r_agg = aggregate_tokens(r, token_agg) if r.size > 0 else 0.0
                rewards.append(beta * (p_agg - r_agg))
            else:
                rewards.append(beta * p_agg)
    else:
        raise ValueError(f"unknown method: {method}")
    if not rewards:
        return None
    return aggregate_steps(rewards, step_agg)


def _metrics(scores: List[float], labels: List[float]) -> Dict:
    if len(set(labels)) < 2:
        return {"auroc": float("nan"), "spearman": float("nan"), "n": len(labels)}
    rho, _ = stats.spearmanr(scores, labels)
    try:
        auroc = float(roc_auc_score(labels, scores))
    except ValueError:
        auroc = float("nan")
    return {"auroc": auroc, "spearman": float(rho), "n": len(labels)}


def evaluate(
    cache: List[Dict],
    methods: List[str],
    token_aggs: List[str],
    step_aggs: List[str],
    beta: float = 1.0,
) -> Dict[str, Dict]:
    out: Dict[str, Dict] = {}
    for method in methods:
        for ta in token_aggs:
            for sa in step_aggs:
                scores: List[float] = []
                labels: List[float] = []
                for tc in cache:
                    if tc["label"] is None:
                        continue
                    s = _score_trajectory(method, tc["policy"], tc["reference"], ta, sa, beta)
                    if s is None:
                        continue
                    scores.append(s)
                    labels.append(tc["label"])
                key = f"{method}__tok={ta}__step={sa}"
                out[key] = _metrics(scores, labels)
    return out


def run(
    traj_dir: str,
    out_path: str,
    policy_pair: str,
    behavior_pair: Optional[str] = None,
    domains: Tuple[str, ...] = ("airline", "retail"),
    token_aggs: Tuple[str, ...] = ("mean",),
    step_aggs: Tuple[str, ...] = ("mean", "last", "max", "min", "bottom10"),
    methods: Tuple[str, ...] = tuple(SCORING_METHODS),
    dtype: str = "bfloat16",
    device_map: str = "cuda:0",
    max_length: int = 16384,
    top_k: int = 20,
    beta: float = 1.0,
) -> Dict:
    """Evaluate a (policy, reference) pair on cached behavior trajectories."""
    pair = MODEL_PAIRS[policy_pair]
    behavior_pair = behavior_pair or policy_pair

    t0 = time.time()
    logger.info(f"loading π_θ = {pair['policy']}")
    policy_scorer = LogprobScorer(pair["policy"], dtype, device_map, top_k, max_length)
    logger.info(f"loading π_ref = {pair['reference']}")
    ref_scorer = LogprobScorer(pair["reference"], dtype, device_map, top_k, max_length)
    logger.info(f"models loaded in {time.time() - t0:.1f}s")

    results: Dict = {
        "policy_pair": policy_pair,
        "behavior_pair": behavior_pair,
        "policy_model": pair["policy"],
        "reference_model": pair["reference"],
        "traj_dir": traj_dir,
        "domains": {},
    }
    for domain in domains:
        sims = _load_simulations(traj_dir, domain)
        if not sims:
            logger.warning(f"  {domain}: no trajectories — skipping")
            continue
        logger.info(f"--- {domain}: {len(sims)} trajectories ---")
        t1 = time.time()
        cache = build_caches(sims, policy_scorer, ref_scorer)
        logger.info(f"  cache built in {time.time() - t1:.1f}s")
        combos = evaluate(cache, list(methods), list(token_aggs), list(step_aggs), beta=beta)
        for key, m in combos.items():
            logger.info(f"  {key:>56s}  AUROC={m['auroc']:.3f}  ρ={m['spearman']:.3f}  n={m['n']}")
        results["domains"][domain] = {"n": len(sims), "combos": combos}

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"wrote {out_path}")

    policy_scorer.free()
    ref_scorer.free()
    return results


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj-dir", required=True,
                    help="directory containing {airline,retail}.json")
    ap.add_argument("--out", required=True, help="output JSON path")
    ap.add_argument("--policy-pair", required=True, choices=list(MODEL_PAIRS.keys()))
    ap.add_argument("--behavior-pair", default=None,
                    help="label for the trajectory generator (off-policy UQ)")
    ap.add_argument("--domains", nargs="+", default=["airline", "retail"])
    ap.add_argument("--methods", nargs="+", default=list(SCORING_METHODS),
                    choices=SCORING_METHODS)
    ap.add_argument("--token-aggs", nargs="+", default=["mean"])
    ap.add_argument("--step-aggs", nargs="+", default=["mean", "last", "max", "min", "bottom10"])
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
        traj_dir=args.traj_dir,
        out_path=args.out,
        policy_pair=args.policy_pair,
        behavior_pair=args.behavior_pair,
        domains=tuple(args.domains),
        token_aggs=tuple(args.token_aggs),
        step_aggs=tuple(args.step_aggs),
        methods=tuple(args.methods),
        dtype=args.dtype,
        device_map=args.device_map,
        max_length=args.max_length,
        top_k=args.top_k,
        beta=args.beta,
    )


if __name__ == "__main__":
    main()
