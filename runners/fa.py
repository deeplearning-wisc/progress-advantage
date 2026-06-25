"""Failure attribution (Who & When step-level accuracy)."""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from pa.aggregations import aggregate_tokens
from pa.models import MODEL_PAIRS
from pa.scoring import LogprobScorer, StepCache
from pa.trajectory import render_fa_messages

logger = logging.getLogger(__name__)


SPLITS = ["Hand-Crafted", "Algorithm-Generated"]
SCORING_METHODS = ["progress_advantage", "self_certainty", "policy_logprob"]


def _load_split(data_dir: str, split: str) -> List[Dict]:
    files = sorted(glob.glob(os.path.join(data_dir, split, "*.json")))
    samples: List[Dict] = []
    for fp in files:
        with open(fp) as f:
            sample = json.load(f)
        sample["_file"] = os.path.basename(fp)
        if sample.get("mistake_step") is None:
            continue
        samples.append(sample)
    return samples


def _step_rewards(
    method: str, policy: StepCache, reference: Optional[StepCache],
    is_action: List[bool], token_agg: str, beta: float,
) -> List[float]:
    out: List[float] = []
    for i, p in enumerate(policy.gathered):
        if not is_action[i]:
            out.append(0.0)
            continue
        if method == "progress_advantage":
            p_agg = aggregate_tokens(p, token_agg) if p.size > 0 else 0.0
            r_agg = 0.0
            if reference is not None and i < len(reference.gathered) and reference.gathered[i].size > 0:
                r_agg = aggregate_tokens(reference.gathered[i], token_agg)
            out.append(beta * (p_agg - r_agg))
        elif method == "policy_logprob":
            out.append(beta * (aggregate_tokens(p, token_agg) if p.size > 0 else 0.0))
        elif method == "self_certainty":
            c = policy.topk_mean[i]
            out.append(aggregate_tokens(c, token_agg) if c.size > 0 else 0.0)
        else:
            raise ValueError(method)
    return out


def _predict_min_cumulative(rewards: List[float], valid: List[bool]) -> int:
    cum = np.cumsum(rewards)
    best_i, best_v = -1, float("inf")
    for i, ok in enumerate(valid):
        if ok and cum[i] < best_v:
            best_i, best_v = i, cum[i]
    if best_i < 0:
        return next((i for i, ok in enumerate(valid) if ok), 0)
    return best_i


def _predict_sharpest_drop(rewards: List[float], valid: List[bool]) -> int:
    valid_idx = [i for i, ok in enumerate(valid) if ok]
    if not valid_idx:
        return 0
    smallest = valid_idx[0]
    if all(rewards[i] >= 0 for i in valid_idx):
        return smallest
    best_i, best_v = -1, float("inf")
    for i in valid_idx:
        if rewards[i] < best_v:
            best_i, best_v = i, rewards[i]
    return best_i if best_i >= 0 else smallest


def _predict_sharpest_abs_drop(rewards: List[float], valid: List[bool]) -> int:
    valid_idx = [i for i, ok in enumerate(valid) if ok]
    if not valid_idx:
        return 0
    smallest = valid_idx[0]
    if len(valid_idx) < 2:
        return smallest
    best_i, best_v = -1, float("inf")
    any_nonpos = False
    for j in range(1, len(valid_idx)):
        delta = rewards[valid_idx[j]] - rewards[valid_idx[j - 1]]
        if delta <= 0:
            any_nonpos = True
        if delta < best_v:
            best_i, best_v = valid_idx[j], delta
    if not any_nonpos:
        return smallest
    return best_i if best_i >= 0 else smallest


def evaluate_split(
    samples: List[Dict],
    split: str,
    pol: LogprobScorer,
    ref: Optional[LogprobScorer],
    methods: List[str],
    token_aggs: List[str],
    beta: float,
) -> Dict:
    cache: List[Dict] = []
    for i, sample in enumerate(samples):
        texts, is_action = render_fa_messages(sample["history"], split)
        p = pol.score(texts, is_action)
        r = ref.score(texts, is_action) if ref is not None else None
        cache.append({
            "id": sample.get("question_ID", sample.get("_file", "")),
            "mistake_step": int(sample["mistake_step"]),
            "is_action": is_action,
            "policy": p,
            "reference": r,
        })
        if (i + 1) % 5 == 0 or (i + 1) == len(samples):
            logger.info(f"  [{split}] cached {i + 1}/{len(samples)}")

    combos: Dict[str, Dict] = {}
    for method in methods:
        for ta in token_aggs:
            preds_mc, preds_sd, preds_sad, gts = [], [], [], []
            for tc in cache:
                rewards = _step_rewards(method, tc["policy"], tc["reference"],
                                        tc["is_action"], ta, beta)
                preds_mc.append(_predict_min_cumulative(rewards, tc["is_action"]))
                preds_sd.append(_predict_sharpest_drop(rewards, tc["is_action"]))
                preds_sad.append(_predict_sharpest_abs_drop(rewards, tc["is_action"]))
                gts.append(tc["mistake_step"])

            def acc(preds: List[int]) -> Dict:
                d = [abs(p - g) for p, g in zip(preds, gts)]
                return {
                    "accuracy": float(np.mean([x == 0 for x in d])) if d else 0.0,
                    "mae": float(np.mean(d)) if d else 0.0,
                    "n": len(d),
                }

            key = f"{method}__tok={ta}"
            combos[key] = {
                "min_cumulative": acc(preds_mc),
                "sharpest_drop": acc(preds_sd),
                "sharpest_abs_drop": acc(preds_sad),
            }
            logger.info(
                f"  {key:>40s}  MC.acc={combos[key]['min_cumulative']['accuracy']:.3f}  "
                f"SD.acc={combos[key]['sharpest_drop']['accuracy']:.3f}  "
                f"SAD.acc={combos[key]['sharpest_abs_drop']['accuracy']:.3f}"
            )
    return {"n_traj": len(cache), "combos": combos}


def run(
    data_dir: str,
    out_path: str,
    policy_pair: str,
    splits: Tuple[str, ...] = tuple(SPLITS),
    methods: Tuple[str, ...] = tuple(SCORING_METHODS),
    token_aggs: Tuple[str, ...] = ("min",),
    dtype: str = "bfloat16",
    device_map: str = "cuda:0",
    max_length: int = 16384,
    top_k: int = 20,
    beta: float = 1.0,
) -> Dict:
    pair = MODEL_PAIRS[policy_pair]
    needs_ref = "progress_advantage" in methods

    t0 = time.time()
    pol = LogprobScorer(pair["policy"], dtype, device_map, top_k, max_length)
    ref = LogprobScorer(pair["reference"], dtype, device_map, top_k, max_length) if needs_ref else None
    logger.info(f"loaded models in {time.time() - t0:.1f}s")

    out: Dict = {"policy_pair": policy_pair, "splits": {}}
    for split in splits:
        samples = _load_split(data_dir, split)
        logger.info(f"--- {split}: {len(samples)} trajectories ---")
        out["splits"][split] = evaluate_split(
            samples, split, pol, ref, list(methods), list(token_aggs), beta
        )

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    logger.info(f"wrote {out_path}")

    pol.free()
    if ref is not None:
        ref.free()
    return out


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True,
                    help="path to Who&When (containing Hand-Crafted/ and Algorithm-Generated/)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--policy-pair", required=True, choices=list(MODEL_PAIRS.keys()))
    ap.add_argument("--splits", nargs="+", default=list(SPLITS), choices=SPLITS)
    ap.add_argument("--methods", nargs="+", default=list(SCORING_METHODS), choices=SCORING_METHODS)
    ap.add_argument("--token-aggs", nargs="+", default=["min"])
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
        data_dir=args.data_dir,
        out_path=args.out,
        policy_pair=args.policy_pair,
        splits=tuple(args.splits),
        methods=tuple(args.methods),
        token_aggs=tuple(args.token_aggs),
        dtype=args.dtype,
        device_map=args.device_map,
        max_length=args.max_length,
        top_k=args.top_k,
        beta=args.beta,
    )


if __name__ == "__main__":
    main()
