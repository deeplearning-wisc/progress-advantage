"""Off-the-shelf reward-model baselines for the three scenarios.

ThinkPRM (``launch/ThinkPRM-{7B,14B}``) applies to UQ, TTS best-of-N, and
FA; WildReward-8B applies to TTS best-of-N and FA. ThinkPRM is a process
RM yielding per-step P(correct); WildReward is an outcome RM yielding one
trajectory scalar (for FA we score growing prefixes to get a step signal).
ThinkPRM needs vLLM. Select with ``--task`` and ``--model``.
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score

from pa.aggregations import aggregate_steps
from pa.baselines import (
    THINKPRM_7B_ID,
    WILDREWARD_ID,
    ThinkPRMScorer,
    WildRewardScorer,
    initial_user_query,
)

logger = logging.getLogger(__name__)


def _make_thinkprm(model_id: Optional[str], template: str) -> ThinkPRMScorer:
    return ThinkPRMScorer(model_id=model_id or THINKPRM_7B_ID, template=template)


# ───────────────────────── UQ (ThinkPRM only) ─────────────────────────

UQ_STEP_AGGS = ["mean", "last", "min", "bottom10"]


def _auroc_spearman(scores: List[float], labels: List[float]) -> Dict:
    if len(set(labels)) < 2:
        return {"auroc": float("nan"), "spearman": float("nan"), "n": len(labels)}
    rho, _ = stats.spearmanr(scores, labels)
    try:
        auroc = float(roc_auc_score(labels, scores))
    except ValueError:
        auroc = float("nan")
    return {"auroc": auroc, "spearman": float(rho), "n": len(labels)}


def run_uq(traj_dir: str, out_path: str, model: str, model_id: Optional[str],
           domains: Tuple[str, ...]) -> Dict:
    if model == "thinkprm":
        scorer = _make_thinkprm(model_id, template="agent")
        name = (model_id or THINKPRM_7B_ID).split("/")[-1]
    else:
        scorer = WildRewardScorer(model_id=WILDREWARD_ID)
        name = "wildreward-8b"
    results: Dict = {"model": name, "domains": {}}
    for domain in domains:
        fpath = os.path.join(traj_dir, f"{domain}.json")
        if not os.path.exists(fpath):
            logger.warning(f"  {domain}: missing — skipping")
            continue
        sims = json.load(open(fpath)).get("simulations", [])
        valid = [s for s in sims if s.get("reward_info", {}).get("reward") is not None]
        labels = [float(s["reward_info"]["reward"]) for s in valid]
        combos: Dict[str, Dict] = {}
        if model == "thinkprm":
            # process RM: per-step P(correct) aggregated to a trajectory scalar
            per_step = scorer.score_trajectories(
                [(initial_user_query(s["messages"]), s["messages"]) for s in valid])
            for sa in UQ_STEP_AGGS:
                sc, lb = [], []
                for ss, label in zip(per_step, labels):
                    if not ss:
                        continue
                    sc.append(aggregate_steps(ss, sa))
                    lb.append(label)
                combos[f"thinkprm__step={sa}"] = _auroc_spearman(sc, lb)
        else:
            # outcome RM: one satisfaction scalar per trajectory
            scores = [scorer.score_messages(s["messages"]) for s in valid]
            combos["wildreward"] = _auroc_spearman(scores, labels)
        for k, m in combos.items():
            logger.info(f"  {domain} {k:>22s}  AUROC={m['auroc']:.3f}  n={m['n']}")
        results["domains"][domain] = {"n": len(sims), "combos": combos}
    return results


# ─────────────────── TTS best-of-N (ThinkPRM + WildReward) ───────────────────


def _score_rollouts(model: str, scorer, messages_list: List[List[Dict]],
                    step_agg: str) -> List[float]:
    if model == "thinkprm":
        pairs = [(initial_user_query(m), m) for m in messages_list]
        per_step = scorer.score_trajectories(pairs)
        return [aggregate_steps(ss, step_agg) if ss else float("-inf")
                for ss in per_step]
    return [scorer.score_messages(m) for m in messages_list]


def run_bon(trials_dir: str, out_path: str, model: str, model_id: Optional[str],
            domain: str, n_trials: int, step_agg: str) -> Dict:
    scorer = (_make_thinkprm(model_id, template="agent") if model == "thinkprm"
              else WildRewardScorer(model_id=WILDREWARD_ID))

    flat: List[Tuple[str, float, List[Dict]]] = []
    for trial in range(n_trials):
        fpath = os.path.join(trials_dir, f"trial_{trial}", f"{domain}.json")
        if not os.path.exists(fpath):
            continue
        for sim in json.load(open(fpath)).get("simulations", []):
            tid = sim.get("task_id", sim.get("id", ""))
            reward = float(sim.get("reward_info", {}).get("reward") or 0.0)
            flat.append((tid, reward, sim.get("messages", [])))

    scores = _score_rollouts(model, scorer, [m for _, _, m in flat], step_agg)
    by_task: Dict[str, List[Tuple[float, float]]] = defaultdict(list)
    for (tid, reward, _), sc in zip(flat, scores):
        by_task[tid].append((sc, reward))

    selected = [max(items, key=lambda x: x[0])[1] for items in by_task.values()]
    bon = float(np.mean([r >= 1.0 for r in selected])) if selected else 0.0
    bon_mean = float(np.mean(selected)) if selected else 0.0
    logger.info(f"  {model} BoN(success)={bon:.3f}  BoN(meanR)={bon_mean:.3f}  "
                f"({len(selected)} tasks)")
    return {"model": model, "domain": domain, "step_agg": step_agg,
            "n_tasks": len(selected), "bon_success_rate": bon,
            "bon_mean_reward": bon_mean}


# ───────────────────────── FA (ThinkPRM only) ─────────────────────────
# WildReward is an outcome RM with no per-step signal, so it is N/A here.

FA_SPLITS = ["Hand-Crafted", "Algorithm-Generated"]


def _is_observation(role: str, idx: int, split: str) -> bool:
    return split == "Hand-Crafted" and idx == 0 and role.lower() == "human"


def _fa_numbered_steps(sample: Dict, split: str,
                       max_chars: int = 20000) -> Tuple[str, int, List[bool]]:
    blocks, is_action, n = [], [], 0
    for i, msg in enumerate(sample.get("history", [])):
        role, content = msg.get("role", ""), str(msg.get("content", "") or "")
        obs = _is_observation(role, i, split)
        is_action.append(not obs)
        if obs:
            continue
        n += 1
        blocks.append(f"Step {n} [{role}]:\n{content}")
    solution = "\n\n".join(blocks)
    return solution[-max_chars:] if len(solution) > max_chars else solution, n, is_action


def _predict_min_step(r: List[float], is_action: List[bool]) -> int:
    best_i, best_v = -1, float("inf")
    for i, ok in enumerate(is_action):
        if ok and i < len(r) and r[i] < best_v:
            best_i, best_v = i, r[i]
    return best_i if best_i >= 0 else 0


def _predict_sharpest_abs_drop(r: List[float], is_action: List[bool]) -> int:
    idx = [i for i, ok in enumerate(is_action) if ok and i < len(r)]
    if len(idx) < 2:
        return idx[0] if idx else 0
    best_i, best_v, any_nonpos = -1, float("inf"), False
    for j in range(1, len(idx)):
        delta = r[idx[j]] - r[idx[j - 1]]
        any_nonpos |= delta <= 0
        if delta < best_v:
            best_i, best_v = idx[j], delta
    return (best_i if best_i >= 0 else idx[0]) if any_nonpos else idx[0]


def _predict_earliest_below(r: List[float], is_action: List[bool],
                            threshold: float) -> int:
    for i, ok in enumerate(is_action):
        if ok and i < len(r) and r[i] < threshold:
            return i
    return _predict_min_step(r, is_action)


def _fa_metrics(preds: List[int], gts: List[int]) -> Dict:
    d = [abs(p - g) for p, g in zip(preds, gts)]
    return {"accuracy": float(np.mean([x == 0 for x in d])) if d else 0.0,
            "mae": float(np.mean(d)) if d else 0.0, "n": len(d)}


def _fa_step_scores(scorer, samples: List[Dict],
                    split: str) -> Tuple[List[List[float]], List[List[bool]]]:
    """Per-history-index ThinkPRM P(correct); observation steps get 1.0 (max)."""
    prompts, n_expected, is_action_per = [], [], []
    for s in samples:
        solution, n, is_action = _fa_numbered_steps(s, split)
        prompts.append(scorer.build_prompt(task=s.get("question", ""), solution=solution))
        n_expected.append(n)
        is_action_per.append(is_action)
    action_scores = scorer.score_prompts(prompts, n_expected)

    full = []
    for k, scores in enumerate(action_scores):
        is_action = is_action_per[k]
        if not scores:
            scores = [0.5] * sum(is_action)
        row, ai = [], 0
        for i in range(len(is_action)):
            if is_action[i] and ai < len(scores):
                row.append(scores[ai]); ai += 1
            else:
                row.append(1.0)
        full.append(row)
    return full, is_action_per


def run_fa(data_dir: str, out_path: str, model_id: Optional[str],
           splits: Tuple[str, ...], threshold: float) -> Dict:
    scorer = _make_thinkprm(model_id, template="fa")
    results: Dict = {"model": (model_id or THINKPRM_7B_ID).split("/")[-1],
                     "threshold": threshold, "splits": {}}
    for split in splits:
        files = sorted(glob.glob(os.path.join(data_dir, split, "*.json")))
        samples = [json.load(open(f)) for f in files]
        samples = [s for s in samples if s.get("mistake_step") is not None]
        logger.info(f"--- {split}: {len(samples)} trajectories ---")
        full, is_action_per = _fa_step_scores(scorer, samples, split)

        gts = [int(s["mistake_step"]) for s in samples]
        preds_ms = [_predict_min_step(full[i], is_action_per[i]) for i in range(len(samples))]
        preds_sad = [_predict_sharpest_abs_drop(full[i], is_action_per[i]) for i in range(len(samples))]
        preds_th = [_predict_earliest_below(full[i], is_action_per[i], threshold)
                    for i in range(len(samples))]
        combos = {"min_step": _fa_metrics(preds_ms, gts),
                  "sharpest_abs_drop": _fa_metrics(preds_sad, gts),
                  "earliest_below_threshold": _fa_metrics(preds_th, gts)}
        for k, m in combos.items():
            logger.info(f"  {k:>26s}  Acc={m['accuracy']:.3f}  MAE={m['mae']:.2f}  n={m['n']}")
        results["splits"][split] = {"n": len(samples), **combos}
    return results


# ───────────────────────────── driver ─────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["uq", "bon", "fa"])
    ap.add_argument("--model", required=True, choices=["thinkprm", "wildreward"])
    ap.add_argument("--model-id", default=None,
                    help="ThinkPRM HF id (launch/ThinkPRM-7B or -14B)")
    ap.add_argument("--out", required=True)
    # uq / bon
    ap.add_argument("--traj-dir")
    ap.add_argument("--trials-dir")
    ap.add_argument("--domains", nargs="+", default=["airline", "retail"])
    ap.add_argument("--domain", default="webshop")
    ap.add_argument("--n-trials", type=int, default=8)
    ap.add_argument("--step-agg", default="mean")
    # fa
    ap.add_argument("--data-dir")
    ap.add_argument("--splits", nargs="+", default=FA_SPLITS, choices=FA_SPLITS)
    ap.add_argument("--threshold", type=float, default=None)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                        datefmt="%H:%M:%S")

    if args.task == "uq":
        out = run_uq(args.traj_dir, args.out, args.model, args.model_id,
                     tuple(args.domains))
    elif args.task == "bon":
        out = run_bon(args.trials_dir, args.out, args.model, args.model_id,
                      args.domain, args.n_trials, args.step_agg)
    else:
        if args.model != "thinkprm":
            raise SystemExit("FA baseline supports only --model thinkprm "
                             "(WildReward is an outcome RM with no per-step signal)")
        thr = args.threshold if args.threshold is not None else 0.5
        out = run_fa(args.data_dir, args.out, args.model_id, tuple(args.splits), thr)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    logger.info(f"wrote {args.out}")


if __name__ == "__main__":
    main()
