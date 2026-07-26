"""One-forward-pass log-prob extraction and reward construction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from pa.aggregations import aggregate_steps, aggregate_tokens


_DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


@dataclass
class StepCache:
    """Per-message log-probs for one (trajectory, model) pair.

    Lists are aligned to ``message_texts``; non-action and truncated
    spans hold empty arrays, and ``is_action`` marks the scorable ones.
    """

    gathered: List[np.ndarray]
    topk_mean: List[np.ndarray]
    is_action: List[bool]


class LogprobScorer:
    """Single-model scorer producing both gathered and top-K mean log-probs."""

    def __init__(
        self,
        model_name: str,
        dtype: str = "bfloat16",
        device_map: str = "cuda:0",
        top_k: int = 20,
        max_length: int = 16384,
    ) -> None:
        self.top_k = int(top_k)
        self.max_length = int(max_length)
        torch_dtype = _DTYPES[dtype]
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=True,
        )
        self.model.eval()

    def free(self) -> None:
        del self.model
        torch.cuda.empty_cache()

    @torch.no_grad()
    def score(
        self, message_texts: Sequence[str], is_action: Sequence[bool], compute_topk: bool = True
    ) -> StepCache:
        spans: List[Tuple[int, int]] = []
        full: List[int] = []
        cur = 0
        bos = self.tokenizer.bos_token_id
        if bos is not None:
            full.append(bos)
            cur = 1
        for txt in message_texts:
            ids = self.tokenizer.encode(txt, add_special_tokens=False)
            spans.append((cur, cur + len(ids)))
            full.extend(ids)
            cur += len(ids)

        if len(full) > self.max_length:
            excess = len(full) - self.max_length
            full = full[excess:]
            spans = [(max(0, s - excess), max(0, e - excess)) for s, e in spans]

        n_msgs = len(message_texts)
        empty = lambda: np.zeros(0, dtype=np.float32)
        if len(full) < 2:
            zeros = [empty() for _ in range(n_msgs)]
            return StepCache(
                gathered=zeros,
                topk_mean=[empty() for _ in range(n_msgs)],
                is_action=list(is_action),
            )

        input_ids = torch.tensor([full], device=self.model.device)
        logits = self.model(input_ids).logits[0].float()
        log_probs = torch.log_softmax(logits, dim=-1)

        targets = torch.tensor(full, device=log_probs.device)
        gathered = log_probs[:-1].gather(1, targets[1:].unsqueeze(-1)).squeeze(-1)
        gathered_np = gathered.cpu().numpy().astype(np.float32)

        # top-K mean is only needed by the certainty baselines; skip the full-vocab
        # topk when a caller (e.g. the merged-reference sweep) only uses `gathered`.
        if compute_topk:
            k = min(self.top_k, log_probs.size(-1))
            topk = log_probs[:-1].topk(k, dim=-1).values.mean(dim=-1)
            topk_np = topk.cpu().numpy().astype(np.float32)
        else:
            topk_np = None

        out_g: List[np.ndarray] = []
        out_t: List[np.ndarray] = []
        for i, (s, e) in enumerate(spans):
            if not is_action[i] or e <= s or e <= 1:
                out_g.append(empty())
                out_t.append(empty())
                continue
            lo = max(s, 1) - 1
            hi = e - 1
            if hi <= lo:
                out_g.append(empty())
                out_t.append(empty())
            else:
                out_g.append(gathered_np[lo:hi])
                out_t.append(topk_np[lo:hi] if topk_np is not None else empty())
        return StepCache(gathered=out_g, topk_mean=out_t, is_action=list(is_action))


def score_trajectory_progress_advantage(
    policy: StepCache,
    reference: Optional[StepCache],
    token_agg: str,
    step_agg: str,
    beta: float = 1.0,
) -> float:
    """Progress advantage aggregated to a trajectory scalar.

    A truncated action span (empty array) still contributes a 0.0 step
    reward rather than being dropped, keeping the step-count divisor stable.
    """
    rewards: List[float] = []
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
    return aggregate_steps(rewards, step_agg)


def score_trajectory_self_certainty(
    policy: StepCache, token_agg: str = "mean", step_agg: str = "mean"
) -> float:
    """Self-Certainty (Kang et al., 2025).

    Empty action spans are skipped — the top-K mean is undefined when a
    span has no tokens.
    """
    rewards: List[float] = []
    for i, is_act in enumerate(policy.is_action):
        if not is_act:
            continue
        c = policy.topk_mean[i]
        if c.size == 0:
            continue
        rewards.append(aggregate_tokens(c, token_agg))
    return aggregate_steps(rewards, step_agg)


def score_trajectory_deepconf(
    policy: StepCache, variant: str = "tail", token_agg: str = "mean"
) -> float:
    if variant == "tail":
        step_agg = "last"
    elif variant in ("b10", "bottom10"):
        step_agg = "bottom10"
    else:
        raise ValueError(f"unknown DeepConf variant: {variant}")
    return score_trajectory_self_certainty(policy, token_agg=token_agg, step_agg=step_agg)


def per_step_progress_advantage(
    policy: StepCache,
    reference: Optional[StepCache],
    token_agg: str,
    beta: float = 1.0,
) -> List[float]:
    """Per-step progress advantage, aligned with ``policy.gathered``.

    Non-action and empty positions contribute 0.0 so callers can index
    by message position directly.
    """
    out: List[float] = []
    for i, p in enumerate(policy.gathered):
        if not policy.is_action[i] or p.size == 0:
            out.append(0.0)
            continue
        p_agg = aggregate_tokens(p, token_agg)
        if reference is not None and i < len(reference.gathered):
            r = reference.gathered[i]
            r_agg = aggregate_tokens(r, token_agg) if r.size > 0 else 0.0
            out.append(beta * (p_agg - r_agg))
        else:
            out.append(beta * p_agg)
    return out
