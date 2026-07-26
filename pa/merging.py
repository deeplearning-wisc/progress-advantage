"""Weight-space merged reference policies.

    pi_alpha = pi_ref + alpha * tau,

with tau the task vector pi_theta - pi_ref, optionally sparsified and
sign-elected. Sparsification / sign consensus follow mergekit
(`ties`, `dare_linear`, `dare_ties`); `emr` follows EMR-Merging. alpha=0
recovers the reference, alpha=1 the policy (exactly, for `linear`).
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from transformers import AutoModelForCausalLM

from pa.scoring import _DTYPES, LogprobScorer

logger = logging.getLogger(__name__)

MERGE_METHODS = ("linear", "ties", "dare_linear", "dare_ties", "emr")

_SPEC = {
    "ties": {"sparsify": "magnitude", "consensus": True, "normalize": True, "rescale": False},
    "dare_linear": {"sparsify": "random", "consensus": False, "normalize": False, "rescale": True},
    "dare_ties": {"sparsify": "random", "consensus": True, "normalize": False, "rescale": True},
}


def _align(src: torch.Tensor, shape: Tuple[int, ...]) -> torch.Tensor:
    """Pad/truncate the vocab axis — sibling checkpoints may differ there."""
    if tuple(src.shape) == shape:
        return src
    if src.shape[1:] != shape[1:]:
        raise ValueError(f"incompatible shapes {tuple(src.shape)} vs {shape}")
    cur, tgt = src.shape[0], shape[0]
    if cur > tgt:
        return src[:tgt]
    return torch.cat([src, src[-1:].expand(tgt - cur, *src.shape[1:]).clone()])


def _sparsify(
    tau: torch.Tensor,
    density: float,
    method: Optional[str],
    rescale: bool,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    if method is None or density >= 1.0:
        return tau
    if method == "magnitude":
        flat = tau.abs().flatten()
        k = max(1, int(density * flat.numel()))
        if k >= flat.numel():
            return tau
        # kthvalue rather than topk/argsort: no size cap, and cheap on the
        # hundreds-of-millions-element embedding tensors.
        mask = (tau.abs() >= torch.kthvalue(flat, flat.numel() - k + 1).values).float()
    elif method == "random":
        mask = torch.bernoulli(torch.full_like(tau, density), generator=generator)
    else:
        raise ValueError(f"unknown sparsification: {method}")
    out = tau * mask
    return out / density if rescale else out


def _load_state(name: str, dtype: torch.dtype) -> Dict[str, torch.Tensor]:
    model = AutoModelForCausalLM.from_pretrained(
        name, torch_dtype=dtype, device_map="cpu", trust_remote_code=True
    )
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


class MergedReference(LogprobScorer):
    """Reference policy interpolated toward the policy checkpoint.

    ``set_alpha`` rewrites the resident model in place, so an alpha sweep
    loads each checkpoint only once.

    `ties`/`dare_ties` sign consensus and the `emr` election need at least
    two task vectors; with the default single (policy, reference) pair they
    are identities, so `ties` reduces to trim-then-add, `dare_ties` to
    `dare_linear`, and `emr` to `linear`. Pass ``extra_policies`` (further
    post-trained siblings of the same reference) or ``anchor_model`` (a third
    pretrained checkpoint, which turns the reference itself into a task
    vector and makes it the member EMR reconstructs) for a non-trivial merge.
    """

    def __init__(
        self,
        policy_model: str,
        reference_model: str,
        method: str = "linear",
        density: float = 0.2,
        rescale: Optional[bool] = None,
        anchor_model: Optional[str] = None,
        extra_policies: Sequence[str] = (),
        seed: int = 0,
        dtype: str = "bfloat16",
        device_map: str = "cuda:0",
        top_k: int = 20,
        max_length: int = 16384,
    ) -> None:
        if method not in MERGE_METHODS:
            raise ValueError(f"unknown merge method: {method}")
        super().__init__(reference_model, dtype, device_map, top_k, max_length)
        self.method = method
        self.density = density
        self.seed = seed
        self.rescale = _SPEC.get(method, {}).get("rescale", False) if rescale is None else rescale
        self.alpha: Optional[float] = None

        td = _DTYPES[dtype]
        self._policy = _load_state(policy_model, td)
        self._anchor = _load_state(reference_model, td)
        self._members: List[Dict[str, torch.Tensor]] = [self._policy]
        self._target = 0
        if anchor_model is not None:
            self._members.append(self._anchor)
            self._target = 1
            self._anchor = _load_state(anchor_model, td)
        self._members += [_load_state(m, td) for m in extra_policies]

        self._shapes = {k: tuple(v.shape) for k, v in self.model.state_dict().items()}
        self._names = [
            k for k, v in self._anchor.items()
            if k in self._shapes and v.dtype.is_floating_point
        ]
        if method == "emr":
            self._tau = self._emr_vector()
        elif method == "linear":
            self._tau = {}
        else:
            self._tau = self._gta_vector()

    def _task_vectors(self, name: str) -> List[torch.Tensor]:
        shape = self._shapes[name]
        anchor = _align(self._anchor[name], shape).float()
        return [_align(m[name], shape).float() - anchor for m in self._members]

    def _gta_vector(self) -> Dict[str, torch.Tensor]:
        spec = _SPEC[self.method]
        gen = torch.Generator().manual_seed(self.seed)
        out: Dict[str, torch.Tensor] = {}
        for name in self._names:
            deltas = torch.stack([
                _sparsify(t, self.density, spec["sparsify"], self.rescale, gen)
                for t in self._task_vectors(name)
            ])
            if spec["consensus"]:
                sign = torch.where(deltas.sum(0) >= 0, 1.0, -1.0)
                keep = deltas.sign() == sign
                mixed = (deltas * keep).sum(0)
                divisor = keep.float().sum(0).clamp(min=1.0)
            else:
                mixed = deltas.sum(0)
                divisor = float(len(self._members))
            out[name] = (mixed / divisor if spec["normalize"] else mixed).to(self.model.dtype)
        return out

    def _emr_vector(self) -> Dict[str, torch.Tensor]:
        if len(self._members) < 2:
            logger.warning(
                "emr with a single task vector is degenerate (reduces to linear); "
                "pass extra_policies=... or anchor_model=..."
            )
        out: Dict[str, torch.Tensor] = {}
        num = den = 0.0
        for name in self._names:
            taus = self._task_vectors(name)
            sign = torch.where(sum(taus) > 0, 1.0, -1.0)
            unified = torch.stack([(t * ((t * sign) > 0)).abs() for t in taus]).amax(0) * sign
            effective = unified * ((taus[self._target] * sign) > 0)
            num += float(taus[self._target].abs().mean())
            den += float(effective.abs().mean())
            out[name] = effective
        lam = num / den if den > 0 else 1.0
        self.emr_rescale = lam
        return {k: (v * lam).to(self.model.dtype) for k, v in out.items()}

    @torch.no_grad()
    def set_alpha(self, alpha: float) -> "MergedReference":
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        for name, param in self.model.state_dict().items():
            shape = tuple(param.shape)
            anchor = _align(self._anchor[name], shape)
            if not anchor.dtype.is_floating_point:
                param.copy_(anchor.to(param.device))
                continue
            if self.method == "linear":
                policy = _align(self._policy[name], shape)
                merged = alpha * policy.float() + (1.0 - alpha) * anchor.float()
            elif name in self._tau:
                merged = anchor.float() + alpha * self._tau[name].float()
            else:
                merged = anchor.float()
            param.copy_(merged.to(param.dtype).to(param.device))
        self.alpha = alpha
        return self

    def free(self) -> None:
        self._policy = self._anchor = self._members = self._tau = None
        super().free()
