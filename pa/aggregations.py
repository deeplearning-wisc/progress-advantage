"""Token-wise and step-wise aggregations for progress advantage."""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np


def aggregate_tokens(values: np.ndarray, strategy: str) -> float:
    if values.size == 0:
        return 0.0
    if strategy == "mean":
        return float(values.mean())
    if strategy == "sum":
        return float(values.sum())
    if strategy == "last":
        return float(values[-1])
    if strategy == "min":
        return float(values.min())
    if strategy == "max":
        return float(values.max())
    raise ValueError(f"unknown token aggregation: {strategy}")


def aggregate_steps(rewards: Sequence[float], strategy: str) -> float:
    if not rewards:
        return 0.0
    if strategy == "mean":
        return float(sum(rewards) / len(rewards))
    if strategy == "sum":
        return float(sum(rewards))
    if strategy == "last":
        return float(rewards[-1])
    if strategy == "max":
        return float(max(rewards))
    if strategy == "min":
        return float(min(rewards))
    if strategy == "bottom10":
        k = max(1, int(math.ceil(0.10 * len(rewards))))
        return float(np.mean(sorted(rewards)[:k]))
    raise ValueError(f"unknown step aggregation: {strategy}")
