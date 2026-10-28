"""Random distribution utilities used by synthetic workloads."""

from __future__ import annotations

import random
from typing import Sequence, TypeVar

T = TypeVar("T")


def weighted_choice(items: Sequence[T], weights: Sequence[float], rng: random.Random) -> T:
    if len(items) != len(weights):
        raise ValueError("items and weights must have the same length")
    if not items:
        raise ValueError("items must not be empty")
    return rng.choices(list(items), weights=list(weights), k=1)[0]


def zipf_weights(count: int, alpha: float) -> list[float]:
    if count <= 0:
        raise ValueError("count must be positive")
    return [1.0 / ((rank + 1) ** alpha) for rank in range(count)]

