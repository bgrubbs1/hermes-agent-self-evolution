"""Small, dependency-free statistics helpers for contest calibration."""

from __future__ import annotations

import math
from typing import Iterable, Sequence

Z_95 = 1.959963984540054


def clip_prob(p: float, eps: float = 1e-6) -> float:
    """Clamp a probability away from 0 and 1 so logs and logits stay finite."""
    return min(1.0 - eps, max(eps, float(p)))


def logit(p: float) -> float:
    p = clip_prob(p)
    return math.log(p / (1.0 - p))


def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def wilson_interval(successes: int, trials: int, z: float = Z_95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion (default 95%)."""
    if trials <= 0:
        raise ValueError("trials must be positive")
    if not 0 <= successes <= trials:
        raise ValueError("successes must be between 0 and trials")
    p = successes / trials
    z2 = z * z
    denom = 1.0 + z2 / trials
    centre = (p + z2 / (2.0 * trials)) / denom
    half = z * math.sqrt(p * (1.0 - p) / trials + z2 / (4.0 * trials * trials)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def weighted_quantile(values: Sequence[float], weights: Sequence[float], q: float) -> float:
    """Quantile of a discrete weighted distribution (inverse CDF, lower value on ties)."""
    if len(values) != len(weights) or not values:
        raise ValueError("values and weights must be non-empty and the same length")
    if not 0.0 <= q <= 1.0:
        raise ValueError("q must be in [0, 1]")
    pairs = sorted(zip(values, weights))
    total = sum(w for _, w in pairs)
    if total <= 0:
        raise ValueError("weights must sum to a positive number")
    target = q * total
    cumulative = 0.0
    for value, weight in pairs:
        cumulative += weight
        if cumulative >= target - 1e-12:
            return value
    return pairs[-1][0]


def brier_score(pairs: Iterable[tuple[float, int]]) -> float:
    pairs = list(pairs)
    if not pairs:
        raise ValueError("no forecasts to score")
    return sum((p - y) ** 2 for p, y in pairs) / len(pairs)


def log_loss(pairs: Iterable[tuple[float, int]]) -> float:
    pairs = list(pairs)
    if not pairs:
        raise ValueError("no forecasts to score")
    total = 0.0
    for p, y in pairs:
        p = clip_prob(p)
        total += -(y * math.log(p) + (1 - y) * math.log(1.0 - p))
    return total / len(pairs)


def logsumexp(xs: Sequence[float]) -> float:
    m = max(xs)
    if m == -math.inf:
        return -math.inf
    return m + math.log(sum(math.exp(x - m) for x in xs))
