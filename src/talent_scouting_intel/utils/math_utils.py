from __future__ import annotations

import math
from typing import Iterable


def safe_log_delta(curr: float, prev: float) -> float:
    return math.log(curr + 1.0) - math.log(prev + 1.0)


def growth_series(series: list[float]) -> list[float]:
    if len(series) < 2:
        return []
    return [safe_log_delta(series[idx], series[idx - 1]) for idx in range(1, len(series))]


def acceleration_series(series: list[float]) -> list[float]:
    growth = growth_series(series)
    if len(growth) < 2:
        return []
    return [growth[idx] - growth[idx - 1] for idx in range(1, len(growth))]


def mean(values: Iterable[float]) -> float:
    vals = list(values)
    if not vals:
        return 0.0
    return sum(vals) / len(vals)


def stddev(values: Iterable[float]) -> float:
    vals = list(values)
    if len(vals) < 2:
        return 0.0
    mu = mean(vals)
    variance = sum((v - mu) ** 2 for v in vals) / (len(vals) - 1)
    return variance ** 0.5


def zscore(value: float, population: list[float]) -> float:
    sigma = stddev(population)
    if sigma == 0:
        return 0.0
    return (value - mean(population)) / sigma


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def minmax_scale(value: float, lo: float, hi: float) -> float:
    if hi <= lo:
        return 0.0
    return clamp01((value - lo) / (hi - lo))


def consecutive_positive_tail(values: list[float]) -> int:
    count = 0
    for value in reversed(values):
        if value > 0:
            count += 1
        else:
            break
    return count


def corr(a: list[float], b: list[float]) -> float:
    if len(a) != len(b) or len(a) < 3:
        return 0.0
    mu_a = mean(a)
    mu_b = mean(b)
    num = sum((x - mu_a) * (y - mu_b) for x, y in zip(a, b, strict=True))
    den_a = sum((x - mu_a) ** 2 for x in a) ** 0.5
    den_b = sum((y - mu_b) ** 2 for y in b) ** 0.5
    denom = den_a * den_b
    if denom == 0:
        return 0.0
    return num / denom
