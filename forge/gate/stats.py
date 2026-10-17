"""Gate statistics: Student t quantiles (no scipy) and the paired log-gain test."""
from __future__ import annotations

import math


def _betacf(a: float, b: float, x: float) -> float:
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 300):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        de = d * c
        h *= de
        if abs(de - 1.0) < 1e-14:
            break
    return h


def betainc(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta I_x(a, b)."""
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    lbt = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log(1 - x)
    if x < (a + 1) / (a + b + 2):
        return math.exp(lbt) * _betacf(a, b, x) / a
    return 1.0 - math.exp(lbt) * _betacf(b, a, 1 - x) / b


def t_cdf(t: float, df: float) -> float:
    x = df / (df + t * t)
    tail = 0.5 * betainc(df / 2, 0.5, x)
    return 1 - tail if t > 0 else tail


def t_ppf(p: float, df: float) -> float:
    """Quantile of Student's t (bisection on the CDF; accurate to ~1e-10)."""
    if not 0 < p < 1:
        raise ValueError("p must be in (0, 1)")
    if p < 0.5:
        return -t_ppf(1 - p, df)
    lo, hi = 0.0, 1.0
    while t_cdf(hi, df) < p:
        hi *= 2
        if hi > 1e8:
            return hi
    for _ in range(200):
        mid = (lo + hi) / 2
        if t_cdf(mid, df) < p:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-11:
            break
    return (lo + hi) / 2


def null_scale(null_gains: list[float]) -> tuple[float, int]:
    """Per-pair noise scale from reference-vs-reference log ratios. Null pairs have mean 0 by
    construction (both sides are the reference), so the scale is the RMS with df = N."""
    n = len(null_gains)
    if n == 0:
        return float("nan"), 0
    return math.sqrt(sum(g * g for g in null_gains) / n), n


def paired_test(gains: list[float], null_gains: list[float], alpha: float) -> dict:
    """Win iff every pair is positive and mean(gain) > t_{1-alpha, df_null} * s_null / sqrt(n)."""
    n = len(gains)
    s, df = null_scale(null_gains)
    mean = sum(gains) / n if n else float("nan")
    if n == 0 or df < 2 or not math.isfinite(s):
        return {"n": n, "mean_log_gain": mean, "s_null": s, "df_null": df, "alpha": alpha,
                "threshold": None, "all_positive": n > 0 and all(g > 0 for g in gains), "win": False,
                "why": "not enough reference pairs"}
    thr = t_ppf(1 - alpha, df) * s / math.sqrt(n)
    allpos = all(g > 0 for g in gains)
    win = allpos and mean > thr
    why = "win" if win else ("a pair was not positive" if not allpos else "mean gain below threshold")
    return {"n": n, "mean_log_gain": mean, "s_null": s, "df_null": df, "alpha": alpha, "threshold": thr,
            "all_positive": allpos, "win": win, "why": why,
            "gain_pct": (math.exp(mean) - 1) * 100, "threshold_pct": (math.exp(thr) - 1) * 100}
