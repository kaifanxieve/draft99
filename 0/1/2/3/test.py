#!/usr/bin/env python3
"""
KFX v0.6 rigorous scan (reference implementation)

Design constraints aligned with the paper:
- Discrete time steps only
- Modular arithmetic as the state-generating primitive
- No probabilistic assumptions; everything is deterministic & executable
- "Operations" are treated as state update rules that compress path information

Default evolution rule (editable):
    x_{t+1} = (x_t**p + 1) mod m

Outputs:
- Phase boundary by steps (RAW ENTROPY): m*(step) = argmax_m |ΔH|
- Phase boundary by steps (ENTROPY RATIO): m*(step) = argmax_m |Δ(H/log2 m)|
- Phase boundary by m (sampled): step*(m) = argmax_step |ΔH|
- Robust anomaly ranking from m-direction gradients using MAD z-scores

Author: (you)
"""

from __future__ import annotations

import math
import statistics
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


# -----------------------------
# Utilities: number theory
# -----------------------------

def v2(n: int) -> int:
    """2-adic valuation: exponent of 2 in n (v2(0) undefined; here n>=1)."""
    if n <= 0:
        raise ValueError("v2 is defined for positive integers only.")
    c = 0
    while (n & 1) == 0:
        n >>= 1
        c += 1
    return c

def factorize(n: int) -> Dict[int, int]:
    """Trial-division prime factorization for n up to ~1e6 comfortably."""
    if n <= 0:
        raise ValueError("factorize expects n>0")
    x = n
    f: Dict[int, int] = {}
    # factor 2
    while x % 2 == 0:
        f[2] = f.get(2, 0) + 1
        x //= 2
    p = 3
    while p * p <= x:
        while x % p == 0:
            f[p] = f.get(p, 0) + 1
            x //= p
        p += 2
    if x > 1:
        f[x] = f.get(x, 0) + 1
    return f

def format_factors(f: Dict[int, int]) -> str:
    """Pretty factor string like '2^3 * 3 * 11'."""
    parts = []
    for prime in sorted(f.keys()):
        exp = f[prime]
        if exp == 1:
            parts.append(str(prime))
        else:
            parts.append(f"{prime}^{exp}")
    return " * ".join(parts) if parts else "1"


# -----------------------------
# Core: entropy + smoothing
# -----------------------------

def shannon_entropy_from_counts(counts: Sequence[int]) -> float:
    """
    Shannon entropy in bits. Counts may be zeros.
    H = -sum p_i log2 p_i.
    """
    total = sum(counts)
    if total <= 0:
        return 0.0
    h = 0.0
    for c in counts:
        if c:
            p = c / total
            h -= p * math.log(p, 2)
    return h

def median_smooth_1d(xs: Sequence[float], half_window: int) -> List[float]:
    """
    Median smoothing with half-window w: each point replaced by median in [i-w, i+w].
    If w=0 -> identity.
    """
    if half_window <= 0:
        return list(xs)
    n = len(xs)
    out = []
    for i in range(n):
        lo = max(0, i - half_window)
        hi = min(n, i + half_window + 1)
        out.append(statistics.median(xs[lo:hi]))
    return out


# -----------------------------
# Model: modular evolution rule
# -----------------------------

def step_map(x: int, m: int, p: int) -> int:
    """
    The ONLY place you need to change if your scan used a different rule.

    Default:
        x_{t+1} = (x^p + 1) mod m
    """
    return (pow(x, p, m) + 1) % m


def trajectory_states(m: int, p: int, steps: int, init_state: int) -> List[int]:
    """
    Generate states x_0..x_steps (length steps+1), deterministic.
    """
    if m <= 0:
        raise ValueError("m must be positive.")
    x = init_state % m
    seq = [x]
    for _ in range(steps):
        x = step_map(x, m, p)
        seq.append(x)
    return seq


def entropy_by_step_for_m(m: int, p: int, steps: int, init_state: int) -> List[float]:
    """
    For each step s=1..steps, compute entropy of the empirical distribution of states
    observed from time 0..s inclusive under modulus m and power p.
    Returns list length=steps, indexed by step-1.
    """
    seq = trajectory_states(m=m, p=p, steps=steps, init_state=init_state)
    # Incrementally maintain counts over residues 0..m-1
    counts = [0] * m
    hs: List[float] = []
    for s in range(1, steps + 1):
        # include state at time s
        counts[seq[s]] += 1
        # also include initial state at s=1? We want 0..s:
        if s == 1:
            counts[seq[0]] += 1
        hs.append(shannon_entropy_from_counts(counts))
    return hs


# -----------------------------
# Robust anomaly scoring (MAD)
# -----------------------------

def mad(values: Sequence[float], *, centre: Optional[float] = None) -> float:
    """
    Median absolute deviation (MAD).
    """
    if not values:
        return 0.0
    c = statistics.median(values) if centre is None else centre
    devs = [abs(v - c) for v in values]
    return statistics.median(devs)

def robust_zscores(values: Sequence[float]) -> List[float]:
    """
    Robust z-score using MAD:
        z = 0.6745 * (x - median) / MAD
    If MAD=0 -> z=0 for all (MAD-safe).
    """
    if not values:
        return []
    med = statistics.median(values)
    m = mad(values, centre=med)
    if m == 0:
        return [0.0] * len(values)
    return [0.6745 * (v - med) / m for v in values]


# -----------------------------
# Scan configuration + report
# -----------------------------

@dataclass(frozen=True)
class ScanConfig:
    steps: int = 30
    m_min: int = 1
    m_max: int = 1000
    powers: Tuple[int, ...] = (1,2,3,4,5,6,7,8,9,10)
    init_state: int = 0

    exclude_small_m_for_scoring: int = 4
    m_smoothing_median_half_window: int = 0
    dedup_adjacent_m_anomalies: bool = False

    # Sampling for "phase boundary (by m, sampled)"
    sampled_m_points: int = 16


def argmax_abs_delta_over_m(h_by_m: Sequence[float], m_values: Sequence[int], *, m_exclude_lt: int) -> int:
    """
    Given h_by_m aligned with m_values, return m at which |ΔH| is maximised, where
    ΔH is adjacent difference along m.
    Only consider m >= m_exclude_lt.
    """
    best_m = m_values[0]
    best_val = -1.0
    for i in range(1, len(m_values)):
        m = m_values[i]
        if m < m_exclude_lt:
            continue
        d = abs(h_by_m[i] - h_by_m[i-1])
        if d > best_val:
            best_val = d
            best_m = m
    return best_m

def argmax_abs_delta_over_steps(h_by_step: Sequence[float]) -> int:
    """
    Return step* = argmax_step |ΔH| where Δ is adjacent along step (1..steps).
    Returns step index in [2..steps], interpreted as boundary between step-1 -> step.
    """
    best_step = 2
    best_val = -1.0
    for s in range(2, len(h_by_step) + 1):
        d = abs(h_by_step[s-1] - h_by_step[s-2])
        if d > best_val:
            best_val = d
            best_step = s
    return best_step

def build_sampled_m_list(m_min: int, m_max: int, n: int) -> List[int]:
    if n <= 1:
        return [m_min]
    out = []
    for i in range(n):
        t = i / (n - 1)
        m = int(round(m_min + t * (m_max - m_min)))
        out.append(m)
    # Ensure strictly within range and unique, then pad if needed
    out = sorted(set(max(m_min, min(m_max, x)) for x in out))
    if out[0] != m_min:
        out = [m_min] + out
    if out[-1] != m_max:
        out = out + [m_max]
    return out

def dedup_adjacent(points: List[Tuple[int,int,int,float,float,int,float]]) -> List[Tuple[int,int,int,float,float,int,float]]:
    """
    Dedup adjacent m anomalies (same step, consecutive m) keeping highest score.
    Tuple format:
        (rank_placeholder, step, m, score, dH, dS, dPeak)
    """
    if not points:
        return points
    points_sorted = sorted(points, key=lambda r: (r[1], r[2]))
    out = []
    i = 0
    while i < len(points_sorted):
        cur = points_sorted[i]
        j = i + 1
        best = cur
        while j < len(points_sorted) and points_sorted[j][1] == cur[1] and points_sorted[j][2] == points_sorted[j-1][2] + 1:
            if points_sorted[j][3] > best[3]:
                best = points_sorted[j]
            j += 1
        out.append(best)
        i = j
    return out

def scan_power(cfg: ScanConfig, p: int) -> str:
    """
    Scan one power p and return the full report block as a string.
    """
    t0 = time.time()

    # Precompute entropies:
    # H[m][s] where m in [m_min..m_max], s in [1..steps]
    m_values = list(range(cfg.m_min, cfg.m_max + 1))
    H_by_m: List[List[float]] = []
    for m in m_values:
        # For m=1, Z_m is {0}; entropy is 0. Make it well-defined.
        if m == 1:
            H_by_m.append([0.0] * cfg.steps)
            continue
        hs = entropy_by_step_for_m(m=m, p=p, steps=cfg.steps, init_state=cfg.init_state)
        H_by_m.append(hs)

    # Optional smoothing along m for each step (on the entropy values as a function of m)
    if cfg.m_smoothing_median_half_window > 0:
        for s in range(cfg.steps):
            col = [H_by_m[i][s] for i in range(len(m_values))]
            col_s = median_smooth_1d(col, cfg.m_smoothing_median_half_window)
            for i in range(len(m_values)):
                H_by_m[i][s] = col_s[i]

    # Phase boundary by steps: for each step boundary (s-1 -> s), choose m with max |ΔH|
    mstar_raw_by_step: List[int] = []
    mstar_ratio_by_step: List[int] = []

    # For each step boundary, we need ΔH at step s: H_s - H_{s-1}
    # We'll find, over m, which m maximises |ΔH|.
    for s in range(2, cfg.steps + 1):
        # build arrays over m
        dH_over_m = [abs(H_by_m[i][s-1] - H_by_m[i][s-2]) for i in range(len(m_values))]
        # ratio variant: H/log2(m)
        dR_over_m = []
        for i, m in enumerate(m_values):
            if m <= 1:
                dR_over_m.append(0.0)
            else:
                denom = math.log(m, 2)
                r1 = H_by_m[i][s-1] / denom if denom > 0 else 0.0
                r0 = H_by_m[i][s-2] / denom if denom > 0 else 0.0
                dR_over_m.append(abs(r1 - r0))

        # choose argmax with m>=exclude_small_m_for_scoring
        best_raw = None
        best_raw_val = -1.0
        best_rat = None
        best_rat_val = -1.0

        for i, m in enumerate(m_values):
            if m < cfg.exclude_small_m_for_scoring:
                continue
            if dH_over_m[i] > best_raw_val:
                best_raw_val = dH_over_m[i]
                best_raw = m
            if dR_over_m[i] > best_rat_val:
                best_rat_val = dR_over_m[i]
                best_rat = m

        mstar_raw_by_step.append(best_raw if best_raw is not None else cfg.m_min)
        mstar_ratio_by_step.append(best_rat if best_rat is not None else cfg.m_min)

    # Phase boundary by m, sampled: for each sampled m, choose step with max |ΔH_step|
    sampled_ms = build_sampled_m_list(cfg.m_min, cfg.m_max, cfg.sampled_m_points)
    stepstar_raw_by_m: List[Tuple[int,int]] = []
    stepstar_ratio_by_m: List[Tuple[int,int]] = []

    m_to_index = {m: (m - cfg.m_min) for m in m_values}
    for m in sampled_ms:
        i = m_to_index[m]
        h_steps = H_by_m[i]  # length steps
        # raw
        s_raw = argmax_abs_delta_over_steps(h_steps)
        # ratio
        if m <= 1:
            s_rat = 2
        else:
            denom = math.log(m, 2)
            r_steps = [h / denom if denom > 0 else 0.0 for h in h_steps]
            s_rat = argmax_abs_delta_over_steps(r_steps)
        stepstar_raw_by_m.append((m, s_raw))
        stepstar_ratio_by_m.append((m, s_rat))

    # Robust anomalies: use m-direction gradients at each step boundary, then MAD z-score
    # We'll score each (step_boundary s, m) by robust z of dH(m) over m.
    anomalies: List[Tuple[int,int,int,float,float,int,float]] = []
    # tuple: (rank_placeholder, step, m, score, dH, dS, dPeak)

    for s in range(2, cfg.steps + 1):
        dH = [abs(H_by_m[i][s-1] - H_by_m[i][s-2]) for i in range(len(m_values))]
        # restrict for scoring
        idx0 = max(0, cfg.exclude_small_m_for_scoring - cfg.m_min)
        dH_scored = dH[idx0:] if idx0 < len(dH) else []
        z = robust_zscores(dH_scored)

        # simple structural side-metrics:
        # dS: number of strict local maxima in dH within a small window around m (proxy for "spikiness")
        # dPeak: distance to nearest local maximum (proxy)
        # These are not canonical; they are deterministic diagnostics.
        local_maxima = set()
        for i in range(1, len(dH)-1):
            if dH[i] > dH[i-1] and dH[i] > dH[i+1]:
                local_maxima.add(m_values[i])

        for j, zj in enumerate(z):
            i = idx0 + j
            m = m_values[i]
            # score only positive spikes
            score = max(0.0, zj)
            if score <= 0:
                continue

            # dS: count of local maxima in [m-5, m+5]
            lo = m - 5
            hi = m + 5
            dS = sum(1 for mm in local_maxima if lo <= mm <= hi)

            # dPeak: distance to nearest local maximum (or 0.5 if none)
            if local_maxima:
                nearest = min(abs(m - mm) for mm in local_maxima)
                dPeak = float(nearest) + 0.5
            else:
                dPeak = 0.5

            anomalies.append((0, s-1, m, score, dH[i], dS, dPeak))

    # Optional dedup adjacent m anomalies
    if cfg.dedup_adjacent_m_anomalies:
        anomalies = dedup_adjacent(anomalies)

    # Rank anomalies by score desc, then by step asc, then by m asc
    anomalies.sort(key=lambda r: (-r[3], r[1], r[2]))
    top = anomalies[:30]

    t1 = time.time()
    scan_time = t1 - t0

    # Build report
    lines: List[str] = []
    lines.append(f"[POWER p = {p}]  scan_time={scan_time:.2f}s")
    lines.append("-" * 96)

    # Print phase boundary by steps
    lines.append("Phase boundary (by steps, RAW ENTROPY): m*(step)=argmax_m |ΔH|  "
                 f"[m>=exclude_small_m={cfg.exclude_small_m_for_scoring}]")
    # steps are boundaries 1->2 .. (steps-1)->steps, we label "k-> m*"
    # To mirror your layout "1-> 4 2->4 ...", we index by step number (boundary at step number).
    row = []
    for k, ms in enumerate(mstar_raw_by_step, start=1):
        row.append(f"{k:>4}-> {ms:>4}")
        if len(row) == 10:
            lines.append("  " + "  ".join(row))
            row = []
    if row:
        lines.append("  " + "  ".join(row))

    lines.append("")
    lines.append("Phase boundary (by steps, ENTROPY RATIO): m*(step)=argmax_m |Δ(H/log2 m)|  "
                 f"[m>=exclude_small_m={cfg.exclude_small_m_for_scoring}]")
    row = []
    for k, ms in enumerate(mstar_ratio_by_step, start=1):
        row.append(f"{k:>4}-> {ms:>4}")
        if len(row) == 10:
            lines.append("  " + "  ".join(row))
            row = []
    if row:
        lines.append("  " + "  ".join(row))

    lines.append("")
    lines.append("Phase boundary (by m, sampled) using RAW ENTROPY: step*(m)=argmax_step |ΔH|")
    lines.append("  " + "  ".join(f"{m}->{s}" for (m, s) in stepstar_raw_by_m))
    lines.append("")
    lines.append("Phase boundary (by m, sampled) using ENTROPY RATIO: step*(m)=argmax_step |Δ(H/log2 m)|")
    lines.append("  " + "  ".join(f"{m}->{s}" for (m, s) in stepstar_ratio_by_m))

    lines.append("")
    lines.append("Top anomaly points (robust m-direction gradients; MAD-safe):")
    lines.append("  anomaly gradient basis: entropy (with optional median smoothing)")
    lines.append("rank | step |     m |  v2 | factors                 |   score |     dH |  dS |   dPeak")
    lines.append("-" * 96)

    for rnk, item in enumerate(top, start=1):
        _, step, m, score, dHval, dSval, dPeak = item
        fac = factorize(m) if m >= 2 else {}
        lines.append(f"{rnk:>4} | {step:>4} | {m:>5} | {v2(m) if m>=1 else 0:>3} | "
                     f"{format_factors(fac):<22} | {score:>7.3f} | {dHval:>6.3f} | {dSval:>3} | {dPeak:>8.3f}")

    return "\n".join(lines)


def run_scan(cfg: ScanConfig) -> None:
    header = "=" * 96
    print(header)
    print("KFX v0.6 rigorous scan")
    print(f"steps: 1–{cfg.steps}, m: {cfg.m_min}–{cfg.m_max}, powers: {list(cfg.powers)}, init_state={cfg.init_state}")
    print(f"exclude_small_m for gradient/anomaly scoring: m < {cfg.exclude_small_m_for_scoring} ignored")
    print(f"m-smoothing (median half-window): {cfg.m_smoothing_median_half_window}")
    print(f"dedup_adjacent_m anomalies: {cfg.dedup_adjacent_m_anomalies}")
    print(header)
    print()

    total_t0 = time.time()
    for p in cfg.powers:
        block = scan_power(cfg, p)
        print("=" * 96)
        print(block)
        print()

    total_t1 = time.time()
    print("=" * 96)
    print(f"Done. total_elapsed={total_t1 - total_t0:.2f}s")
    print("=" * 96)


if __name__ == "__main__":
    # Default config mirrors your log header.
    cfg = ScanConfig(
        steps=30,
        m_min=1,
        m_max=1000,
        powers=(1,2,3,4,5,6,7,8,9,10),
        init_state=0,
        exclude_small_m_for_scoring=4,
        m_smoothing_median_half_window=0,
        dedup_adjacent_m_anomalies=False,
        sampled_m_points=16,
    )
    run_scan(cfg)