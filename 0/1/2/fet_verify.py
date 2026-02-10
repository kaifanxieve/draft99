#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fragmented Evolution Theory (FET) - multi-parameter verifier / model checker
Purpose: produce reproducible evidence across many finite models to show
         the framework is non-trivial and you've examined edge cases.

No external deps. Python 3.10+ recommended.
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Any
import json
import math
import random
import argparse

# -----------------------------
# Core finite structures
# -----------------------------

@dataclass(frozen=True)
class ModelConfig:
    n: int                      # modulus for Z_n
    T: int                      # time horizon length (discrete times 0..T-1)
    step: int                   # base step for translation dynamics x_{t+1}=x_t+step (mod n)
    x0: int                     # initial state
    delta_kind: str             # "moddiff" or "discrete"
    eps: int                    # threshold for decidability (>=0)
    coarse_k: int               # coarse-grain buckets (1..n). k=n means identity, k=1 means total collapse.
    F_family: str               # "affine_bij" or "affine_all" or "custom"
    allow_bad_F: bool           # include constant maps as adversarial (to show collapse if allowed)

@dataclass
class CheckResult:
    ok: bool
    name: str
    detail: str
    counterexample: Optional[Dict[str, Any]] = None

@dataclass
class Report:
    config: ModelConfig
    predicates: Dict[str, Any]
    checks: List[CheckResult]
    notes: List[str]

# -----------------------------
# Utilities
# -----------------------------

def zn(n: int) -> List[int]:
    return list(range(n))

def mod(x: int, n: int) -> int:
    return x % n

def gcd(a: int, b: int) -> int:
    return math.gcd(a, b)

# -----------------------------
# Trajectory generators
# -----------------------------

def traj_translation(n: int, T: int, x0: int, step: int) -> List[int]:
    x = [0] * T
    x[0] = mod(x0, n)
    for t in range(T - 1):
        x[t + 1] = mod(x[t] + step, n)
    return x

# Optional: affine dynamics x_{t+1} = a*x_t + b (mod n)
def traj_affine(n: int, T: int, x0: int, a: int, b: int) -> List[int]:
    x = [0] * T
    x[0] = mod(x0, n)
    for t in range(T - 1):
        x[t + 1] = mod(a * x[t] + b, n)
    return x

# -----------------------------
# Difference + scale operators
# -----------------------------

def delta_moddiff(n: int, x: int, y: int) -> int:
    """Δ(x,y) = (y-x) mod n, returns 0..n-1"""
    return (y - x) % n

def delta_discrete(_: int, x: int, y: int) -> int:
    """Δ(x,y) = 0 if x=y else 1"""
    return 0 if x == y else 1

def coarse_map(n: int, k: int, x: int) -> int:
    """
    Coarse-grain x in Z_n into k buckets (k|n not required).
    k=n -> identity; k=1 -> everything maps to 0.
    """
    if k <= 1:
        return 0
    if k >= n:
        return x
    # bucket index 0..k-1
    return (x * k) // n

def delta_with_scale(n: int,
                     x: int,
                     y: int,
                     delta_fn: Callable[[int, int, int], int],
                     k: int) -> int:
    """
    Scale acts on states via coarse-graining first (operator view),
    then apply base delta on coarse states (embedded into integers).
    """
    cx = coarse_map(n, k, x)
    cy = coarse_map(n, k, y)
    # we treat coarse states as integers; for moddiff, modulus becomes k (natural for buckets)
    if delta_fn is delta_moddiff:
        return delta_moddiff(k if k >= 1 else 1, cx, cy)
    return delta_fn(k if k >= 1 else 1, cx, cy)

def decidable(d: int, eps: int) -> bool:
    """Decidable iff d > eps"""
    return d > eps

# -----------------------------
# Internal transformations F
# -----------------------------

def affine_maps(n: int, bijective_only: bool) -> List[Callable[[int], int]]:
    Fs = []
    for a in range(n):
        for b in range(n):
            if bijective_only and gcd(a, n) != 1:
                continue
            def makeF(aa: int, bb: int):
                return lambda x, aa=aa, bb=bb: (aa * x + bb) % n
            Fs.append(makeF(a, b))
    return Fs

def constant_maps(n: int) -> List[Callable[[int], int]]:
    return [lambda x, c=c: c for c in range(n)]

# -----------------------------
# Predicates in this formalization
# -----------------------------

def pred_dead_loop(traj: List[int], n: int, delta_fn, eps: int, k: int) -> bool:
    """
    DeadLoop at time t* means for all future deltas (here next-step only)
    Δ(x_t*, x_{t*+1}) <= eps and remains so; in discrete horizon, we use next-step.
    """
    # A practical finite version: dead loop if all remaining adjacent deltas are indecidable
    for t in range(len(traj) - 1):
        d = delta_with_scale(n, traj[t], traj[t + 1], delta_fn, k)
        if decidable(d, eps):
            return False
    return True

def pred_evolving(traj: List[int], n: int, delta_fn, eps: int, k: int,
                  min_len: int = 2) -> bool:
    """
    Evolving if there exists an interval of length >= min_len where all adjacent deltas are decidable.
    """
    T = len(traj)
    for start in range(T - 1):
        ok = True
        length = 1
        for t in range(start, T - 1):
            d = delta_with_scale(n, traj[t], traj[t + 1], delta_fn, k)
            if not decidable(d, eps):
                ok = False
                break
            length += 1
            if ok and length >= min_len:
                return True
    return False

def pred_fragmented(traj: List[int], n: int, delta_fn, eps: int, k: int) -> Tuple[bool, int]:
    """
    Fragmented if exists c>eps such that for all t: Δ >= c.
    In finite horizon, c can be chosen as min Δ over adjacent steps.
    Returns (is_fragmented, c_candidate).
    """
    ds = []
    for t in range(len(traj) - 1):
        ds.append(delta_with_scale(n, traj[t], traj[t + 1], delta_fn, k))
    c = min(ds) if ds else 0
    return (c > eps), c

def pred_driven(traj: List[int], n: int, delta_fn, eps: int, k: int,
                Fs: List[Callable[[int], int]]) -> Tuple[bool, Optional[Dict[str, Any]]]:
    """
    Driven (operational): for all allowed internal transforms F, there exists a time t
    such that Δ(F(x_t), F(x_{t+1})) is decidable.
    Stronger variant (used earlier): for all F and all t, Δ != 0.
    Here we use "exists t" version (weaker, easier to satisfy; closer to your frozen text).
    """
    for idx, F in enumerate(Fs):
        found = False
        for t in range(len(traj) - 1):
            d = delta_with_scale(n, F(traj[t]), F(traj[t + 1]), delta_fn, k)
            if decidable(d, eps):
                found = True
                break
        if not found:
            return False, {"F_index": idx, "t": t, "x_t": traj[t], "x_t1": traj[t+1], "d": d}
    return True, None

# -----------------------------
# Checks (derived claims)
# -----------------------------

def check_deadloop_absorbing(traj, n, delta_fn, eps, k) -> CheckResult:
    # If dead loop holds, evolving must be false.
    DL = pred_dead_loop(traj, n, delta_fn, eps, k)
    EV = pred_evolving(traj, n, delta_fn, eps, k)
    ok = (not DL) or (not EV)
    cx = None
    if not ok:
        cx = {"dead_loop": DL, "evolving": EV}
    return CheckResult(ok, "DeadLoop -> not Evolving", "死循环为吸收态：死循环成立则不存在演化区间。", cx)

def check_fragmented_excludes_deadloop(traj, n, delta_fn, eps, k) -> CheckResult:
    FR, c = pred_fragmented(traj, n, delta_fn, eps, k)
    DL = pred_dead_loop(traj, n, delta_fn, eps, k)
    ok = (not FR) or (not DL)
    cx = None
    if not ok:
        cx = {"fragmented": FR, "c": c, "dead_loop": DL}
    return CheckResult(ok, "Fragmented -> not DeadLoop", "残破排除死循环。", cx)

def check_scale_induced_zero(n, delta_fn, eps, k1, k2) -> CheckResult:
    # Find x,y such that Δ_{k1}=0 but Δ_{k2}>eps (or vice versa)
    # Here scale is represented by coarse_k.
    for x in range(n):
        for y in range(n):
            d1 = delta_with_scale(n, x, y, delta_fn, k1)
            d2 = delta_with_scale(n, x, y, delta_fn, k2)
            if (not decidable(d1, eps)) and decidable(d2, eps):
                return CheckResult(True, "Scale-induced zero exists",
                                   f"存在 (x,y) 使尺度 k={k1} 下不可判定，但尺度 k={k2} 下可判定。",
                                   {"x": x, "y": y, "d_k1": d1, "d_k2": d2})
    return CheckResult(False, "Scale-induced zero exists",
                       f"未找到尺度 k={k1} 与 k={k2} 的显影/塌缩例子（可能参数使两尺度等价）。",
                       None)

def check_scale_relative_evolution(configs: List[Tuple[int,int,int,int,int,str,int]]) -> CheckResult:
    """
    Given list of (n,T,step,x0,eps,delta_kind, k) configs, try find same dynamics but different k
    that flips evolving truth.
    """
    # Group by (n,T,step,x0,eps,delta_kind) and compare across k
    groups: Dict[Tuple[int,int,int,int,int,str], List[Tuple[int,bool]]] = {}
    for n,T,step,x0,eps,dk,k in configs:
        traj = traj_translation(n,T,x0,step)
        delta_fn = delta_moddiff if dk=="moddiff" else delta_discrete
        ev = pred_evolving(traj, n, delta_fn, eps, k)
        key = (n,T,step,x0,eps,dk)
        groups.setdefault(key, []).append((k, ev))
    for key, items in groups.items():
        # look for k1 with True and k2 with False
        for k1,ev1 in items:
            for k2,ev2 in items:
                if ev1 and (not ev2) and k1!=k2:
                    return CheckResult(True, "Scale-relative evolution witness",
                                       "同一系统在不同尺度下演化判定可翻转（尺度相对性）。",
                                       {"base": {"n":key[0],"T":key[1],"step":key[2],"x0":key[3],"eps":key[4],"delta_kind":key[5]},
                                        "k_true": k1, "k_false": k2})
    return CheckResult(False, "Scale-relative evolution witness",
                       "在给定参数扫描内未找到演化判定翻转的实例。扩大尺度/阈值范围可增强检出率。",
                       None)

# -----------------------------
# Runner
# -----------------------------

def build_model(cfg: ModelConfig) -> Tuple[List[int], Callable, List[Callable[[int], int]]]:
    if cfg.n <= 1:
        raise ValueError("n must be >= 2")
    if cfg.T < 2:
        raise ValueError("T must be >= 2")
    if cfg.delta_kind not in ("moddiff", "discrete"):
        raise ValueError("delta_kind must be 'moddiff' or 'discrete'")
    if cfg.coarse_k < 1 or cfg.coarse_k > cfg.n:
        raise ValueError("coarse_k must be in [1,n]")

    traj = traj_translation(cfg.n, cfg.T, cfg.x0, cfg.step)
    delta_fn = delta_moddiff if cfg.delta_kind == "moddiff" else delta_discrete

    if cfg.F_family == "affine_bij":
        Fs = affine_maps(cfg.n, bijective_only=True)
    elif cfg.F_family == "affine_all":
        Fs = affine_maps(cfg.n, bijective_only=False)
    else:
        Fs = affine_maps(cfg.n, bijective_only=True)

    if cfg.allow_bad_F:
        Fs = Fs + constant_maps(cfg.n)

    return traj, delta_fn, Fs

def run_one(cfg: ModelConfig) -> Report:
    traj, delta_fn, Fs = build_model(cfg)

    FR, c = pred_fragmented(traj, cfg.n, delta_fn, cfg.eps, cfg.coarse_k)
    EV = pred_evolving(traj, cfg.n, delta_fn, cfg.eps, cfg.coarse_k)
    DL = pred_dead_loop(traj, cfg.n, delta_fn, cfg.eps, cfg.coarse_k)
    DR, dr_cx = pred_driven(traj, cfg.n, delta_fn, cfg.eps, cfg.coarse_k, Fs)

    preds = {
        "fragmented": FR,
        "fragmented_c": c,
        "evolving": EV,
        "dead_loop": DL,
        "driven": DR,
        "driven_counterexample": dr_cx,
        "trajectory": traj,
    }

    checks: List[CheckResult] = []
    checks.append(check_deadloop_absorbing(traj, cfg.n, delta_fn, cfg.eps, cfg.coarse_k))
    checks.append(check_fragmented_excludes_deadloop(traj, cfg.n, delta_fn, cfg.eps, cfg.coarse_k))

    notes = []
    if cfg.allow_bad_F:
        notes.append("注意：allow_bad_F=True 会引入常值映射，通常会导致 Driven 失败（用于展示 F 过强会塌缩）。")
    if cfg.coarse_k == 1:
        notes.append("注意：coarse_k=1 表示全塌缩尺度，几乎必然导致不可判定与不演化。")
    if cfg.eps > 0:
        notes.append("注意：eps>0 引入阈值判定门（Δ>eps 才可判定），会更严格。")

    return Report(cfg, preds, checks, notes)

def run_sweep(seed: int = 0) -> Dict[str, Any]:
    random.seed(seed)
    all_reports: List[Report] = []

    # Parameter sweep ranges (tweak as needed)
    ns = [5, 7, 8, 9, 10, 12]
    steps = [1, 2, 3]
    coarse_ks = []  # populate per n
    eps_list = [0, 0, 1]  # bias towards eps=0 but include eps=1
    delta_kinds = ["moddiff", "discrete"]
    F_families = ["affine_bij"]
    allow_bad = [False, True]  # show the collapse case too

    # Build configs
    for n in ns:
        ks = sorted(set([1, 2, max(2, n//2), n]))
        for step in steps:
            for k in ks:
                for eps in eps_list:
                    for dk in delta_kinds:
                        for fam in F_families:
                            for bad in allow_bad:
                                cfg = ModelConfig(
                                    n=n, T=n, step=step, x0=0,
                                    delta_kind=dk, eps=eps, coarse_k=k,
                                    F_family=fam, allow_bad_F=bad
                                )
                                all_reports.append(run_one(cfg))

    # Global checks for scale-induced zero and scale-relative evolution (witness search)
    # We'll search within each (n, delta_kind, eps) across k pairs.
    witness_checks: List[CheckResult] = []
    for n in ns:
        for dk in delta_kinds:
            for eps in set(eps_list):
                # choose two scales
                k1 = 1
                k2 = n
                delta_fn = delta_moddiff if dk=="moddiff" else delta_discrete
                witness_checks.append(check_scale_induced_zero(n, delta_fn, eps, k1, k2))

    # Scale-relative evolution witness search
    cfg_records = []
    for rep in all_reports:
        c = rep.config
        cfg_records.append((c.n,c.T,c.step,c.x0,c.eps,c.delta_kind,c.coarse_k))
    witness_checks.append(check_scale_relative_evolution(cfg_records))

    # Summarize pass/fail statistics
    stats = {
        "total_models": len(all_reports),
        "fragmented_true": sum(1 for r in all_reports if r.predicates["fragmented"]),
        "evolving_true": sum(1 for r in all_reports if r.predicates["evolving"]),
        "dead_loop_true": sum(1 for r in all_reports if r.predicates["dead_loop"]),
        "driven_true": sum(1 for r in all_reports if r.predicates["driven"]),
        "checks_all_pass": sum(1 for r in all_reports if all(ch.ok for ch in r.checks)),
        "witness_checks": [asdict(w) for w in witness_checks],
    }

    # Pick a few representative reports to print (one good, one collapsed)
    def pick(predicate_name: str, value: bool, allow_bad_F: Optional[bool]=None) -> Optional[Report]:
        for r in all_reports:
            if r.predicates.get(predicate_name) == value:
                if allow_bad_F is None or r.config.allow_bad_F == allow_bad_F:
                    return r
        return None

    exemplar_good = pick("driven", True, allow_bad_F=False)
    exemplar_collapse = pick("driven", False, allow_bad_F=True)

    out = {
        "stats": stats,
        "exemplar_good": asdict(exemplar_good) if exemplar_good else None,
        "exemplar_collapse": asdict(exemplar_collapse) if exemplar_collapse else None,
    }
    return out

# -----------------------------
# CLI
# -----------------------------

def main():
    ap = argparse.ArgumentParser(description="FET multi-parameter verifier / model checker")
    ap.add_argument("--mode", choices=["one","sweep"], default="sweep")
    ap.add_argument("--seed", type=int, default=0)

    # one-mode params
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--T", type=int, default=5)
    ap.add_argument("--step", type=int, default=1)
    ap.add_argument("--x0", type=int, default=0)
    ap.add_argument("--delta", choices=["moddiff","discrete"], default="moddiff")
    ap.add_argument("--eps", type=int, default=0)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--F", choices=["affine_bij","affine_all"], default="affine_bij")
    ap.add_argument("--allow-bad-F", action="store_true")

    ap.add_argument("--json", action="store_true", help="output JSON only")
    args = ap.parse_args()

    if args.mode == "one":
        cfg = ModelConfig(
            n=args.n, T=args.T, step=args.step, x0=args.x0,
            delta_kind=args.delta, eps=args.eps, coarse_k=args.k,
            F_family=args.F, allow_bad_F=args.allow_bad_F
        )
        rep = run_one(cfg)
        if args.json:
            print(json.dumps(asdict(rep), ensure_ascii=False, indent=2))
            return
        print("=== FET Model Check (one) ===")
        print(json.dumps(asdict(rep), ensure_ascii=False, indent=2))
        return

    # sweep mode
    out = run_sweep(seed=args.seed)
    if args.json:
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return

    print("=== FET Model Check (sweep) ===")
    print("Stats:")
    for k,v in out["stats"].items():
        if k != "witness_checks":
            print(f"  {k}: {v}")
    print("\nWitness checks:")
    for w in out["stats"]["witness_checks"]:
        ok = w["ok"]
        name = w["name"]
        print(f"  - {name}: {'OK' if ok else 'FAIL'}")
        if w.get("counterexample"):
            print(f"    witness: {w['counterexample']}")

    print("\nExemplar (good, allow_bad_F=False):")
    if out["exemplar_good"]:
        eg = out["exemplar_good"]
        print(f"  config={eg['config']}")
        print(f"  predicates={{fragmented:{eg['predicates']['fragmented']}, evolving:{eg['predicates']['evolving']}, driven:{eg['predicates']['driven']}}}")

    print("\nExemplar (collapse, allow_bad_F=True):")
    if out["exemplar_collapse"]:
        ec = out["exemplar_collapse"]
        print(f"  config={ec['config']}")
        print(f"  predicates={{fragmented:{ec['predicates']['fragmented']}, evolving:{ec['predicates']['evolving']}, driven:{ec['predicates']['driven']}}}")
        if ec['predicates'].get('driven_counterexample'):
            print(f"  driven_counterexample={ec['predicates']['driven_counterexample']}")

if __name__ == "__main__":
    main()