"""
KFX: Trajectory Counts Under Cyclic Constraints
Exact Enumeration and Structural Classification

Kaifan Xie, March 2026
Verification code for all theorems in the paper.
"""

from math import comb
from itertools import product as iproduct
from collections import defaultdict
import math


# ══════════════════════════════════════════════════════════════════════════════
# Core functions
# ══════════════════════════════════════════════════════════════════════════════

def run_traj(initial, pattern, mod):
    """Generate trajectory from initial state using step pattern."""
    traj = [initial]
    cur = initial
    for s in pattern:
        b, p = cur
        cur = (b, (p + 1) % mod) if s == "R1" else (1 - b, (p - 1) % mod)
        traj.append(cur)
    return traj


def A(m, h):
    """
    Half-count A(m,h): constrained binomial sum.
    Conditions:
      - phase return:  2k - h ≡ 0  (mod m)
      - parity flip:   h - k  ≡ 1  (mod 2)
    """
    total = 0
    for k in range(h + 1):
        if (2 * k - h) % m != 0:
            continue
        if (h - k) % 2 == 0:
            continue
        total += comb(h, k)
    return total


def W(m, h):
    """Exact count: |W(m,h)| = A(m,h)^2."""
    return A(m, h) ** 2


def A_fourier(m, h):
    """
    Fourier representation of A(m,h):
      A(m,h) = (1/m) * sum_{j=0}^{m-1} w^{-jh}
               * [(1+w^{2j})^h - (-1)^h * (1-w^{2j})^h] / 2
    where w = exp(2*pi*i/m).
    """
    import cmath
    omega = cmath.exp(2j * math.pi / m)
    total = 0.0
    for j in range(m):
        w = omega ** j
        poly = ((1 + w ** 2) ** h - (-1) ** h * (1 - w ** 2) ** h) / 2
        total += (w ** (-h)) * poly
    return round(total.real / m)


# ══════════════════════════════════════════════════════════════════════════════
# Brute-force verification
# ══════════════════════════════════════════════════════════════════════════════

def count_legal_brute(half_len, mod):
    """Enumerate admissible step patterns by exhaustive search."""
    states = [(b, p) for b in (0, 1) for p in range(mod)]
    length = half_len * 2
    legal = set()
    for init in states:
        for pat in iproduct(("R1", "R2"), repeat=length):
            traj = run_traj(init, pat, mod)
            s0   = traj[0]
            smid = traj[half_len]
            send = traj[length]
            b0, p0 = s0
            if smid == (1 - b0, p0) and send == s0:
                legal.add(pat)
    return len(legal)


# ══════════════════════════════════════════════════════════════════════════════
# Theorem verifications
# ══════════════════════════════════════════════════════════════════════════════

def verify_formula(max_m=6, max_h=6):
    """Verify |W(m,h)| = A(m,h)^2 against brute force."""
    print("=" * 60)
    print("Formula verification: A(m,h)^2 vs brute force")
    print("=" * 60)
    all_ok = True
    for m in range(1, max_m + 1):
        for h in range(1, max_h + 1):
            formula = W(m, h)
            brute   = count_legal_brute(h, m)
            fourier = A_fourier(m, h) ** 2
            ok = (formula == brute == fourier)
            if not ok:
                all_ok = False
                print(f"  MISMATCH W({m},{h}): "
                      f"formula={formula}, brute={brute}, fourier={fourier}")
    if all_ok:
        print(f"  All values match for m,h in [1,{max_m}] x [1,{max_h}].")
    return all_ok


def verify_theorem2(test_mods=None):
    """Theorem 2: |W(m,2)| = 4 for all m."""
    if test_mods is None:
        test_mods = [1, 2, 3, 5, 7, 11, 13, 17, 97, 101, 1000, 9999]
    print("=" * 60)
    print("Theorem 2: |W(m,2)| = 4 for all m")
    print("=" * 60)
    all_ok = True
    for m in test_mods:
        n  = W(m, 2)
        ok = (n == 4)
        if not ok:
            all_ok = False
        print(f"  m={m:>6}: |W|={n}  {'OK' if ok else 'FAIL ***'}")
    print(f"  All correct: {all_ok}")
    return all_ok


def verify_theorem3():
    """Theorem 3: |W(m,h)| = 1 iff m = h odd."""
    print("=" * 60)
    print("Theorem 3: |W(m,h)| = 1 iff m = h odd")
    print("=" * 60)

    print("  Odd diagonal m = h = odd:")
    for k in [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 51, 99, 101]:
        n = W(k, k)
        mark = "* rigid" if n == 1 else "ERROR"
        print(f"  ({k:3d},{k:3d}): |W|={n}  {mark}")

    print("  Even diagonal m = h = even:")
    for k in [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]:
        n = W(k, k)
        print(f"  ({k:3d},{k:3d}): |W|={n}")

    print("  Off-diagonal: should not be 1 (spot check)")
    for m, h in [(3, 5), (5, 3), (3, 7), (7, 3), (5, 7), (7, 5)]:
        n = W(m, h)
        ok = (n != 1)
        print(f"  ({m},{h}): |W|={n}  {'OK (not 1)' if ok else 'ERROR'}")


# ══════════════════════════════════════════════════════════════════════════════
# Phase diagram
# ══════════════════════════════════════════════════════════════════════════════

def print_phase_diagram(mod_range=range(1, 10), hl_range=range(1, 8)):
    """Print the existence phase diagram."""
    mods = list(mod_range)
    hls  = list(hl_range)
    print("=" * 60)
    print("Existence phase diagram  |W(m,h)|")
    print("  1* = rigid unique,  . = non-existent")
    print("=" * 60)
    header = f"{'h\\m':>5}" + "".join(f"{m:>7}" for m in mods)
    print(header)
    print("-" * (5 + 7 * len(mods)))
    for hl in hls:
        row = f"{hl:>5}"
        for mod in mods:
            n = W(mod, hl)
            if   n == 0: cell = "."
            elif n == 1: cell = "1*"
            else:        cell = str(n)
            row += f"{cell:>7}"
        print(row)

    total = len(mods) * len(hls)
    exist = sum(1 for m in mods for h in hls if W(m, h) > 0)
    rigid = sum(1 for m in mods for h in hls if W(m, h) == 1)
    print(f"\n  Total cells: {total}")
    print(f"  Existent (|W|>0): {exist}  ({exist/total:.1%})")
    print(f"  Rigid unique (|W|=1): {rigid}")


# ══════════════════════════════════════════════════════════════════════════════
# Cost functions: sqrt(2) and pi
# ══════════════════════════════════════════════════════════════════════════════

def cost_sqrt2():
    """sqrt(2): structural gap, constant for all discretization levels."""
    print("=" * 60)
    print("Cost: sqrt(2) -- structural gap (constant)")
    print("=" * 60)
    sqrt2 = math.sqrt(2)
    for n in [1, 2, 4, 16, 64, 256, 1024]:
        cost = 2.0 - sqrt2
        print(f"  n={n:>6}: L1=2.000000, L2={sqrt2:.6f}, "
              f"cost={cost:.10f}  [CONSTANT]")


def cost_pi():
    """pi: closure limit, converges at rate 1/4 per edge-doubling."""
    print("=" * 60)
    print("Cost: pi -- closure limit (converges to zero)")
    print("=" * 60)
    pi   = math.pi
    prev = None
    print(f"  {'n':>6}  {'P_n/(2r)':>14}  {'cost':>16}  {'ratio':>8}")
    for n in [3, 4, 6, 8, 12, 24, 48, 96, 192, 384, 768, 1536]:
        pn    = n * math.sin(pi / n)
        cost  = pi - pn
        ratio = cost / prev if prev is not None else float('nan')
        print(f"  {n:>6}  {pn:>14.8f}  {cost:>16.12f}  {ratio:>8.4f}")
        prev = cost
    print("  Convergence ratio -> 0.25 (each edge-doubling reduces cost 4x)")


# ══════════════════════════════════════════════════════════════════════════════
# Language slice / information loss analysis
# ══════════════════════════════════════════════════════════════════════════════

def language_analysis(mod=5, length=4):
    """
    Information loss analysis for various language slices.
    C_language = H(trajectory | language) in bits.
    """
    states = [(b, p) for b in (0, 1) for p in range(mod)]
    raw = []
    for init in states:
        for pat in iproduct(("R1", "R2"), repeat=length):
            raw.append((init, pat, run_traj(init, pat, mod)))

    slices = {
        "b_only":    lambda t: tuple(b for b, p in t),
        "p_only":    lambda t: tuple(p for b, p in t),
        "p_mod2":    lambda t: tuple(p % 2 for b, p in t),
        "b+p_mod2":  lambda t: tuple((b, p % 2) for b, p in t),
        "b+p_mod3":  lambda t: tuple((b, p % 3) for b, p in t),
        "b+p_full":  lambda t: tuple((b, p) for b, p in t),
        "flip_only": lambda t: tuple(
            "F" if t[i][0] != t[i + 1][0] else "K"
            for i in range(len(t) - 1)),
        "diff_both": lambda t: tuple(
            ((t[i+1][0] - t[i][0]), (t[i+1][1] - t[i][1]) % mod)
            for i in range(len(t) - 1)),
    }

    total = len(raw)
    max_h = math.log2(total)
    print("=" * 60)
    print(f"Language slice analysis  (mod={mod}, length={length})")
    print(f"Total trajectories: {total},  max entropy: {max_h:.3f} bits")
    print("=" * 60)
    print(f"  {'Slice':>14}  {'Buckets':>8}  {'Collisions':>11}  "
          f"{'H(t|lang)':>11}  {'Loss%':>7}  {'Lossless':>9}")
    for name, fn in slices.items():
        buckets = defaultdict(list)
        for _, _, traj in raw:
            buckets[fn(traj)].append(traj)
        n_col  = sum(1 for v in buckets.values() if len(v) > 1)
        cond_h = sum(
            (len(v) / total) * math.log2(len(v))
            for v in buckets.values() if len(v) > 1
        )
        loss = cond_h / max_h
        ll   = "YES" if n_col == 0 else "no"
        print(f"  {name:>14}  {len(buckets):>8}  {n_col:>11}  "
              f"{cond_h:>11.3f}  {loss:>7.1%}  {ll:>9}")


# ══════════════════════════════════════════════════════════════════════════════
# SU(2) interface verification
# ══════════════════════════════════════════════════════════════════════════════

def verify_su2_interface():
    """
    Verify: for odd m, A(m,h) = C(h, k*) = dim(V_{j, m_z}).
    """
    print("=" * 60)
    print("SU(2) counting interface (odd m)")
    print("  A(m,h) = C(h,k*) = dim V_{j, m_z}")
    print("=" * 60)
    print(f"  {'(m,h)':>8}  {'k*':>4}  {'j':>5}  {'m_z':>6}  "
          f"{'C(h,k*)':>8}  {'A(m,h)':>8}  {'match':>6}")
    for m in [3, 5, 7, 9, 11]:
        for h in range(m, 3 * m, 2):
            legal_ks = [k for k in range(h + 1)
                        if (2 * k - h) % m == 0 and (h - k) % 2 == 1]
            a = A(m, h)
            if len(legal_ks) == 1:
                k  = legal_ks[0]
                j  = h / 2
                mz = k - h / 2
                d  = comb(h, k)
                ok = (d == a)
                print(f"  ({m:2d},{h:2d})     {k:>4}  {j:>5.1f}  {mz:>+6.1f}  "
                      f"{d:>8}  {a:>8}  {'OK' if ok else 'FAIL'}")
        print()


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("KFX -- Verification Suite")
    print("=" * 60 + "\n")

    verify_formula(max_m=6, max_h=6)
    print()
    verify_theorem2()
    print()
    verify_theorem3()
    print()
    print_phase_diagram()
    print()
    cost_sqrt2()
    print()
    cost_pi()
    print()
    language_analysis()
    print()
    verify_su2_interface()
