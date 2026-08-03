#!/usr/bin/env python3
"""
============================================================================
 Chapter 5 — Stage 5: age as a positive control.  DECLARED POST HOC
============================================================================

 This is NOT the pre-registered test. It was added after the null result of
 stage 4, and it is reported as a diagnostic, in the same spirit as the volume
 comparison of PREREGISTRO_CAP5.md section 5b.

 WHY IT IS WORTH RUNNING
 -----------------------
 A null tells you nothing unless the machinery can detect a signal that is
 known to be present. Age correlates with the inradius at r = 0.750 in this
 cohort, so an age split is a signal we know is there. If the same statistic
 rejects on age and not on diagnosis, the null for diagnosis is informative:
 the pipeline works and the effect is absent. If it fails on age too, the null
 says nothing at all.

 THE SPLIT
 ---------
 Median split on age, which happens to give 8 and 8, and which is balanced by
 diagnosis: 4 Control and 4 PD on each side. So the age contrast is not itself
 confounded with the group contrast.

 Everything else is identical to stage 4: the same landscapes, the same
 sup-over-(k,t) Welch statistic, the same exhaustive C(16,8) permutation set.
 The diagrams are read from disk, not recomputed.

 Usage:  python3 pipeline/05_diagnostico_edad.py
============================================================================
"""
import os, csv, json, itertools
import numpy as np
from scipy.interpolate import LSQUnivariateSpline

BASE = os.environ.get("RP", os.path.expanduser("~/rp"))
MAN  = f"{BASE}/T1_data_def/T1.csv"
TDA  = f"{BASE}/deriv/07_tda"

K_LAYERS, N_GRID = 5, 101

meta = {r["Subject"]: r for r in csv.DictReader(open(MAN))}
subjects = sorted(meta)
ages = np.array([float(meta[s]["Age"]) for s in subjects])
is_ctrl = np.array([meta[s]["Group"] == "Control" for s in subjects])

cut = float(np.median(ages))
is_young = ages < cut
assert is_young.sum() == 8, f"the median split gives {is_young.sum()} and {16-is_young.sum()}"

print(f"median age {cut:.1f}\n")
print(f"{'younger':>28} | {'older':>28}")
y = [f"{s}({meta[s]['Group'][0]},{meta[s]['Age']})" for s, k in zip(subjects, is_young) if k]
o = [f"{s}({meta[s]['Group'][0]},{meta[s]['Age']})" for s, k in zip(subjects, is_young) if not k]
for a, b in zip(y, o):
    print(f"{a:>28} | {b:>28}")
print(f"\n  younger: {sum(is_ctrl & is_young)} Control, {sum(~is_ctrl & is_young)} PD")
print(f"  older:   {sum(is_ctrl & ~is_young)} Control, {sum(~is_ctrl & ~is_young)} PD")


def landscape(dgm, grid, K=K_LAYERS):
    if len(dgm) == 0:
        return np.zeros((K, grid.size))
    b, d = dgm[:, 0], dgm[:, 1]
    tents = np.maximum(np.minimum(grid[None, :] - b[:, None],
                                  d[:, None] - grid[None, :]), 0.0)
    tents = np.sort(tents, axis=0)[::-1]
    L = np.zeros((K, grid.size))
    L[:min(K, tents.shape[0])] = tents[:K]
    return L


def smooth(curve, grid):
    """Same second-order basis as stage 4 and Definition def:bspline."""
    if not np.any(curve):
        return curve.copy()
    return np.maximum(LSQUnivariateSpline(grid, curve, grid[1:-2], k=1)(grid), 0.0)


def degenerate_columns(L, rel_tol=1e-9):
    """Grid points whose between-subject spread is numerically zero.

    Two families of column carry no information and must not enter the
    supremum, and neither is caught by a "variance > 0" guard:

      * near t = 0 every landscape equals t, because every dominant bar is
        born at zero. The sixteen curves are identical, the pooled variance is
        of order 1e-35, and the mean difference is zero. In exact arithmetic
        that is 0/0; in floating point it is noise divided by noise.
      * at the far tail a single subject can carry a spline residual of order
        1e-19 while the other fifteen are exactly zero, giving a variance of
        1e-37 and a ratio of order one out of nothing at all.

    Measured on this cohort, without this guard a relative perturbation of
    1e-15 -- machine epsilon -- moved the statistic by 0.76 on average and by
    up to 2.54, so the test was reproducible neither across implementations
    nor across machines.

    The threshold is on the variance relative to the scale of the data, and it
    is not a tuning parameter. In the main definition the same 106 of 505
    columns are excluded for every rel_tol between 1e-14 and 1e-4, because the
    largest excluded variance is 8.4145e-31 and the smallest included one is
    1.3666e-06: a gap of 24.2 orders of magnitude with nothing inside it.
    Across all eight combinations the count is stable to 1e-6, and only two of
    the four H1 cases move, by a single column, at 1e-4. It is computed from
    the POOLED sample, so it is invariant under relabelling and does not affect
    the validity of the permutation test.
    """
    scale = float(np.max(np.abs(L)))
    if scale == 0.0:
        return np.ones(L.shape[1], dtype=bool)
    var_all = L.var(axis=0, ddof=1)
    return var_all <= (rel_tol * scale) ** 2


def active_columns(L, rel_tol=1e-9):
    return ~degenerate_columns(L, rel_tol)

def lp_test(L, sel, dx, p):
    """L^p distance between group mean landscapes (Garg et al. 2017). No free
    parameter, and not subject to the edge-of-support pathology of the sup."""
    n, n1 = L.shape[0], int(sel.sum())
    def dist(m):
        d = np.abs(L[m].mean(0) - L[~m].mean(0))
        return float(d.max()) if np.isinf(p) else float((np.sum(d ** p) * dx) ** (1.0 / p))
    obs = dist(sel)
    T = np.array([dist(np.isin(np.arange(n), c))
                  for c in itertools.combinations(range(n), n1)])
    return float((T >= obs - 1e-12).mean())


def permutation_test(L, sel, block=2000):
    L = L[:, active_columns(L)]
    n, n1 = L.shape[0], int(sel.sum())
    combos = list(itertools.combinations(range(n), n1))
    T = np.empty(len(combos)); L2 = L ** 2
    for i0 in range(0, len(combos), block):
        chunk = combos[i0:i0 + block]
        P = np.zeros((len(chunk), n))
        for j, c in enumerate(chunk):
            P[j, list(c)] = 1.0
        Q, n2 = 1.0 - P, n - n1
        m1, m2 = (P @ L) / n1, (Q @ L) / n2
        v1 = ((P @ L2) - n1 * m1 ** 2) / (n1 - 1)
        v2 = ((Q @ L2) - n2 * m2 ** 2) / (n2 - 1)
        v = v1 / n1 + v2 / n2
        T[i0:i0 + len(chunk)] = np.where(
            v > 0, np.abs(m1 - m2) / np.sqrt(np.where(v > 0, v, 1)), 0.0).max(axis=1)
    A, B = L[sel], L[~sel]
    m = A.mean(0) - B.mean(0)
    v = A.var(0, ddof=1) / A.shape[0] + B.var(0, ddof=1) / B.shape[0]
    Tobs = float(np.where(v > 0, np.abs(m) / np.sqrt(np.where(v > 0, v, 1)), 0.0).max())
    return dict(Tobs=Tobs, crit95=float(np.quantile(T, 0.95)),
                p=float((T >= Tobs - 1e-12).mean()), nperm=len(combos))


results = {}
print(f"\n{'':>8} {'':>6} {'':>2} | {'--- AGE ---':^30} | {'--- DIAGNOSIS ---':^30}")
print(f"{'space':>8} {'def':>6} {'H':>2} | {'sup-t':>7} {'L1':>6} {'L2':>6} {'Linf':>6} "
      f"| {'sup-t':>7} {'L1':>6} {'L2':>6} {'Linf':>6}")
for space in ["native", "mni"]:
    for definition in ["main", "full"]:
        for k in (1, 2):
            paths = [f"{TDA}/{s}_H{k}_{space}_{definition}.csv" for s in subjects]
            if not all(os.path.exists(p) for p in paths):
                continue
            dgms = []
            for p in paths:
                D = np.loadtxt(p, delimiter=",", skiprows=1)
                dgms.append(np.atleast_2d(D) if D.size else np.empty((0, 2)))
            tmax = max((d[:, 1].max() if len(d) else 0.0) for d in dgms)
            grid = np.linspace(0.0, float(tmax), N_GRID)
            sm = np.stack([[smooth(landscape(d, grid)[j], grid) for j in range(K_LAYERS)]
                           for d in dgms])
            flat = sm.reshape(len(subjects), -1)
            dx = float(grid[1] - grid[0])
            ra = permutation_test(flat, is_young)
            rd = permutation_test(flat, is_ctrl)
            ra["lp"] = {n_: lp_test(flat, is_young, dx, v_)
                        for n_, v_ in [("L1", 1), ("L2", 2), ("Linf", np.inf)]}
            rd["lp"] = {n_: lp_test(flat, is_ctrl, dx, v_)
                        for n_, v_ in [("L1", 1), ("L2", 2), ("Linf", np.inf)]}
            results[f"{space}_{definition}_H{k}"] = dict(age=ra, diagnosis=rd)
            print(f"{space:>8} {definition:>6} {k:>2} | {ra['p']:7.4f} "
                  f"{ra['lp']['L1']:6.3f} {ra['lp']['L2']:6.3f} {ra['lp']['Linf']:6.3f} "
                  f"| {rd['p']:7.4f} {rd['lp']['L1']:6.3f} {rd['lp']['L2']:6.3f} "
                  f"{rd['lp']['Linf']:6.3f}")

with open(f"{TDA}/diagnostico_edad.json", "w") as fh:
    json.dump(dict(median_age=cut, subjects=subjects,
                   younger=[s for s, k in zip(subjects, is_young) if k],
                   results=results), fh, indent=2)
print(f"\nWritten to {TDA}/diagnostico_edad.json")
print("Post hoc. Reported as a diagnostic, not as a test of the hypothesis.")
