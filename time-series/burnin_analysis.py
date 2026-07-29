"""
burnin_analysis.py
==================
Regenerates the Part B landscapes with an initial transient (burn-in) discarded,
and recomputes the downstream functional analysis, so that the comparison is
between the two ATTRACTORS rather than between "approach + attractor" and
"attractor".

Motivation: Costantino et al. (1995) classify the ASYMPTOTIC dynamics, and warn
that "oscillatory approaches to point equilibria, stable periodic oscillations
and aperiodic oscillations are difficult to distinguish in short time series
with model-free methods". Persistent homology on a 201-step run is exactly such
a model-free method, and regime_diagnostics.py shows that without a burn-in
every stable series carries transient loops.

Design: the burn-in is simulated IN ADDITION to the 201 retained steps, so the
series length, embedding, grid size and every downstream stage are unchanged.
Only the starting point on the trajectory moves.

Both variants are produced:
    output_burnin/   -- burn-in discarded  (proposed main analysis)
    output/          -- as committed, no burn-in (kept for the sensitivity
                        comparison; NOT overwritten by this script)

The FDA stage replicates results_partB.qmd:
  * order-2 B-spline basis, nbasis = 499, over range(tseq)   -> interpolation,
    so PCA on the sampled curves reproduces pca.fd (verified against Part A,
    which returns 63.5% / 20.0% either way)
  * permutation test evaluated on argvals = seq(min, max, length.out = 101)
  * statistic  T = max |m1 - m2| / sqrt(v1/n1 + v2/n2)  over points where the
    pooled variance is strictly positive

Usage:  python burnin_analysis.py
Deps:   numpy, ripser
"""

import os
import numpy as np
from ripser import ripser

# ---------------------------------------------------------------- parameters
B, C_EA, C_PA, C_EL, U_L = 7.48, 0.009, 0.004, 0.012, 0.267
U_A_STABLE, U_A_APERIODIC = 0.73, 0.96
N_SIM, T_SIM = 200, 200
BURN_IN = 200                     # <-- the change; 0 reproduces the committed run
EMBED_DIM, EMBED_LAG = 2, 3
N_BINS = 500
N_ARGVALS = 101
NPERM_MAIN = 10_000
N_SPLITS, NPERM_CTRL = 200, 1_000  # control: 200 splits, reduced Monte Carlo
SEED = 20250729

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "output_burnin")


# ---------------------------------------------------------------- model / TDA
def simulate(u_a, L0, P0, A0, n_keep=T_SIM, burn_in=BURN_IN):
    """Adult series A_t of length n_keep+1, after discarding `burn_in` steps."""
    L, P, A = float(L0), float(P0), float(A0)
    for _ in range(burn_in):
        L, P, A = (B * A * np.exp(-C_EL * L - C_EA * A),
                   L * (1.0 - U_L),
                   P * np.exp(-C_PA * A) + A * (1.0 - u_a))
    out = [A]
    for _ in range(n_keep):
        L, P, A = (B * A * np.exp(-C_EL * L - C_EA * A),
                   L * (1.0 - U_L),
                   P * np.exp(-C_PA * A) + A * (1.0 - u_a))
        out.append(A)
    return np.array(out)


def normalise_01(s):
    lo, hi = s.min(), s.max()
    return np.zeros_like(s, dtype=float) if hi == lo else (s - lo) / (hi - lo)


def takens(s, m=EMBED_DIM, tau=EMBED_LAG):
    n = len(s) - (m - 1) * tau
    return np.array([[s[i + j * tau] for j in range(m)] for i in range(n)])


def finite_h1(series):
    d = ripser(takens(normalise_01(series)), maxdim=1)["dgms"][1]
    return d[np.isfinite(d[:, 1])]


def landscape_k1(bd, tseq):
    if len(bd) == 0:
        return np.zeros(len(tseq))
    return np.maximum(0.0, np.minimum(tseq[:, None] - bd[:, 0][None, :],
                                      bd[:, 1][None, :] - tseq[:, None])).max(axis=1)


# ---------------------------------------------------------------- statistics
def t_statistic(M, n1):
    """max |m1-m2| / sqrt(v1/n1 + v2/n2) over points with positive pooled var."""
    a, b = M[:, :n1], M[:, n1:]
    n2 = b.shape[1]
    den = a.var(axis=1, ddof=1) / n1 + b.var(axis=1, ddof=1) / n2
    ok = den > 0
    if not ok.any():
        return np.nan                     # fully degenerate: statistic undefined
    return float(np.max(np.abs(a.mean(1)[ok] - b.mean(1)[ok]) / np.sqrt(den[ok])))


def permutation_test(M, n1, nperm, rng):
    obs = t_statistic(M, n1)
    if np.isnan(obs):
        return dict(pval=np.nan, Tobs=np.nan, crit95=np.nan, degenerate=True)
    null = np.empty(nperm)
    n = M.shape[1]
    for i in range(nperm):
        null[i] = t_statistic(M[:, rng.permutation(n)], n1)
    return dict(pval=float(np.mean(obs < null)), Tobs=obs,
                crit95=float(np.quantile(null, 0.95)), degenerate=False)


def fpca_pve(M):
    """Percent variance explained by the leading components (centred SVD)."""
    C = M - M.mean(axis=0)
    s = np.linalg.svd(C, compute_uv=False)
    return 100 * s**2 / np.sum(s**2)


# ---------------------------------------------------------------- pipeline
def main():
    rng = np.random.default_rng(SEED)
    os.makedirs(OUT, exist_ok=True)

    print(f"Simulating {N_SIM} series per regime, burn-in = {BURN_IN}, "
          f"retained length = {T_SIM + 1}\n")
    dgm_s, dgm_a = [], []
    for _ in range(N_SIM):
        dgm_s.append(finite_h1(simulate(U_A_STABLE, *rng.integers(2, 101, 3))))
    for _ in range(N_SIM):
        dgm_a.append(finite_h1(simulate(U_A_APERIODIC, *rng.integers(2, 101, 3))))

    n_empty_s = sum(len(d) == 0 for d in dgm_s)
    n_empty_a = sum(len(d) == 0 for d in dgm_a)
    pooled = [d for d in dgm_s + dgm_a if len(d)]
    allp = np.concatenate(pooled)
    tseq = np.linspace(allp[:, 0].min(), allp[:, 1].max(), N_BINS)

    LS = np.array([landscape_k1(d, tseq) for d in dgm_s])
    LA = np.array([landscape_k1(d, tseq) for d in dgm_a])
    np.savetxt(os.path.join(OUT, "stable_landscapes.csv"), LS, delimiter=",")
    np.savetxt(os.path.join(OUT, "aperiodic_landscapes.csv"), LA, delimiter=",")
    np.savetxt(os.path.join(OUT, "tseq.csv"), tseq, delimiter=",")

    print("=" * 70)
    print("TOPOLOGY")
    print("=" * 70)
    print(f"  stable   : empty H1 diagrams {n_empty_s:3d}/{N_SIM}   "
          f"zero landscapes {int((LS.max(1) == 0).sum()):3d}/{N_SIM}")
    print(f"  aperiodic: empty H1 diagrams {n_empty_a:3d}/{N_SIM}   "
          f"zero landscapes {int((LA.max(1) == 0).sum()):3d}/{N_SIM}")
    print(f"  mean landscape peak: stable {LS.max(1).mean():.5f}   "
          f"aperiodic {LA.max(1).mean():.5f}")
    print(f"  grid: [{tseq[0]:.4f}, {tseq[-1]:.4f}]\n")

    # ---- FPCA on the pooled sample
    both = np.vstack([LS, LA])
    pve = fpca_pve(both)
    scores = (both - both.mean(0)) @ np.linalg.svd(
        both - both.mean(0), full_matrices=False)[2][:2].T
    if scores[:N_SIM, 0].mean() > 0:       # orient PC1 so stable is negative
        scores[:, 0] *= -1
    print("=" * 70)
    print("FPCA")
    print("=" * 70)
    print(f"  PC1 {pve[0]:.2f}%   PC2 {pve[1]:.2f}%   cumulative {pve[0]+pve[1]:.2f}%")
    print(f"  stable with PC1 < 0   : {int((scores[:N_SIM,0] < 0).sum())}/{N_SIM}")
    print(f"  aperiodic with PC1 < 0: {int((scores[N_SIM:,0] < 0).sum())}/{N_SIM}\n")

    # ---- permutation test on the 101-point grid
    argvals = np.linspace(tseq.min(), tseq.max(), N_ARGVALS)
    interp = lambda M: np.array([np.interp(argvals, tseq, row) for row in M]).T
    X = np.hstack([interp(LS), interp(LA)])

    print("=" * 70)
    print("PERMUTATION TEST  (stable vs aperiodic)")
    print("=" * 70)
    res = permutation_test(X, N_SIM, NPERM_MAIN, rng)
    print(f"  T_max = {res['Tobs']:.2f}   crit95 = {res['crit95']:.2f}   "
          f"p = {res['pval']:.4f}"
          + ("  (p below 1/nperm)" if res["pval"] == 0 else "") + "\n")

    # ---- within-regime controls
    print("=" * 70)
    print(f"WITHIN-REGIME CONTROLS  ({N_SPLITS} random 100-vs-100 splits, "
          f"{NPERM_CTRL} perms)")
    print("=" * 70)
    for name, L in [("stable", interp(LS)), ("aperiodic", interp(LA))]:
        pv, n_deg = [], 0
        for _ in range(N_SPLITS):
            idx = rng.permutation(N_SIM)
            r = permutation_test(L[:, idx], N_SIM // 2, NPERM_CTRL, rng)
            if r["degenerate"]:
                n_deg += 1
            else:
                pv.append(r["pval"])
        if n_deg == N_SPLITS:
            print(f"  {name:9s}: ALL {N_SPLITS} splits DEGENERATE -- every curve is "
                  f"identically zero, so the\n"
                  f"             statistic has no point with positive pooled "
                  f"variance and is undefined.\n"
                  f"             There is no topological signal left to test, "
                  f"which is the intended outcome.\n")
        else:
            pv = np.array(pv)
            print(f"  {name:9s}: mean p {pv.mean():.3f}  sd {pv.std():.3f}  "
                  f"%<0.05 {100*np.mean(pv < 0.05):.1f}%"
                  + (f"  ({n_deg} degenerate splits skipped)" if n_deg else "") + "\n")

    print(f"Landscapes written to {OUT}/")


if __name__ == "__main__":
    main()
