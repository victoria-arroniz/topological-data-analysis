"""
regime_diagnostics.py
=====================
Diagnostics backing three claims made in Chapter 4 (Time Series) that are not
produced by beetles_landscapes.ipynb:

  (1) The aperiodic regime (mu_a = 0.96) is QUASI-PERIODIC, not chaotic:
      the largest Lyapunov exponent of the deterministic LPA map is zero to
      numerical precision and the other two are negative, so the attractor is
      an invariant closed curve.

  (2) The 129/200 identically-zero stable landscapes are a GRID-RESOLUTION
      effect, not empty H1 diagrams: every stable series carries H1 features,
      but their lifetimes fall below the spacing of the shared evaluation grid.

  (3) Discarding an initial transient (burn-in) removes the stable loops
      entirely and strengthens the aperiodic signal.

Parameters are those of Costantino et al. (1995), SS strain:
    b = 7.48, c_ea = 0.009, c_pa = 0.004, mu_l = 0.267, c_el = 0.012 (MLE),
    mu_a = 0.73 (stable equilibria) / 0.96 (aperiodic oscillations).

Usage:  python regime_diagnostics.py
Deps:   numpy, ripser
"""

import numpy as np
from ripser import ripser

# ---------------------------------------------------------------- parameters
B, C_EA, C_PA, C_EL, U_L = 7.48, 0.009, 0.004, 0.012, 0.267
U_A_STABLE, U_A_APERIODIC = 0.73, 0.96
N_SIM, T_SIM = 200, 200          # 200 series per regime, 201 observations each
EMBED_DIM, EMBED_LAG = 2, 3
N_BINS = 500
SEED = 20250729


# ---------------------------------------------------------------- LPA map
def lpa_step(state, u_a):
    """One iteration of the deterministic LPA map."""
    L, P, A = state
    return np.array([
        B * A * np.exp(-C_EL * L - C_EA * A),
        L * (1.0 - U_L),
        P * np.exp(-C_PA * A) + A * (1.0 - u_a),
    ])


def lpa_jacobian(state, u_a):
    """Jacobian of the LPA map, used for the Lyapunov spectrum."""
    L, P, A = state
    e = np.exp(-C_EL * L - C_EA * A)
    return np.array([
        [-C_EL * B * A * e, 0.0, B * e - C_EA * B * A * e],
        [1.0 - U_L,         0.0, 0.0],
        [0.0, np.exp(-C_PA * A), -C_PA * P * np.exp(-C_PA * A) + (1.0 - u_a)],
    ])


def simulate(u_a, L0, P0, A0, n_steps, burn_in=0):
    """Iterate the map and return the adult series A_t (length n_steps+1)."""
    state = np.array([float(L0), float(P0), float(A0)])
    out = [state[2]]
    for _ in range(n_steps + burn_in):
        state = lpa_step(state, u_a)
        out.append(state[2])
    return np.array(out)[burn_in:]


# ---------------------------------------------------------------- TDA helpers
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
    tents = np.maximum(
        0.0,
        np.minimum(tseq[:, None] - bd[:, 0][None, :],
                   bd[:, 1][None, :] - tseq[:, None]),
    )
    return tents.max(axis=1)


# ---------------------------------------------------------------- (1) Lyapunov
def lyapunov_spectrum(u_a, n_transient=20_000, n_iter=200_000):
    """Full Lyapunov spectrum via QR reorthonormalisation."""
    state = np.array([50.0, 30.0, 60.0])
    for _ in range(n_transient):
        state = lpa_step(state, u_a)
    Q = np.eye(3)
    total = np.zeros(3)
    for _ in range(n_iter):
        Q, R = np.linalg.qr(lpa_jacobian(state, u_a) @ Q)
        total += np.log(np.abs(np.diag(R)))
        state = lpa_step(state, u_a)
    return total / n_iter


def report_lyapunov():
    print("=" * 68)
    print("(1) LYAPUNOV SPECTRUM  -- is the aperiodic regime chaotic?")
    print("=" * 68)
    for u_a, name in [(U_A_STABLE, "stable   "), (U_A_APERIODIC, "aperiodic")]:
        lam = lyapunov_spectrum(u_a)
        if lam[0] > 1e-4:
            verdict = "CHAOTIC (largest exponent > 0)"
        elif lam[0] > -1e-4:
            verdict = "QUASI-PERIODIC (largest exponent = 0): invariant closed curve"
        else:
            verdict = "FIXED POINT / PERIODIC (all exponents < 0)"
        print(f"  mu_a = {u_a}  [{name}]  spectrum = "
              f"[{lam[0]:+.6f}, {lam[1]:+.6f}, {lam[2]:+.6f}]")
        print(f"      -> {verdict}\n")


# ------------------------------------------- (2) grid resolution & (3) burn-in
def run_regimes(burn_in, rng):
    """Simulate both regimes, return H1 diagrams and the shared grid."""
    stable, aperiodic = [], []
    for _ in range(N_SIM):
        ic = rng.integers(2, 101, size=3)
        stable.append(finite_h1(simulate(U_A_STABLE, *ic, T_SIM, burn_in)))
    for _ in range(N_SIM):
        ic = rng.integers(2, 101, size=3)
        aperiodic.append(finite_h1(simulate(U_A_APERIODIC, *ic, T_SIM, burn_in)))
    pooled = [p for p in stable + aperiodic if len(p)]
    allp = np.concatenate(pooled)
    tseq = np.linspace(allp[:, 0].min(), allp[:, 1].max(), N_BINS)
    return stable, aperiodic, tseq


def summarise(stable, aperiodic, tseq, label):
    spacing = tseq[1] - tseq[0]
    LS = np.array([landscape_k1(p, tseq) for p in stable])
    LA = np.array([landscape_k1(p, tseq) for p in aperiodic])
    n_empty = sum(len(p) == 0 for p in stable)
    n_zero = int((LS.max(axis=1) == 0).sum())
    lifetimes = [float((p[:, 1] - p[:, 0]).max()) if len(p) else 0.0 for p in stable]
    print(f"  --- {label} ---")
    print(f"    grid: {N_BINS} points on [{tseq[0]:.4f}, {tseq[-1]:.4f}], "
          f"spacing = {spacing:.5f}")
    print(f"    stable, EMPTY H1 diagram        : {n_empty:3d}/{N_SIM}")
    print(f"    stable, landscape zero ON GRID  : {n_zero:3d}/{N_SIM} "
          f"({100 * n_zero / N_SIM:.0f}%)")
    print(f"    stable, median max H1 lifetime  : {np.median(lifetimes):.5f} "
          f"(vs grid spacing {spacing:.5f})")
    print(f"    aperiodic, landscape zero       : "
          f"{int((LA.max(axis=1) == 0).sum()):3d}/{N_SIM}")
    print(f"    mean landscape peak  stable = {LS.max(axis=1).mean():.5f}   "
          f"aperiodic = {LA.max(axis=1).mean():.5f}\n")


def report_grid_and_burnin():
    rng = np.random.default_rng(SEED)
    print("=" * 68)
    print("(2)/(3) GRID RESOLUTION AND THE EFFECT OF A BURN-IN")
    print("=" * 68)
    print("  If 'EMPTY H1 diagram' is 0 but 'landscape zero ON GRID' is large,")
    print("  the vanishing landscapes are a resolution artefact, not absent loops.\n")
    for burn_in, label in [(0, "NO burn-in (as committed)"),
                           (200, "burn-in = 200 (series length unchanged)")]:
        s, a, t = run_regimes(burn_in, rng)
        summarise(s, a, t, label)


if __name__ == "__main__":
    report_lyapunov()
    report_grid_and_burnin()
