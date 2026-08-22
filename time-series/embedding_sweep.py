#!/usr/bin/env python3
"""
Cross the embedding dimension with the burn-in, and report what each changes.

Produces the four rows of Table 4.1 in the thesis. The question it answers is
whether the imperfect separation of Chapter 4 -- the 45 aperiodic series that
fall on the stable side of the leading functional principal component -- comes
from the embedding dimension or from the retained transient. The chapter used
to attribute it to the embedding dimension; this script is what showed that the
attribution was wrong.

Everything except (m, BURN_IN) is held at the values beetles_landscapes.ipynb
uses: the same LPA parameters, the same seed, the same 201-step series, the same
min-max normalisation, the same delay tau = 3, the same ripser call, and the
same data-adaptive 500-point grid built from the global birth-death range.

The m = 2, burn-in = 0 row must reproduce the committed run exactly: 0 of 200
stable diagrams empty and 45 of 200 aperiodic series on the stable side. If it
does not, something upstream has changed and the other three rows cannot be
trusted either.

Run:
    python3 embedding_sweep.py

Writes embedding_sweep_output.txt beside this file, which is the committed
artefact the thesis cites.
"""

import sys

import numpy as np
from ripser import ripser

# ---------------------------------------------------------------------------
# Parameters, identical to beetles_landscapes.ipynb
# ---------------------------------------------------------------------------
B, C_EA, C_PA, C_EL, U_L = 7.48, 0.009, 0.004, 0.012, 0.267
U_A_STABLE, U_A_APERIODIC = 0.73, 0.96
N_SIM, T_SIM = 200, 200
SEED = 20250729
EMBED_LAG = 3
N_BINS = 500


def simulate_beetles(u_a, L0, P0, A0, T=T_SIM, burn_in=0):
    """One LPA trajectory; returns the adult series A, length T + 1."""
    L_t, P_t, A_t = float(L0), float(P0), float(A0)
    for _ in range(burn_in):
        L_t, P_t, A_t = (
            B * A_t * np.exp(-C_EL * L_t - C_EA * A_t),
            L_t * (1 - U_L),
            P_t * np.exp(-C_PA * A_t) + A_t * (1 - u_a),
        )
    L, P, A = [L_t], [P_t], [A_t]
    for _ in range(T):
        L.append(B * A[-1] * np.exp(-C_EL * L[-1] - C_EA * A[-1]))
        P.append(L[-2] * (1 - U_L))
        A.append(P[-2] * np.exp(-C_PA * A[-1]) + A[-1] * (1 - u_a))
    return np.array(A)


def make_series(burn_in):
    """The 200 + 200 series, drawn in the notebook's order from the same seed."""
    rng = np.random.default_rng(SEED)
    stable, aperiodic = [], []
    for _ in range(N_SIM):
        L0, P0, A0 = rng.integers(2, 101, size=3)
        stable.append(simulate_beetles(U_A_STABLE, L0, P0, A0, burn_in=burn_in))
    for _ in range(N_SIM):
        L0, P0, A0 = rng.integers(2, 101, size=3)
        aperiodic.append(simulate_beetles(U_A_APERIODIC, L0, P0, A0, burn_in=burn_in))
    return np.array(stable), np.array(aperiodic)


def normalize_01(s):
    lo, hi = s.min(), s.max()
    return np.zeros_like(s, dtype=float) if hi == lo else (s - lo) / (hi - lo)


def takens_embedding(s, m, tau=EMBED_LAG):
    n = len(s) - (m - 1) * tau
    return np.array([[s[i + j * tau] for j in range(m)] for i in range(n)])


def finite_h1(pts):
    h1 = ripser(pts, maxdim=1)["dgms"][1]
    return h1[h1[:, 1] < np.inf]


def landscape_k1(bd, tseq):
    if len(bd) == 0:
        return np.zeros(len(tseq))
    tents = np.maximum(
        0.0,
        np.minimum(tseq[:, None] - bd[None, :, 0], bd[None, :, 1] - tseq[:, None]),
    )
    return tents.max(axis=1)


def one_cell(m, burn_in):
    """One row of the table."""
    stable, aperiodic = make_series(burn_in)
    stable_n = np.array([normalize_01(s) for s in stable])
    aperiodic_n = np.array([normalize_01(s) for s in aperiodic])

    dg_s = [finite_h1(takens_embedding(s, m)) for s in stable_n]
    dg_a = [finite_h1(takens_embedding(s, m)) for s in aperiodic_n]

    non_empty = [d for d in dg_s + dg_a if len(d)]
    if not non_empty:
        raise RuntimeError("every diagram is empty; nothing to build a grid on")
    allp = np.concatenate(non_empty)
    tseq = np.linspace(allp[:, 0].min(), allp[:, 1].max(), N_BINS)

    Ls = np.array([landscape_k1(d, tseq) for d in dg_s])
    La = np.array([landscape_k1(d, tseq) for d in dg_a])

    # FPCA on the pooled sample, oriented so the aperiodic group sits on the
    # positive side of the leading component; the count reported is how many
    # aperiodic series end up on the stable side of it.
    M = np.vstack([Ls, La])
    X = M - M.mean(axis=0)
    _, V = np.linalg.eigh(X.T @ X / (len(M) - 1))
    pc1 = X @ V[:, -1]
    if np.median(pc1[N_SIM:]) < np.median(pc1[:N_SIM]):
        pc1 = -pc1

    return dict(
        m=m,
        burn_in=burn_in,
        n_points=takens_embedding(stable_n[0], m).shape[0],
        empty_stable=sum(1 for d in dg_s if len(d) == 0),
        zero_stable=int((Ls.max(axis=1) == 0).sum()),
        aperiodic_on_stable_side=int((pc1[N_SIM:] < 0).sum()),
        peak_stable=float(Ls.max(axis=1).mean()),
        peak_aperiodic=float(La.max(axis=1).mean()),
        grid_max=float(tseq[-1]),
    )


def main():
    lines = []
    w = lines.append
    w("=" * 74)
    w("EMBEDDING DIMENSION x RETAINED TRANSIENT")
    w("=" * 74)
    w("")
    w("  Everything except (m, burn-in) is held at the values of")
    w("  beetles_landscapes.ipynb.  tau = 3, seed = %d, 200 series per regime." % SEED)
    w("")
    w("  %-3s %-8s %-7s %-14s %-14s %-12s" % ("m", "burn-in", "points",
                                              "H1 diag empty", "zero landscape",
                                              "aperiodic on"))
    w("  %-3s %-8s %-7s %-14s %-14s %-12s" % ("", "", "/cloud", "(stable)",
                                              "(stable)", "stable side"))
    w("  " + "-" * 66)
    rows = []
    for burn_in in (0, 200):
        for m in (2, 3):
            r = one_cell(m, burn_in)
            rows.append(r)
            w("  %-3d %-8d %-7d %-14s %-14s %-12s"
              % (r["m"], r["burn_in"], r["n_points"],
                 "%d/200" % r["empty_stable"],
                 "%d/200" % r["zero_stable"],
                 "%d/200" % r["aperiodic_on_stable_side"]))
    w("")
    w("  mean landscape peak, by cell:")
    for r in rows:
        ratio = (r["peak_aperiodic"] / r["peak_stable"]) if r["peak_stable"] else float("inf")
        w("    m=%d burn-in=%-4d  stable %.5f   aperiodic %.5f   ratio %s   grid max %.5f"
          % (r["m"], r["burn_in"], r["peak_stable"], r["peak_aperiodic"],
             ("%.1f" % ratio) if np.isfinite(ratio) else "inf", r["grid_max"]))
    w("")

    # The control: the committed run must come back unchanged.
    base = rows[0]
    ok = base["empty_stable"] == 0 and base["aperiodic_on_stable_side"] == 45
    w("  control (m=2, burn-in=0 must reproduce the committed run): %s"
      % ("PASS" if ok else "FAIL"))
    if not ok:
        w("    expected 0/200 empty and 45/200 on the stable side, got %d/200 and %d/200"
          % (base["empty_stable"], base["aperiodic_on_stable_side"]))
    w("")
    w("  Reading: raising m to the value Takens' bound requires leaves the stable")
    w("  diagrams non-empty and moves the overlap only a little.  Discarding the")
    w("  transient empties them and removes most of the overlap, at either m.")

    text = "\n".join(lines)
    print(text)
    with open("embedding_sweep_output.txt", "w") as f:
        f.write(text + "\n")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
