#!/usr/bin/env python3
"""
============================================================================
 Chapter 5 — Stage 4: persistent homology, landscapes, FPCA and the test
============================================================================

 THE FILTRATION  (PREREGISTRO_CAP5.md section 7.2)
 -------------------------------------------------
 Tissue is defined as the COMPLEMENT of the cavity, and the filtration is by
 sublevel sets of

     d(x) = dist(x, C^c),   computed inside C, physical sampling = voxel size

 In this construction {d <= 0} = C^c, so every H2 class of a bounded component
 of int(C) is born at t = 0, and the largest death is max_{x in C} d(x), the
 radius of the largest inscribed ball. Persistence therefore equals that radius.

 This is the whole point of using the complement rather than a brain mask: with
 the earlier definition the class was born late whenever the mask touched the
 edge of the brain mask, and the measured value fell short of the inradius by up
 to 2.24 mm.

 WHAT IS MEASURED
 ----------------
   spaces        native (eTIV-corrected) and MNI 1 mm
   definitions   main = labels 4,43        sensitivity = labels 4,43,5,44
   homology      H2 (the chapter's question) and H1 (rem:voids-vs-loops)

 All four space x definition combinations are reported, and both homological
 orders, whatever they show.

 LANDSCAPES AND THE STATISTIC  (sections 7.3 and 5)
 --------------------------------------------------
 K = 5 layers, evaluated on a common grid in millimetres, represented in the
 second-order B-spline basis of Definition def:bspline. Pre-registered statistic:

     T = sup over k in 1..K and over the grid of
         |mean1 - mean2| / sqrt(var1/n1 + var2/n2)

 calibrated by the EXHAUSTIVE set of C(16,8) = 12870 label assignments.

 Reported alongside it, and DECLARED POST HOC in section 10.4 of the
 pre-registration, are the L1, L2 and L-infinity distances between the group
 mean landscapes: the statistic of Garg et al. (2017), which carries no free
 parameter and is not subject to the edge-of-support pathology that the
 pointwise standardised supremum shows on these data.

 Note on the achievable minimum p-value: with two groups of equal size, a
 selection S and its complement give the same statistic, so every value appears
 twice and the smallest reachable p is 2/12870, not 1/12870.

 Usage:  python3 pipeline/04_analisis.py
 Output: deriv/07_tda/
============================================================================
"""
import os, csv, json, itertools
import numpy as np
import nibabel as nib
from scipy import ndimage as ndi
from scipy.interpolate import LSQUnivariateSpline
import cripser

BASE = os.environ.get("RP", os.path.expanduser("~/rp"))
MAN  = f"{BASE}/T1_data_def/T1.csv"
SEG  = f"{BASE}/deriv/05_synthseg"
MNI  = f"{BASE}/deriv/06_mni"
OUT  = f"{BASE}/deriv/07_tda"
os.makedirs(OUT, exist_ok=True)

K_LAYERS = 5          # section 7.3
N_GRID   = 101
BIG      = 1e5        # cripser marks the essential class with DBL_MAX

meta = {r["Subject"]: r for r in csv.DictReader(open(MAN))}
assert len(meta) == 16, f"expected 16 subjects, the manifest has {len(meta)}"
subjects = sorted(meta)
groups   = np.array([meta[s]["Group"] for s in subjects])
is_ctrl  = groups == "Control"
assert is_ctrl.sum() == 8 and (~is_ctrl).sum() == 8, "expected 8 and 8"


# ---------------------------------------------------------------------------
# 1. Filtration and persistent homology
# ---------------------------------------------------------------------------
def diagrams(mask_path):
    """Complement filtration on the cavity. Returns {1: (b,d), 2: (b,d)}."""
    img = nib.load(mask_path)
    C = np.asarray(img.dataobj) > 0.5
    vox = tuple(float(z) for z in img.header.get_zooms()[:3])
    if C.sum() == 0:
        return {1: np.empty((0, 2)), 2: np.empty((0, 2))}
    # crop to the bounding box with a margin, purely for speed
    sl = tuple(slice(max(0, q.start - 6), q.stop + 6)
               for q in ndi.find_objects(C.astype(np.uint8))[0])
    d = ndi.distance_transform_edt(C[sl], sampling=vox)
    pd = cripser.computePH(np.ascontiguousarray(d, dtype=np.float64), maxdim=2)
    out = {}
    for k in (1, 2):
        h = pd[pd[:, 0] == k]
        h = h[np.abs(h[:, 2]) < BIG]          # drop the essential class
        out[k] = h[:, 1:3].copy()
    return out


# ---------------------------------------------------------------------------
# 2. Landscapes
# ---------------------------------------------------------------------------
def landscape(dgm, grid, K=K_LAYERS):
    """Bubenik landscape, layers 1..K, on a fixed grid."""
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
    """B-spline representation exactly as Definition def:bspline of Chapter 2.

    SECOND-ORDER (piecewise linear) basis with L = Q - 1 functions, which for
    Q = 101 grid points means 98 interior knots and a residual space of
    dimension one. Chapter 2 is explicit that this is a change of
    representation, not denoising: a landscape is genuinely piecewise linear
    and its kinks carry the individual birth-death pairs.

    An earlier version used a cubic basis with 20 interior knots. That
    introduced errors of up to 0.106 mm, the same size as the between-group
    difference being measured, and produced negative values in every subject,
    which a landscape cannot take by definition.

    The linear basis is near-interpolating but not interpolating. With
    L = Q - 1 the residual space has dimension one and it sits in the last
    interval [t_98, t_100], over which the basis is linear, so a kink at t_99
    cannot be reproduced. On this cohort the resulting error is at rounding
    level in native space (1e-17) but reaches 1.2646e-02 mm in MNI, on the
    second layer of one subject. That is 12% of the between-group difference.
    """
    if not np.any(curve):
        return curve.copy()
    interior = grid[1:-2]                      # 98 knots -> L = Q - 1 = 100
    fit = LSQUnivariateSpline(grid, curve, interior, k=1)(grid)
    # A landscape is non-negative by definition: the tent functions of
    # def:persistence-landscape are positive parts. The same last-interval
    # residual drives the fit negative past the final kink -- measured minimum
    # -6.323e-03 -- and the fitted object then is not a landscape. Clamp.
    return np.maximum(fit, 0.0)


def sup_location(L, sel, grid_size):
    """Where the supremum falls, and how many curves are non-zero there.

    Diagnostic for a known pathology. The supremum of a POINTWISE STANDARDISED
    statistic over functions of compact support is unstable at the edge of the
    support: almost every curve is exactly zero there, the pooled variance
    collapses, and the ratio is driven by the two or three that are not.

    Columns where every landscape vanishes are excluded by active_columns();
    without that the statistic is not even reproducible across implementations,
    let alone interpretable. What remains, and is reported here rather than
    corrected, is that the supremum still lands near the edge of the support,
    where few curves are non-zero. The L^p family below carries no such
    dependence and is reported alongside.
    """
    active = active_columns(L)
    A, B = L[sel], L[~sel]
    m = A.mean(0) - B.mean(0)
    v = A.var(0, ddof=1) / A.shape[0] + B.var(0, ddof=1) / B.shape[0]
    ok = active & (v > 0)
    z = np.where(ok, np.abs(m) / np.sqrt(np.where(ok, v, 1)), 0.0)
    j = int(z.argmax()); layer, q = divmod(j, grid_size)
    return dict(layer=layer + 1, grid_index=q,
                curves_nonzero_there=int((L[:, j] != 0).sum()),
                n_subjects=int(L.shape[0]))


def lp_test(L, sel, dx, p):
    """L^p distance between the group mean landscapes, calibrated by the same
    exhaustive permutation set. This is the statistic of Garg et al. (2017),
    who report p = 1, 2 and infinity. DECLARED POST HOC: it was added after
    the pre-registered result, as a check on the pathology above.
    """
    n, n1 = L.shape[0], int(sel.sum())

    def dist(m):
        d = np.abs(L[m].mean(0) - L[~m].mean(0))
        return float(d.max()) if np.isinf(p) else float((np.sum(d ** p) * dx) ** (1.0 / p))

    obs = dist(sel)
    T = np.array([dist(np.isin(np.arange(n), c))
                  for c in itertools.combinations(range(n), n1)])
    return dict(stat=obs, crit95=float(np.quantile(T, 0.95)),
                p=float((T >= obs - 1e-12).mean()))


# ---------------------------------------------------------------------------
# 3. Exhaustive permutation test, statistic = sup over layers and grid
# ---------------------------------------------------------------------------
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

def welch_sup(L, sel, active=None):
    """L: (n_subjects, n_features). sel: boolean group-1 mask."""
    if active is None:
        active = active_columns(L)
    A, B = L[sel], L[~sel]
    m = A.mean(0) - B.mean(0)
    v = A.var(0, ddof=1) / A.shape[0] + B.var(0, ddof=1) / B.shape[0]
    ok = active & (v > 0)
    z = np.zeros_like(m)
    z[ok] = np.abs(m[ok]) / np.sqrt(v[ok])
    return float(z.max()) if z.size else 0.0


def permutation_test(L, sel, block=2000):
    active = active_columns(L)
    Lm = L[:, active]
    n, n1 = Lm.shape[0], int(sel.sum())
    combos = list(itertools.combinations(range(n), n1))
    T = np.empty(len(combos))
    L2 = Lm ** 2
    for i0 in range(0, len(combos), block):
        chunk = combos[i0:i0 + block]
        P = np.zeros((len(chunk), n))
        for j, c in enumerate(chunk):
            P[j, list(c)] = 1.0
        Q = 1.0 - P
        n2 = n - n1
        m1, m2 = (P @ Lm) / n1, (Q @ Lm) / n2
        v1 = ((P @ L2) - n1 * m1 ** 2) / (n1 - 1)
        v2 = ((Q @ L2) - n2 * m2 ** 2) / (n2 - 1)
        v = v1 / n1 + v2 / n2
        z = np.where(v > 0, np.abs(m1 - m2) / np.sqrt(np.where(v > 0, v, 1)), 0.0)
        T[i0:i0 + len(chunk)] = z.max(axis=1)
    Tobs = welch_sup(L, sel, active)
    p = float((T >= Tobs - 1e-12).mean())
    return dict(Tobs=Tobs, crit95=float(np.quantile(T, 0.95)), p=p,
                nperm=len(combos), p_min=float(2.0 / len(combos)),
                active_columns=int(active.sum()), total_columns=int(active.size))


def fpca(L, dx, n_comp=3):
    """PCA of the centred curves under the L2 inner product.

    Proposition prop:kl-expansion defines the scores as L2 inner products, so
    the discretised curves carry the quadrature weight sqrt(dx). The grid is
    uniform, so the explained-variance ratios are unaffected, but without the
    weight the eigenvalues nu_p are off by a factor of dx and are not the
    quantities the proposition names.

    The same proposition takes nu_p to be an eigenvalue of the sample
    covariance of def:mean-cov, which carries the 1/(N-1) factor, and requires
    Var(xi_p) = nu_p. The squared singular values are therefore (N-1) times too
    large on their own. On this cohort that is a factor of fifteen: the raw
    S**2 gives nu_1 = 81.42 where the sample variance of the first scores, and
    hence the nu_1 the proposition names, is 5.43. Ratios are unaffected, so
    the reported explained variance does not move.
    """
    X = (L - L.mean(0)) * np.sqrt(dx)
    U, S, _ = np.linalg.svd(X, full_matrices=False)
    ev = S ** 2 / (L.shape[0] - 1)
    return dict(eigenvalues=[float(x) for x in ev[:n_comp]],
                explained=[float(x) for x in (ev / ev.sum())[:n_comp]],
                scores=(U[:, :n_comp] * S[:n_comp]).tolist())


# ---------------------------------------------------------------------------
# 4. Run every combination
# ---------------------------------------------------------------------------
etiv = {r["subject"]: float(r["eTIV_mL"])
        for r in csv.DictReader(open(f"{SEG}/volumenes_ventriculos.csv"))}
etiv_ref = float(np.mean(list(etiv.values())))

CASES = [("native", "main", lambda s: f"{SEG}/{s}_vent.nii.gz",      True),
         ("native", "full", lambda s: f"{SEG}/{s}_vent_full.nii.gz", True),
         ("mni",    "main", lambda s: f"{MNI}/{s}_vent_mni.nii.gz",  False),
         ("mni",    "full", lambda s: f"{MNI}/{s}_vent_full_mni.nii.gz", False)]

results = {}
for space, definition, path_of, correct_etiv in CASES:
    if not all(os.path.exists(path_of(s)) for s in subjects):
        print(f"[skip] {space}/{definition}: masks not found "
              f"(run 03_espacio_comun.sh for the MNI arm)")
        continue
    tag = f"{space}_{definition}"
    print(f"\n=== {tag} ===")

    dgms = {}
    for s in subjects:
        D = diagrams(path_of(s))
        if correct_etiv:
            # head size correction: a length scales as the cube root of a volume
            f = (etiv_ref / etiv[s]) ** (1.0 / 3.0)
            D = {k: v * f for k, v in D.items()}
        dgms[s] = D
        for k in (1, 2):
            np.savetxt(f"{OUT}/{s}_H{k}_{tag}.csv", D[k],
                       delimiter=",", header="birth,death", comments="")
        print(f"  {s} {meta[s]['Group']:>7}  H1 {len(D[1]):4d} bars   "
              f"H2 {len(D[2]):4d} bars   max H2 persistence "
              f"{(D[2][:,1]-D[2][:,0]).max() if len(D[2]) else 0:.4f} mm")

    for k in (1, 2):
        # the grid spans all subjects, which is invariant under relabelling and
        # therefore does not affect the validity of the permutation test
        tmax = max((dgms[s][k][:, 1].max() if len(dgms[s][k]) else 0.0)
                   for s in subjects)
        grid = np.linspace(0.0, float(tmax), N_GRID)
        raw = np.stack([landscape(dgms[s][k], grid) for s in subjects])       # (n,K,G)
        sm = np.stack([[smooth(raw[i, j], grid) for j in range(K_LAYERS)]
                       for i in range(len(subjects))])
        flat = sm.reshape(len(subjects), -1)                                   # (n, K*G)
        dx = float(grid[1] - grid[0])

        res = permutation_test(flat, is_ctrl)
        res.update(homology=k, space=space, definition=definition,
                   K=K_LAYERS, grid_max=float(tmax),
                   sup_at=sup_location(flat, is_ctrl, N_GRID),
                   lp_posthoc={nm: lp_test(flat, is_ctrl, dx, pv)
                               for nm, pv in [("L1", 1), ("L2", 2), ("Linf", np.inf)]},
                   fpca_all_layers=fpca(flat, dx),
                   fpca_layer1=fpca(sm[:, 0, :], dx),
                   peak_control=float(sm[is_ctrl, 0].mean(0).max()),
                   peak_pd=float(sm[~is_ctrl, 0].mean(0).max()))
        results[f"{tag}_H{k}"] = res
        print(f"  H{k}: T = {res['Tobs']:.4f}   crit95 = {res['crit95']:.4f}   "
              f"p = {res['p']:.4f}   (p_min = {res['p_min']:.6f})")
        lp = res['lp_posthoc']; sa = res['sup_at']
        print(f"        post hoc L1/L2/Linf p = {lp['L1']['p']:.4f} / "
              f"{lp['L2']['p']:.4f} / {lp['Linf']['p']:.4f}   |   sup at layer "
              f"{sa['layer']}, {sa['curves_nonzero_there']}/{sa['n_subjects']} curves non-zero")

        np.savetxt(f"{OUT}/landscapes_{tag}_H{k}.csv",
                   np.column_stack([grid, sm[:, 0, :].T]), delimiter=",",
                   header="t_mm," + ",".join(subjects), comments="")

with open(f"{OUT}/resultados_test.json", "w") as fh:
    json.dump(dict(subjects=subjects, groups=groups.tolist(),
                   etiv_ref_mL=etiv_ref, results=results), fh, indent=2)

print(f"\nWritten to {OUT}/resultados_test.json")
print("Report every combination that ran, whatever it shows (sections 3 and 8.5).")
