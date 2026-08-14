import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import os
import glob
import shutil
import networkx as nx
from pathlib import Path

# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------
# Blue for the reference group, red for the contrast. The three Quarto reports
# use the light pair below verbatim (COL_BLUE_LIGHT / COL_RED_LIGHT); they used
# to use the base R names "blue" and "red", which are pure primaries and read
# as harsh next to these figures.
#
# Three shades of each rather than one: the figures below are drawn at
# different line densities, and a single saturated pair reads badly where forty
# curves overlap. The shades pair up, dark with dark and light with light, so
# any two figures placed side by side stay consistent.
#
# This replaces six hard-coded hex codes scattered across the file, which had
# already drifted: the second group was drawn in a dark red in one figure and
# in orange in two others.
COL_BLUE_DARK  = '#16336E'
COL_BLUE_MID   = '#1A3A7A'
COL_BLUE_LIGHT = '#1a5fa8'
COL_RED_DARK   = '#8B1A26'
COL_RED_MID    = '#A01828'
COL_RED_LIGHT  = '#C0392B'
ROOT = Path(__file__).resolve().parent
FIGURES_DIR = ROOT / 'figures'
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ─── helpers ──────────────────────────────────────────────────────────────────

def landscape_k1(bd, tseq):
    if len(bd) == 0:
        return np.zeros(len(tseq))
    b, d = bd[:, 0], bd[:, 1]
    tents = np.maximum(0, np.minimum(tseq[:, None] - b[None, :], d[None, :] - tseq[:, None]))
    return tents.max(axis=1)

def load_h1(path):
    data = np.loadtxt(path, delimiter=',')
    if data.ndim == 1:
        data = data[None, :]
    h1 = data[(data[:, 2] == 1) & (data[:, 1] < np.inf)]
    return h1[:, :2]

def zero_safe_ylim(peak, fallback=0.05):
    """(0, peak*1.10), or a small fixed fallback range when peak is exactly
    0 -- after the Part B burn-in fix, every stable landscape is identically
    zero, so set_ylim(0, 0) would otherwise collapse the axis to nothing."""
    return (0, peak * 1.10) if peak > 0 else (0, fallback)

# ─── Figure 1: grn_17_subjects.png ────────────────────────────────────────────

ph_dir = ROOT / 'funTDA' / 'output' / 'PH'
asymp_files = sorted(glob.glob(str(ph_dir / 'Asymp*_PH.csv')))
symp_files  = sorted(glob.glob(str(ph_dir / 'Symp*_PH.csv')))

asymp_h1 = [load_h1(f) for f in asymp_files]
symp_h1  = [load_h1(f) for f in symp_files]

all_h1 = np.concatenate(asymp_h1 + symp_h1)
t_max  = all_h1[:, 1].max() * 1.05
tseq   = np.linspace(0, t_max, 500)

asymp_ls = [landscape_k1(h1, tseq) for h1 in asymp_h1]
symp_ls  = [landscape_k1(h1, tseq) for h1 in symp_h1]

fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)

for ls in asymp_ls:
    axes[0].plot(tseq, ls, color=COL_BLUE_MID, alpha=0.55, linewidth=0.9)
axes[0].set_title(f'Asymptomatic (n={len(asymp_ls)})')
axes[0].set_xlabel('Filtration value')
axes[0].set_ylabel('λ₁(t)')
axes[0].grid(True, alpha=0.3)

for ls in symp_ls:
    axes[1].plot(tseq, ls, color=COL_RED_MID, alpha=0.55, linewidth=0.9)
axes[1].set_title(f'Symptomatic (n={len(symp_ls)})')
axes[1].set_xlabel('Filtration value')
axes[1].grid(True, alpha=0.3)

fig.suptitle('H1 persistence landscapes of 17 H3N2 gene regulatory networks', fontsize=13)
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'grn_17_subjects.png', dpi=200, bbox_inches='tight')
plt.close()
print('figures/grn_17_subjects.png saved')

# ─── Figure 2: network_to_landscape.png ───────────────────────────────────────

adj  = np.loadtxt(ROOT / 'funTDA' / 'data' / 'GRN_adjacency_matrices' / 'Symptomaticadjmatrix1.csv', delimiter=',')
rng  = np.random.default_rng(0)
nodes = rng.choice(adj.shape[0], size=60, replace=False)
sub  = adj[np.ix_(nodes, nodes)]

upper = sub[np.triu_indices(60, k=1)]
thr   = np.percentile(upper, 85)

G = nx.Graph()
G.add_nodes_from(range(60))
for i in range(60):
    for j in range(i + 1, 60):
        if sub[i, j] > thr:
            G.add_edge(i, j, weight=sub[i, j])

pos = nx.circular_layout(G)
ew  = np.array([G[u][v]['weight'] for u, v in G.edges()])
ew_norm = (ew - ew.min()) / (ew.max() - ew.min() + 1e-9)

h1_s1   = load_h1(ROOT / 'funTDA' / 'output' / 'PH' / 'Symptomaticadjmatrix1_PH.csv')
t_max_s = h1_s1[:, 1].max() * 1.05 if len(h1_s1) else 1.0
tseq_s  = np.linspace(0, t_max_s, 500)
ls_s1   = landscape_k1(h1_s1, tseq_s)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

nx.draw_networkx_nodes(G, pos, ax=axes[0], node_size=25, node_color=COL_BLUE_MID, alpha=0.8)
nx.draw_networkx_edges(G, pos, ax=axes[0], alpha=0.3,
                       width=[0.4 + 1.2 * w for w in ew_norm], edge_color=COL_BLUE_MID)
axes[0].set_title('H3N2 GRN — Symptomatic 1\n60-node subgraph, edges > 85th pct', fontsize=10)
axes[0].axis('off')

axes[1].fill_between(tseq_s, ls_s1, alpha=0.15, color=COL_RED_MID)
axes[1].plot(tseq_s, ls_s1, color=COL_RED_MID, linewidth=2)
axes[1].set_xlabel('Filtration value')
axes[1].set_ylabel('λ₁(t)')
axes[1].set_title('H1 persistence landscape\n(Symptomatic 1)', fontsize=10)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(FIGURES_DIR / 'network_to_landscape.png', dpi=200, bbox_inches='tight')
plt.close()
print('figures/network_to_landscape.png saved')

# ─── Figure 3: timeseries_to_landscape.png ────────────────────────────────────

def simulate_beetles(mu_a, n=200, B=7.48, c_ea=0.009, c_pa=0.004, c_el=0.012, mu_l=0.267):
    L, P, A = 10.0, 5.0, 50.0
    series = [A]
    for _ in range(n - 1):
        L_new = B * A * np.exp(-c_el * L - c_ea * A)
        P_new = L * (1 - mu_l)
        A_new = P * np.exp(-c_pa * A) + A * (1 - mu_a)
        L, P, A = L_new, P_new, A_new
        series.append(A)
    return np.array(series)

stable_ts    = simulate_beetles(0.73)
aperiodic_ts = simulate_beetles(0.96)

# Read from the committed output/ CSVs (the burn-in run, same source
# results_partB.qmd treats as the single source of truth) rather than the
# .npy arrays directly, so this script can't silently drift from the report.
ts_output_dir    = ROOT / 'time-series' / 'output'
stable_ls_all    = np.loadtxt(ts_output_dir / 'stable_landscapes.csv', delimiter=',')
aperiodic_ls_all = np.loadtxt(ts_output_dir / 'aperiodic_landscapes.csv', delimiter=',')
tseq_ts          = np.loadtxt(ts_output_dir / 'tseq.csv', delimiter=',')

fig, axes = plt.subplots(2, 2, figsize=(12, 7))

t_ax = np.arange(80)
axes[0, 0].plot(t_ax, stable_ts[:80], color=COL_BLUE_LIGHT, linewidth=1.8)
axes[0, 0].set_title('Stable regime (μ_a = 0.73)')
axes[0, 0].set_xlabel('Time step')
axes[0, 0].set_ylabel('Adult population (A)')
axes[0, 0].grid(True, alpha=0.3)

axes[1, 0].plot(t_ax, aperiodic_ts[:80], color=COL_RED_LIGHT, linewidth=1.8)
axes[1, 0].set_title('Aperiodic regime (μ_a = 0.96)')
axes[1, 0].set_xlabel('Time step')
axes[1, 0].set_ylabel('Adult population (A)')
axes[1, 0].grid(True, alpha=0.3)

# Landscapes are plotted on INDEPENDENT y-scales: the stable regime's peak
# is always far smaller than the aperiodic one (whether it samples to
# exactly zero on the grid, as with a burn-in, or to a small transient
# loop, as in the heritage no-burn-in run), so a shared axis would flatten
# it to nothing. zero_safe_ylim falls back to a small fixed range only in
# the exactly-zero case; the text label reports whichever is true.
# Colours are the light pair of the palette above, shared with the three Quarto
# reports so that the matplotlib and ggplot figures of this project match.
for i in range(40):
    axes[0, 1].plot(tseq_ts, stable_ls_all[i], color=COL_BLUE_LIGHT, alpha=0.55, linewidth=1.1)
stable_peak_40 = stable_ls_all[:40].max()
axes[0, 1].set_title('H1 landscapes — Stable' + ('  (identically zero)' if stable_peak_40 == 0 else ''))
axes[0, 1].set_xlabel('Filtration value')
axes[0, 1].set_ylabel('λ₁(t)')
axes[0, 1].set_ylim(*zero_safe_ylim(stable_peak_40))
axes[0, 1].grid(True, alpha=0.3)
stable_label_40 = 'identically zero' if stable_peak_40 == 0 else f'peak ≈ {stable_peak_40:.2f}'
axes[0, 1].text(0.97, 0.92, stable_label_40,
                transform=axes[0, 1].transAxes, ha='right', va='top',
                fontsize=9, color=COL_BLUE_LIGHT)

for i in range(40):
    axes[1, 1].plot(tseq_ts, aperiodic_ls_all[i], color=COL_RED_LIGHT, alpha=0.55, linewidth=1.1)
axes[1, 1].set_title('H1 landscapes — Aperiodic')
axes[1, 1].set_xlabel('Filtration value')
axes[1, 1].set_ylabel('λ₁(t)')
axes[1, 1].set_ylim(0, aperiodic_ls_all[:40].max() * 1.10)
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].text(0.97, 0.92, f'peak ≈ {aperiodic_ls_all[:40].max():.2f}',
                transform=axes[1, 1].transAxes, ha='right', va='top',
                fontsize=9, color=COL_RED_LIGHT)

plt.tight_layout()
plt.savefig(FIGURES_DIR / 'timeseries_to_landscape.png', dpi=200, bbox_inches='tight')
plt.close()
print('figures/timeseries_to_landscape.png saved')

# ─── Figure 3b: timeseries_regimes.png (raw series only, for the memoria's Data
#     section; the landscape half of the old 2x2 is already covered there by
#     landscapes_overlaid.png in Results) ──────────────────────────────────────
# Same example series, same colors and axis labels as the left column of the
# 2x2 above, so a reader comparing the two recognises the same trajectory.
# Full y-axis range (no clipping): the stable series' decaying oscillation
# before it settles is exactly what the text discusses, so it must stay visible.
fig, axes = plt.subplots(2, 1, figsize=(7, 6))

axes[0].plot(t_ax, stable_ts[:80], color=COL_BLUE_LIGHT, linewidth=1.8)
axes[0].set_title('Stable regime (μ_a = 0.73)')
axes[0].set_xlabel('Time step')
axes[0].set_ylabel('Adult population (A)')
axes[0].grid(True, alpha=0.3)

axes[1].plot(t_ax, aperiodic_ts[:80], color=COL_RED_LIGHT, linewidth=1.8)
axes[1].set_title('Aperiodic regime (μ_a = 0.96)')
axes[1].set_xlabel('Time step')
axes[1].set_ylabel('Adult population (A)')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(FIGURES_DIR / 'timeseries_regimes.png', dpi=200, bbox_inches='tight')
plt.close()
print('figures/timeseries_regimes.png saved')

# ─── Figure 4: landscapes_individual_dark.png ─────────────────────────────────

# Two panels with INDEPENDENT y-scales: the stable regime's peak is always
# far smaller than the aperiodic one, whether it samples to exactly zero on
# the grid (a burn-in run) or to a small transient loop (the heritage
# no-burn-in run), so a shared axis would flatten it out. zero_safe_ylim
# falls back to a small fixed range only in the exactly-zero case.
fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))

for i in range(50):
    axes[0].plot(tseq_ts, stable_ls_all[i], color=COL_BLUE_DARK, alpha=0.6, linewidth=1.2)
axes[0].set_title('Stable', color=COL_BLUE_DARK, fontsize=13, fontweight='bold')
axes[0].set_xlabel('Filtration value')
axes[0].set_ylabel('λ₁(t)')
stable_peak_50 = stable_ls_all[:50].max()
axes[0].set_ylim(*zero_safe_ylim(stable_peak_50))
axes[0].grid(True, alpha=0.3)
stable_label_50 = 'identically zero' if stable_peak_50 == 0 else f'peak ≈ {stable_peak_50:.2f}'
axes[0].text(0.97, 0.92, stable_label_50,
             transform=axes[0].transAxes, ha='right', va='top', fontsize=10, color=COL_BLUE_DARK)

for i in range(50):
    axes[1].plot(tseq_ts, aperiodic_ls_all[i], color=COL_RED_DARK, alpha=0.6, linewidth=1.2)
axes[1].set_title('Aperiodic', color=COL_RED_DARK, fontsize=13, fontweight='bold')
axes[1].set_xlabel('Filtration value')
axes[1].set_ylabel('λ₁(t)')
axes[1].set_ylim(0, aperiodic_ls_all[:50].max() * 1.10)
axes[1].grid(True, alpha=0.3)
axes[1].text(0.97, 0.92, f'peak ≈ {aperiodic_ls_all[:50].max():.2f}',
             transform=axes[1].transAxes, ha='right', va='top', fontsize=10, color=COL_RED_DARK)

fig.suptitle(
    'H1 persistence landscapes — the stable regime is identically zero\n'
    '(its attractor is a fixed point with no H1 loop; see results_partB.qmd)'
    if stable_peak_50 == 0 else
    'H1 persistence landscapes — the stable regime carries only a small transient loop\n'
    '(the approach to the fixed point, not the attractor itself; see results_partB.qmd)',
    fontsize=12)

plt.tight_layout()
plt.savefig(FIGURES_DIR / 'landscapes_individual_dark.png', dpi=200, bbox_inches='tight')
plt.close()
print('figures/landscapes_individual_dark.png saved')

# ─── Figure 4b: landscapes_overlaid.png (both regimes, same axis) ──────────────
# Colours are the light pair of the palette above, shared with the three Quarto
# reports so that the matplotlib and ggplot figures of this project match.
fig, ax = plt.subplots(figsize=(10, 4.2))
for i in range(50):
    ax.plot(tseq_ts, aperiodic_ls_all[i], color=COL_RED_LIGHT, alpha=0.45, linewidth=1.1, zorder=1)
for i in range(50):
    ax.plot(tseq_ts, stable_ls_all[i], color=COL_BLUE_LIGHT, alpha=0.85, linewidth=1.3, zorder=2)
handles = [
    Line2D([0], [0], color=COL_BLUE_LIGHT, linewidth=2.5, label='Stable'),
    Line2D([0], [0], color=COL_RED_LIGHT, linewidth=2.5, label='Aperiodic'),
]
ax.legend(handles=handles, fontsize=12, loc='upper right')
ax.set_xlabel('Filtration value')
ax.set_ylabel('λ₁(t)')
ax.set_title('H1 persistence landscapes — Stable vs Aperiodic (same axis)')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'landscapes_overlaid.png', dpi=200, bbox_inches='tight')
plt.close()
print('figures/landscapes_overlaid.png saved')

# ─── Figure 5: delay_embedding.png ────────────────────────────────────────────
# The reconstructed attractor itself, which is the one step of the chain the
# other figures skip: timeseries_to_landscape.png goes straight from the series
# to the landscape. Chapter 4 claims that the SHAPE of the delay cloud is what
# separates the regimes -- a point for an equilibrium, a closed curve for a
# sustained oscillation -- and until now the thesis asserted that without
# showing it.
#
# Same m = 2, tau = 3 the chapter fixes, and the same min-max normalisation
# applied before the filtration, so this is the cloud persistent homology
# actually sees rather than a schematic of one.
TAU, BURN = 3, 25

fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
for ax, (mu, col, dark, name) in zip(axes, [
        (0.73, COL_BLUE_LIGHT, COL_BLUE_DARK, 'Stable'),
        (0.96, COL_RED_LIGHT,  COL_RED_DARK,  'Aperiodic')]):
    x = simulate_beetles(mu)
    x = (x - x.min()) / (x.max() - x.min())
    X, Y = x[:-TAU], x[TAU:]
    ax.plot(X, Y, color=col, linewidth=0.6, alpha=0.45, zorder=1)
    # The transient is drawn apart because Section 4.5.2 attributes the surviving
    # stable-regime loops to it, not to the attractor.
    ax.scatter(X[BURN:], Y[BURN:], s=11, color=col,  zorder=3, edgecolors='none',
               label='after step 25')
    ax.scatter(X[:BURN], Y[:BURN], s=11, color=dark, zorder=2, edgecolors='none',
               alpha=0.85, label='first 25 steps')
    ax.set_title(f'{name} regime (μ_a = {mu})')
    ax.set_xlabel('$x_t$')
    ax.set_ylabel(r'$x_{t+3}$')
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc='upper right', framealpha=0.9)

plt.tight_layout()
plt.savefig(FIGURES_DIR / 'delay_embedding.png', dpi=200, bbox_inches='tight')
plt.close()
print('figures/delay_embedding.png saved')

# ─── Distribute the figures to everything that consumes them ──────────────────

# Two consumers, and both used to be fed by hand.
#
# results_partB.qmd embeds two of these PNGs by bare filename, so Quarto
# resolves them next to the .qmd rather than here. Copying them across after
# each regeneration was a manual step nobody wrote down, and the copies in
# time-series/ duly froze on 30 July while the originals moved on. Same failure
# as the memoria's, which had already frozen once when TFM_UCD/figures/ was
# renamed to ResearchProject_UCD/figures/.
#
# So: every consumer is listed here, and the copy is part of the run. If you
# add a figure to a report, add it to this table -- do not copy it by hand.
targets = [
    # (source dir, destination, required, [(source name, name at destination), ...])
    (FIGURES_DIR, ROOT / 'time-series', True, [
        ('timeseries_to_landscape.png', 'timeseries_to_landscape.png'),
        ('landscapes_overlaid.png',     'landscapes_overlaid.png'),
    ]),
    (FIGURES_DIR, ROOT.parent / 'ResearchProject_UCD' / 'figures', False, [
        ('timeseries_regimes.png',  'ts_timeseries_regimes.png'),
        ('landscapes_overlaid.png', 'ts_landscapes_overlaid.png'),
        ('delay_embedding.png',     'ts_delay_embedding.png'),
    ]),
    # Part C's anatomy figure is the one image in the thesis that this script
    # does not draw. It is rendered by images/make_brain_figure.py from DICOM
    # that the repository cannot carry, so it is committed under
    # images/figures/ and only copied here. It still goes through this table
    # rather than by hand, for the reason the comment above gives.
    (ROOT / 'images' / 'figures',
     ROOT.parent / 'ResearchProject_UCD' / 'figures', False, [
        ('ventricle_example.png', 'img_ventricles.png'),
    ]),
]

for src_dir, dest, required, pairs in targets:
    if not dest.is_dir():
        # The memoria lives outside this repository, so a standalone clone must
        # still run this script cleanly; time-series/ is part of the repository
        # and its absence is a real error.
        if required:
            raise SystemExit(f'{dest} is missing: this is part of the repository')
        print(f'{dest} not found; skipping export')
        continue
    for src_name, dst_name in pairs:
        shutil.copyfile(src_dir / src_name, dest / dst_name)
        print(f'copied {src_name} -> {dest / dst_name}')
