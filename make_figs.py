import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import os
import glob
import networkx as nx

os.makedirs('figures', exist_ok=True)

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

# ─── Figure 1: grn_17_subjects.png ────────────────────────────────────────────

ph_dir = 'funTDA/output/PH'
asymp_files = sorted(glob.glob(f'{ph_dir}/Asymp*_PH.csv'))
symp_files  = sorted(glob.glob(f'{ph_dir}/Symp*_PH.csv'))

asymp_h1 = [load_h1(f) for f in asymp_files]
symp_h1  = [load_h1(f) for f in symp_files]

all_h1 = np.concatenate(asymp_h1 + symp_h1)
t_max  = all_h1[:, 1].max() * 1.05
tseq   = np.linspace(0, t_max, 500)

asymp_ls = [landscape_k1(h1, tseq) for h1 in asymp_h1]
symp_ls  = [landscape_k1(h1, tseq) for h1 in symp_h1]

mean_asymp = np.mean(asymp_ls, axis=0)
mean_symp  = np.mean(symp_ls,  axis=0)

fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)

for ls in asymp_ls:
    axes[0].plot(tseq, ls, color='#1A3A7A', alpha=0.55, linewidth=0.8)
axes[0].plot(tseq, mean_asymp, color='#1A3A7A', linewidth=2.5, label='Asymptomatic (n=8)')
axes[0].set_title('Asymptomatic')
axes[0].set_xlabel('Filtration value')
axes[0].set_ylabel('λ₁(t)')
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3)

for ls in symp_ls:
    axes[1].plot(tseq, ls, color='#A01828', alpha=0.55, linewidth=0.8)
axes[1].plot(tseq, mean_symp, color='#A01828', linewidth=2.5, label='Symptomatic (n=9)')
axes[1].set_title('Symptomatic')
axes[1].set_xlabel('Filtration value')
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3)

fig.suptitle('H1 persistence landscapes of 17 H3N2 gene regulatory networks', fontsize=13)
plt.tight_layout()
plt.savefig('figures/grn_17_subjects.png', dpi=200, bbox_inches='tight')
plt.close()
print('figures/grn_17_subjects.png saved')

# ─── Figure 2: network_to_landscape.png ───────────────────────────────────────

adj  = np.loadtxt('funTDA/data/GRN_adjacency_matrices/Symptomaticadjmatrix1.csv', delimiter=',')
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

h1_s1   = load_h1('funTDA/output/PH/Symptomaticadjmatrix1_PH.csv')
t_max_s = h1_s1[:, 1].max() * 1.05 if len(h1_s1) else 1.0
tseq_s  = np.linspace(0, t_max_s, 500)
ls_s1   = landscape_k1(h1_s1, tseq_s)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

nx.draw_networkx_nodes(G, pos, ax=axes[0], node_size=25, node_color='#1A3A7A', alpha=0.8)
nx.draw_networkx_edges(G, pos, ax=axes[0], alpha=0.3,
                       width=[0.4 + 1.2 * w for w in ew_norm], edge_color='#1A3A7A')
axes[0].set_title('H3N2 GRN — Symptomatic 1\n60-node subgraph, edges > 85th pct', fontsize=10)
axes[0].axis('off')

axes[1].fill_between(tseq_s, ls_s1, alpha=0.15, color='#A01828')
axes[1].plot(tseq_s, ls_s1, color='#A01828', linewidth=2)
axes[1].set_xlabel('Filtration value')
axes[1].set_ylabel('λ₁(t)')
axes[1].set_title('H1 persistence landscape\n(Symptomatic 1)', fontsize=10)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('figures/network_to_landscape.png', dpi=200, bbox_inches='tight')
plt.close()
print('figures/network_to_landscape.png saved')

# ─── Figure 3: timeseries_to_landscape.png ────────────────────────────────────

def simulate_beetles(u_a, n=200, B=7.48, c_ea=0.009, c_pa=0.004, c_el=0.012, u_l=0.267):
    np.random.seed(123)
    L, P, A = 10.0, 5.0, 50.0
    series = [A]
    for _ in range(n - 1):
        L_new = B * A * np.exp(-c_el * L - c_ea * A)
        P_new = L * (1 - u_l)
        A_new = P * np.exp(-c_pa * A) + A * (1 - u_a)
        L, P, A = L_new, P_new, A_new
        series.append(A)
    return np.array(series)

stable_ts    = simulate_beetles(0.73)
aperiodic_ts = simulate_beetles(0.96)

stable_ls_all    = np.load('time-series/stable_landscapes.npy')
aperiodic_ls_all = np.load('time-series/aperiodic_landscapes.npy')
tseq_ts          = np.load('time-series/tseq.npy')

y_max = max(stable_ls_all[:40].max(), aperiodic_ls_all[:40].max()) * 1.05

fig, axes = plt.subplots(2, 2, figsize=(12, 7))

t_ax = np.arange(80)
axes[0, 0].plot(t_ax, stable_ts[:80], color='#1A3A7A', linewidth=1.5)
axes[0, 0].set_title('Stable regime (u_a = 0.73)')
axes[0, 0].set_xlabel('Time step')
axes[0, 0].set_ylabel('Adult population (A)')
axes[0, 0].grid(True, alpha=0.3)

axes[1, 0].plot(t_ax, aperiodic_ts[:80], color='#C85A12', linewidth=1.5)
axes[1, 0].set_title('Aperiodic regime (u_a = 0.96)')
axes[1, 0].set_xlabel('Time step')
axes[1, 0].set_ylabel('Adult population (A)')
axes[1, 0].grid(True, alpha=0.3)

for i in range(40):
    axes[0, 1].plot(tseq_ts, stable_ls_all[i], color='#1A3A7A', alpha=0.35, linewidth=0.8)
axes[0, 1].set_title('H1 landscapes — Stable')
axes[0, 1].set_xlabel('Filtration value')
axes[0, 1].set_ylabel('λ₁(t)')
axes[0, 1].set_ylim(0, y_max)
axes[0, 1].grid(True, alpha=0.3)

for i in range(40):
    axes[1, 1].plot(tseq_ts, aperiodic_ls_all[i], color='#C85A12', alpha=0.35, linewidth=0.8)
axes[1, 1].set_title('H1 landscapes — Aperiodic')
axes[1, 1].set_xlabel('Filtration value')
axes[1, 1].set_ylabel('λ₁(t)')
axes[1, 1].set_ylim(0, y_max)
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('figures/timeseries_to_landscape.png', dpi=200, bbox_inches='tight')
plt.close()
print('figures/timeseries_to_landscape.png saved')

# ─── Figure 4: landscapes_individual_dark.png ─────────────────────────────────

fig, ax = plt.subplots(figsize=(10, 4))

for i in range(50):
    ax.plot(tseq_ts, stable_ls_all[i],    color='#1A3A7A', alpha=0.5, linewidth=0.8)
for i in range(50):
    ax.plot(tseq_ts, aperiodic_ls_all[i], color='#C85A12', alpha=0.5, linewidth=0.8)

handles = [
    Line2D([0], [0], color='#1A3A7A', linewidth=2, label='Stable (regular)'),
    Line2D([0], [0], color='#C85A12', linewidth=2, label='Aperiodic (irregular)'),
]
ax.legend(handles=handles, fontsize=11)
ax.set_xlabel('Filtration value')
ax.set_ylabel('λ₁(t)')
ax.set_title('H1 persistence landscapes — Stable vs Aperiodic (50 series each)')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('figures/landscapes_individual_dark.png', dpi=200, bbox_inches='tight')
plt.close()
print('figures/landscapes_individual_dark.png saved')
