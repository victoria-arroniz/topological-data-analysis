# TDA Research: Networks and Time Series

Topological data analysis (TDA) applied to two data modalities:

- **Part A — funTDA (Networks):** statistical analysis of gene regulatory networks using persistent homology + functional data analysis.
- **Part B — Time Series:** persistence landscape analysis of simulated Tribolium beetle population dynamics.

---

## Repository structure

```
tda-research/
├── funTDA/                          # Part A: network TDA
│   ├── analysis/
│   │   ├── funTDA.ipynb             # Python: computes PH for each GRN
│   │   └── funTDA.R                 # R: landscapes, FPCA, permutation tests
│   ├── data/
│   │   └── GRN_adjacency_matrices/  # 17 adjacency matrices (250×250)
│   └── output/
│       └── PH/                      # Pre-computed birth–death pairs (CSV)
│
├── time-series/                     # Part B: time series TDA
│   ├── Example1Beetles.R            # Original R code (simulation + PH)
│   └── beetles_landscapes.ipynb     # Python: full pipeline incl. landscapes
│
├── references/
│   ├── ***REMOVED***                                        # funTDA paper
│   ├── ***REMOVED***
│   └── ***REMOVED***                                           # Pereira & de Mello (2015)
│
├── requirements.txt
└── README.md
```

---

## Part A — funTDA

**Data:** 17 H3N2 influenza gene regulatory networks (8 asymptomatic, 9 symptomatic subjects). Each network is a 250×250 weighted adjacency matrix.

**Pipeline:**

1. **`funTDA.ipynb` (Python):** converts each adjacency matrix to a graph, computes graph geodesic distances, and runs Flagser persistent homology (H0 and H1) via `giotto-tda`. Outputs are stored as `[birth, death, dimension]` CSV files in `output/PH/`.

2. **`funTDA.R` (R):** reads the birth–death CSVs, computes H1 persistence landscapes on a 500-point grid `tseq = seq(0, 1, 500)`, builds functional data objects (B-splines), runs functional PCA, and applies permutation-based hypothesis tests (`tperm.fd`) to compare symptomatic vs. asymptomatic groups.

**Dependencies (Part A):**
```
pip install giotto-tda networkx numpy
```
R packages: `TDA`, `fda`, `ggplot2`, `ggrepel`, `patchwork`

---

## Part B — Time Series

**Data:** simulated Tribolium flour beetle population dynamics (Costantino et al., 1995). Two dynamical regimes:
- **Stable** (`u_a = 0.73`): converges to a fixed-point attractor
- **Aperiodic** (`u_a = 0.96`): chaotic oscillations

**Pipeline (`beetles_landscapes.ipynb`):**

| Step | Implementation | Notes |
|------|---------------|-------|
| Simulate beetle ODE | Discrete difference equations | Matches R's `ode(..., method='iteration')` |
| Normalise to [0, 1] | Per-series min-max | Makes Rips filtration scale-invariant |
| Takens embedding | m=2, τ=3 → 198-point cloud in ℝ² | Matches R's `buildTakens(x, 2, 3)` |
| Persistent homology | Vietoris-Rips via `ripser` | Same Ripser backend as `TDAstats` in R |
| Persistence landscape | H1, first layer, 500 bins | Equivalent to `landscape(PH, dim=1, KK=1, tseq)` in R |

**Dependencies (Part B):**
```
pip install -r requirements.txt
```

---

## Methodological connection

Both parts apply the same funTDA workflow: raw data → persistent homology → persistence landscape → functional data analysis. The landscape representation converts each topological summary into a function in L², enabling statistical comparisons (mean, FPCA, hypothesis tests) across samples.

| Aspect | Part A (networks) | Part B (time series) |
|--------|-------------------|----------------------|
| Input | Adjacency matrix | Scalar time series |
| Geometry | Graph geodesic distances | Takens phase-space embedding |
| PH algorithm | Flagser | Vietoris-Rips |
| Relevant homology | H1 (graph cycles) | H1 (attractor loops) |
| Landscape grid | Fixed [0, 1], 500 pts | Data-adaptive range, 500 pts |

---

## References

- Higgins C, Wu H, Carey M. *Statistical Analysis of Network Collections Using Persistent Homology and Functional Data Analysis.* (funTDA paper)
- Berry E, Chen Y, Cisewski-Kehe J, Fasy BT. *Functional Summaries of Persistence Diagrams.*
- Pereira CMM, de Mello RF. *Persistent homology for time series and spatial data clustering.* Expert Systems with Applications, 2015.
- Costantino RF et al. *Experimentally induced transitions in the dynamic behaviour of insect populations.* Nature, 1995.
