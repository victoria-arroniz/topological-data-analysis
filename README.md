# TDA Research: Networks and Time Series

Topological data analysis (TDA) applied to two data modalities:

- **Part A — funTDA (Networks):** statistical analysis of gene regulatory networks using persistent homology + functional data analysis.
- **Part B — Time Series:** persistence landscape analysis of simulated Tribolium beetle population dynamics.

---

## Repository structure

```
topological-data-analysis/
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
│   ├── Example1Beetles.Rmd          # Original R Markdown code (simulation + k-means/confusion-matrix check), kept as reference
│   ├── beetles_landscapes.ipynb     # Python: simulation, embedding, PH, landscapes
│   ├── export_landscapes.py         # Exports the notebook's .npy landscapes to output/*.csv
│   ├── output/                      # stable/aperiodic_landscapes.csv (200x500), tseq.csv
│   └── results_partB.qmd            # R/knitr: FPCA + permutation test (reads output/*.csv)
│
├── references/
│   ├── ***REMOVED***                                        # funTDA paper
│   ├── ***REMOVED***
│   └── ***REMOVED***                                           # Pereira & de Mello (2015)
│
├── requirements.txt
├── requirements-partA.txt           # Pinned deps for funTDA.ipynb (giotto-tda/numpy), kept separate — see requirements.txt
└── README.md
```

---

## Part A — funTDA

**Data:** 17 H3N2 influenza gene regulatory networks (8 asymptomatic, 9 symptomatic subjects). Each network is a 250×250 weighted adjacency matrix.

**Pipeline:**

1. **`funTDA.ipynb` (Python):** converts each adjacency matrix to a graph, computes graph geodesic distances, and runs Flagser persistent homology (H0 and H1) via `giotto-tda`. Outputs are stored as `[birth, death, dimension]` CSV files in `output/PH/`.

2. **`funTDA.R` (R):** reads the birth–death CSVs, computes H1 persistence landscapes on a 500-point grid `tseq = seq(0, 1, length.out = 500)`, builds functional data objects (B-splines), runs functional PCA, and applies permutation-based hypothesis tests (`tperm.fd`) to compare symptomatic vs. asymptomatic groups.

**Reproducible report:** [`analysis/results_partA.qmd`](funTDA/analysis/results_partA.qmd) (rendered: [`results_partA.pdf`](funTDA/analysis/results_partA.pdf), [`results_partA.html`](funTDA/analysis/results_partA.html)) reruns this pipeline end-to-end on the committed CSVs — FPCA, the asymptomatic-vs-symptomatic permutation test, and a 196-test split-half calibration check — with `stopifnot()` assertions against the reported baseline numbers. It reimplements `TDA::landscape(KK=1)` and a vectorized equivalent of `tperm.fd` in plain R (both validated bit-for-bit against the original functions) so the split-half loop renders in seconds instead of ~2 hours; results are cached in `output/results_partA_permtest_cache.rds`. The PDF is the one to open on GitHub (GitHub previews PDFs inline but only shows raw source for `.html` files); the HTML is self-contained (`embed-resources: true`) for opening locally in a browser.

To render it yourself:
```
quarto render funTDA/analysis/results_partA.qmd
```
Renders both the PDF and HTML. Requires [Quarto](https://quarto.org), a LaTeX distribution with `xelatex` (e.g. [TinyTeX](https://quarto.org/docs/output-formats/pdf-basics.html)) for the PDF, and R with packages `fda`, `ggplot2`, `ggrepel`, `patchwork`, `igraph`, `knitr` (see the document's appendix for exact versions via `sessionInfo()`). Delete `output/results_partA_permtest_cache.rds` to force the permutation tests to recompute from scratch.

**Dependencies (Part A):**
```
pip install -r requirements-partA.txt
```
Kept separate from `requirements.txt` because giotto-tda is sensitive to the installed numpy version.
R packages: `TDA`, `fda`, `ggplot2`, `ggrepel`, `patchwork`, `igraph` (the last one only for the network-comparison figure in `results_partA.qmd`)

---

## Part B — Time Series

**Data:** simulated Tribolium flour beetle population dynamics (Costantino et al., 1995, via Pereira & de Mello, 2015). Two dynamical regimes:
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

**Reproducible report:** [`results_partB.qmd`](time-series/results_partB.qmd) (rendered: [`results_partB.pdf`](time-series/results_partB.pdf), [`results_partB.html`](time-series/results_partB.html)) picks up where the notebook stops and does the part it explicitly leaves as a "next step": FPCA and a functional permutation test comparing the two regimes, following the same `fda`-based pipeline as Part A. Unlike Part A, this isn't a reproduction of a previously reported number — neither the funTDA paper nor the original `Example1Beetles.Rmd` (which uses k-means + a per-series classification rule, not a functional hypothesis test) runs this analysis, so the document says so explicitly rather than presenting it as a known baseline. [`export_landscapes.py`](time-series/export_landscapes.py) converts the notebook's `.npy` landscape matrices to the CSVs (`time-series/output/`) the report reads; the report itself does not re-simulate or re-run persistent homology (slow, and Python/R RNGs aren't interchangeable anyway).

**Dependencies (Part B):**
```
pip install -r requirements.txt
```
R packages for `results_partB.qmd`: `fda`, `ggplot2`, `patchwork`, `knitr`. Render with `quarto render time-series/results_partB.qmd` (requires [Quarto](https://quarto.org) and `xelatex` for the PDF); delete `time-series/output/results_partB_permtest_cache.rds` to force the permutation tests to recompute (the control-analysis step alone runs 400 permutation tests of 10,000 permutations each, so expect several minutes on a cold cache).

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

- Higgins C, Wu H, Carey M. *Statistical Analysis of Network Collections Using Persistent Homology and Functional Data Analysis.*
- Berry E, Chen Y, Cisewski-Kehe J, Fasy BT. *Functional Summaries of Persistence Diagrams.*
- Pereira CMM, de Mello RF. *Persistent homology for time series and spatial data clustering.* Expert Systems with Applications, 42 (2015), 6026–6038. — basis of the time-series code (Part B).
- Costantino RF, Cushing JM, Dennis B, Desharnais RA. *Experimentally induced transitions in the dynamic behaviour of insect populations.* Nature, 375(6528) (1995), 227–230. — original Tribolium (L, P, A) model, cited here via Pereira & de Mello (2015).
