# Audit: similarity vs. distance semantics in `GraphGeodesicDistance` (Part A)

Investigation only. No code, notebook outputs, or `output/PH/` CSVs were changed.

## The question

`funTDA/analysis/funTDA.ipynb` calls:

```python
X_ggd = GraphGeodesicDistance(directed=False, unweighted=False).fit_transform([adj])
```

directly on the raw values loaded from `funTDA/data/GRN_adjacency_matrices/*.csv`
(`mat = genfromtxt(...)`, no transform in between). `GraphGeodesicDistance` with
`unweighted=False` uses the matrix entries as edge *lengths* (Dijkstra shortest path,
summing weights along the path). If a high entry means "strongly connected" rather
than "far apart," this inverts the intended metric: the most-correlated gene pairs
would be treated as topologically the *most distant*.

## What the paper (Higgins, Wu & Carey) actually specifies

Two passages are directly relevant, and neither describes an inversion step:

1. **Section 2.1, p.5** (general method): "we first compute the geodesic distance
   matrix Γg = {γij}, such that the element γij is the length of the shortest
   directed path from vertex i → j... **For weighted graphs, this corresponds to the
   path minimizing the sum of edge weights**." — the weights themselves, whatever they
   are, are used directly as path-length components. No `1 - w` or `1/w` is
   mentioned here or anywhere else in the methodology section.

2. **Section 5, p.17** (the actual GRN application): "we constructed a gene
   regulatory network (GRN) by computing pairwise Spearman rank correlations among
   the temporal expression profiles of these 250 genes. This procedure yielded 17
   correlation matrices, each of size 250 × 250, **which serve as the weighted
   adjacency matrices** for the respective GRNs." — the correlation matrices are
   said to serve *directly* as the adjacency matrices. No conversion step is
   described.

Appendix E (normalization) only rescales by the matrix maximum for the *literary*
networks and the Quotient-space method ("adjacency matrices were normalized by their
maximum entry, rescaling all edge weights to the interval [0, 1]") — a pure linear
rescale that preserves order and would not flip a similarity into a distance. It is
not stated to apply to the GRN analysis, and no other transform is mentioned there
either.

**Conclusion from the paper text alone: `funTDA.ipynb` is a literal, faithful
implementation of what the paper describes.** If the semantics are backwards, that
would be a property of the *published method itself*, not a deviation this repo
introduced.

## What `funTDA.ipynb` does, confirmed by reading the code

`compute_persistence()` (cell 1 of `funTDA/analysis/funTDA.ipynb`):

```python
def compute_persistence(adjacency_matrix):
    G = nx.convert.to_networkx_graph(adjacency_matrix)
    adj = nx.adjacency_matrix(G)
    X_ggd = GraphGeodesicDistance(directed=False, unweighted=False).fit_transform([adj])
    PH = FlagserPersistence(directed=False).fit_transform(X_ggd)
    return PH[0]
```

The CSV values go straight into `to_networkx_graph` → `GraphGeodesicDistance`, with no
intermediate transform anywhere in the notebook. This matches the "no transform
described" reading of the paper above.

## What the actual data looks like

`funTDA/data/GRN_adjacency_matrices/*.csv` (17 files, 250×250 each):

- Range: **exactly [0, 2]** across the whole collection (per-file max ranges 1.94–2.0).
- Symmetric, diagonal exactly 0.
- Dense: only ~0.1% of off-diagonal entries are exactly 0 — essentially a fully
  connected weighted graph, not a sparse "edge present/absent" adjacency matrix.
- Distribution skews toward the higher half of [0, 2] (mean ≈ 1.11, median ≈ 1.20).

There is no script anywhere in this repo that generates these CSVs from raw
expression data or documents how they were derived — they were committed as-is in
the initial commit (`git log --follow` confirms this; `5e2e384`). The paper itself
only links to a private Google Drive folder for code/data ("available privately...
until publication"), which isn't accessible here.

The **[0, 2] range is the key clue, and it cuts both ways.** Spearman correlation
lives in [-1, 1]. Two simple positivity-preserving transforms both land exactly on
[0, 2] with the diagonal at 0 (since r=1 for a gene against itself):

- **`d = 1 - r`** — genuine distance semantics. r = 1 (perfect co-expression) → d = 0
  ("close"). r = -1 → d = 2. Under this reading, `funTDA.ipynb`'s direct use of the
  matrix as edge length is **correct**, and there is no bug.
- **`w = 1 + r`** — similarity semantics preserved, just shifted positive (needed
  because Dijkstra-based `GraphGeodesicDistance` requires non-negative weights, and
  raw Spearman correlation can be negative). r = 1 → w = 2 ("strong"). r = -1 → w = 0.
  Under this reading, using `w` directly as a distance **is backwards**, exactly the
  concern raised: the most co-expressed genes would be treated as farthest apart.

Both transforms are standard in different contexts — `1 - r` is the conventional
correlation-to-distance conversion in clustering/co-expression-network literature;
`1 + r` (or similar) is a natural "just make it non-negative" fix if the intent was
to keep correlation-as-strength semantics, which is also what the paper's own
language on p.2 suggests ("edge values reflecting the strength of interactions") and
what Section 5 literally says ("correlation matrices... serve as the weighted
adjacency matrices," i.e., unchanged).

I could not find a way to distinguish the two from the data alone. A biological
argument (whether co-regulated genes in an immune response skew positively or
negatively correlated) is too weak to be dispositive — a mix of up- and
down-regulated gene clusters can produce either sign on average — so I'm not relying
on it.

## Bottom line

- `funTDA.ipynb`'s code is consistent with the paper's stated methodology: use the
  adjacency matrix values directly as edge weights for geodesic distance, no
  transform. This repo does not deviate from the paper here.
- Whether that is *actually* correct for the H3N2 GRN application hinges entirely on
  a transform (`1 - r` vs. `1 + r`, or something else) applied **before** the CSVs in
  `funTDA/data/GRN_adjacency_matrices/` were generated — a step this repo has no
  record of and doesn't perform itself.
- I cannot resolve which transform was used from the evidence available in this
  repo. Resolving it would require either the original authors' code (the paper
  links a private Google Drive folder, not accessible here) or independently
  recomputing the Spearman correlations from the raw GSE30550 expression data and
  checking which transform reproduces these exact committed CSVs.
- Given `results_partA.qmd` validates its numbers bit-for-bit against this exact
  pipeline and data, **do not change anything here** without first resolving the
  transform question above — flipping the semantics would change every H1 birth/death
  pair, every CSV in `output/PH/`, and invalidate both the cached permutation test
  and the report's `stopifnot()` baselines.
