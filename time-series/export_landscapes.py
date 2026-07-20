"""
Export the H1 persistence landscape matrices computed in
`beetles_landscapes.ipynb` (stable_landscapes.npy, aperiodic_landscapes.npy,
tseq.npy) to plain CSV, so that `results_partB.qmd` (R / knitr) can read them
without re-simulating or re-running persistent homology.

This mirrors Part A's split: Python (ripser) computes persistent homology and
landscapes -> CSV; R (fda) does the functional data analysis. Run this after
`beetles_landscapes.ipynb`, whenever the landscape matrices change.
"""

import numpy as np
import os

here = os.path.dirname(os.path.abspath(__file__))
out_dir = os.path.join(here, "output")
os.makedirs(out_dir, exist_ok=True)

stable = np.load(os.path.join(here, "stable_landscapes.npy"))       # (200, 500)
aperiodic = np.load(os.path.join(here, "aperiodic_landscapes.npy")) # (200, 500)
tseq = np.load(os.path.join(here, "tseq.npy"))                      # (500,)

np.savetxt(os.path.join(out_dir, "stable_landscapes.csv"), stable, delimiter=",")
np.savetxt(os.path.join(out_dir, "aperiodic_landscapes.csv"), aperiodic, delimiter=",")
np.savetxt(os.path.join(out_dir, "tseq.csv"), tseq, delimiter=",")

print(f"Wrote {out_dir}/stable_landscapes.csv    {stable.shape}")
print(f"Wrote {out_dir}/aperiodic_landscapes.csv  {aperiodic.shape}")
print(f"Wrote {out_dir}/tseq.csv                  {tseq.shape}")
