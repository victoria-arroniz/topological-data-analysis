"""
Export the H1 persistence landscape matrices computed in
`beetles_landscapes.ipynb` (stable_landscapes.npy, aperiodic_landscapes.npy,
tseq.npy) to plain CSV, so that `results_partB.qmd` (R / knitr) can read them
without re-simulating or re-running persistent homology.

This mirrors Part A's split: Python (ripser) computes persistent homology and
landscapes -> CSV; R (fda) does the functional data analysis. Run this after
`beetles_landscapes.ipynb`, whenever the landscape matrices change.

WHY THIS SCRIPT REFUSES TO OVERWRITE SILENTLY
---------------------------------------------
On 3 Aug 2026 the committed .npy files were found to disagree with the committed
output/*.csv they are supposed to produce, and not by rounding: the grid ended at
0.765500 in the .npy against 0.768232 in the CSVs, and the .npy carried 74 non-zero
stable landscapes against 71.

That was resolved the same day. Re-running beetles_landscapes.ipynb, which has a fixed
seed, reproduced the .npy bit for bit and did not reproduce the CSVs, so the CSVs were
the stale side. They were regenerated, Part B was re-rendered against them, and the
memoria was updated to the figures that came out: 126 identically zero stable landscapes
of 200, a grid ending at 0.765500, and a mean landscape peak of 0.00202 against 0.22820.
The .npy and the CSVs have been the same data since, and results_partB.qmd asserts it at
render time.

The guard stays because the failure it caught was silent: running this script as
documented would have replaced the reported data with a different run without saying so.
It compares before writing and stops if the difference is larger than rounding. Pass
--force once you know which run you want; then re-render Part B and expect its assertions
to need updating.
"""

import argparse
import os
import sys

import numpy as np

TOL = 1e-9          # anything above this is a different run, not float noise

here = os.path.dirname(os.path.abspath(__file__))
out_dir = os.path.join(here, "output")
os.makedirs(out_dir, exist_ok=True)

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--force", action="store_true",
                    help="overwrite the CSVs even if they disagree with the .npy")
args = parser.parse_args()

arrays = {
    "stable_landscapes": np.load(os.path.join(here, "stable_landscapes.npy")),
    "aperiodic_landscapes": np.load(os.path.join(here, "aperiodic_landscapes.npy")),
    "tseq": np.load(os.path.join(here, "tseq.npy")),
}

conflicts = []
for name, arr in arrays.items():
    csv = os.path.join(out_dir, f"{name}.csv")
    if not os.path.exists(csv):
        continue
    old = np.loadtxt(csv, delimiter=",")
    if old.shape != arr.shape:
        conflicts.append(f"  {name:<22} shape {old.shape} on disk, {arr.shape} in the .npy")
    else:
        d = float(np.abs(old - arr).max())
        if d > TOL:
            conflicts.append(f"  {name:<22} max difference {d:.3e} against what is on disk")

if conflicts and not args.force:
    print("The .npy do not match the CSVs already in output/:\n", file=sys.stderr)
    print("\n".join(conflicts), file=sys.stderr)
    print(
        "\nThese CSVs are what results_partB.qmd reads, so overwriting them changes\n"
        "every number in Part B. Decide which run is the right one before you do.\n"
        "See the module docstring. Re-run with --force to write anyway.",
        file=sys.stderr,
    )
    raise SystemExit(1)

for name, arr in arrays.items():
    path = os.path.join(out_dir, f"{name}.csv")
    np.savetxt(path, arr, delimiter=",")
    print(f"Wrote {path}  {arr.shape}")
