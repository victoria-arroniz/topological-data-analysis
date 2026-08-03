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
On 3 Aug 2026 the committed .npy files were found to disagree with the
committed output/*.csv they are supposed to produce, and not by rounding:

    tseq          grid ends at 0.765500 (npy) vs 0.768232 (csv)
    stable        74 non-zero landscapes (npy) vs 71 (csv), peak .0626 vs .0487
    aperiodic     mean absolute difference 4.8e-02 on values up to 0.37

The .npy are dated 30 July, the CSVs 29 July, and the .npy match neither the
main arm nor the burn-in replica in output_burnin/. Every number reported in
Part B, and the "71 of them still carry a real (if small) transient H1 loop"
in results_partB.qmd, comes from the 29 July CSVs. The .npy are an unidentified
later run that was never propagated.

So running this script as documented would have replaced the reported data with
an unidentified one, quietly. The stopifnot() assertions in results_partB.qmd
would have caught it at render time, but only after the fact.

Rather than pick a winner, this script now compares before writing and stops if
the difference is larger than rounding. Pass --force once you know which run you
want; then re-render Part B and expect its assertions to need updating.
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
