#!/usr/bin/env python3
"""
Build the anatomical figure for results_partC.qmd.

WHAT IT SHOWS
-------------
Two subjects, both controls, one aged 54 and one aged 74, with the SynthSeg
segmentation of the lateral ventricles overlaid and the cavity itself rendered
in three dimensions. Two controls rather than one of each group, deliberately:
the visible difference between them is age, not diagnosis, which is the finding
of the report.

WHY THE SAGITTAL VIEW IS NOT INCLUDED
-------------------------------------
A mid-sagittal slice of a T1 shows the facial profile, and facial geometry is
identifying: surface renderings of the face can be reconstructed from
unmodified structural MRI. Only axial and coronal views are drawn, and both are
cropped to the brain, so no facial feature reaches the figure. Subject labels
are the sub-NN aliases, never the PPMI numbers.

Reads from the local derivatives, which are not in this repository; writes the
PNG, which is.

Usage:  python3 tda-research/images/make_brain_figure.py
"""
import os, csv
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy import ndimage as ndi
from skimage import measure

BASE = os.environ.get("RP", os.path.expanduser("~/rp"))
HERE = os.path.dirname(os.path.abspath(__file__))
os.makedirs(f"{HERE}/figures", exist_ok=True)

alias = {r["ppmi_id"]: r["alias"] for r in csv.DictReader(open(f"{HERE}/subject_map.csv"))}
ppmi_of = {v: k for k, v in alias.items()}
meta = {r["Subject"]: r for r in csv.DictReader(open(f"{BASE}/T1_data_def/T1.csv"))}

# The two subjects are named by alias and resolved through subject_map.csv,
# which is git-ignored and never leaves this machine. Naming them by PPMI
# identifier here would put two identifiers, with their group and age, into a
# public repository, which the data use agreement does not permit. The aliases
# carry nothing on their own: output/metadata.csv already pairs them with group
# and age, and that pairing is what any published table would carry anyway.
# Both are controls, chosen to span the age range of the cohort.
PAIR = [ppmi_of[a] for a in ("sub-06", "sub-11")]
# The same red as the group contrast in the three reports, so the figure
# does not introduce a third colour into the document.
RED = "#C0392B"

fig = plt.figure(figsize=(12.5, 8.0))
gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 1.1], wspace=0.04, hspace=0.06)

for row, sid in enumerate(PAIR):
    t1 = nib.as_closest_canonical(nib.load(f"{BASE}/deriv/01_nifti/{sid}.nii.gz"))
    sg = nib.as_closest_canonical(nib.load(f"{BASE}/deriv/05_synthseg/{sid}_seg.nii.gz"))
    T = np.asarray(t1.dataobj, dtype=float)
    S = np.asarray(sg.dataobj)
    brain = S > 0                                   # every labelled voxel is brain
    vent = np.isin(S, [4, 43])                      # the cavity, main definition

    # crop to the brain, so neither skull nor face reaches the figure
    sl = ndi.find_objects(brain.astype(np.uint8))[0]
    pad = 3
    sl = tuple(slice(max(0, s.start - pad), s.stop + pad) for s in sl)
    T, S, brain, vent = T[sl], S[sl], brain[sl], vent[sl]
    T = np.where(brain, T, 0.0)                     # mask out anything not brain

    c = np.array(np.nonzero(vent)).mean(1).round().astype(int)
    vmax = np.percentile(T[T > 0], 99.5)

    for col, (name, f) in enumerate([("axial",   lambda A: np.rot90(A[:, :, c[2]])),
                                     ("coronal", lambda A: np.rot90(A[:, c[1], :]))]):
        ax = fig.add_subplot(gs[row, col]); ax.set_axis_off()
        ax.imshow(f(T), cmap="gray", vmin=0, vmax=vmax, interpolation="nearest")
        m = f(vent)
        ax.imshow(np.ma.masked_where(~m, m), cmap=ListedColormap([RED]),
                  alpha=0.75, interpolation="nearest")
        if row == 0:
            ax.set_title(name, fontsize=12)

    ax = fig.add_subplot(gs[row, 2], projection="3d")
    sub = tuple(slice(max(0, q.start - 4), q.stop + 4)
                for q in ndi.find_objects(vent.astype(np.uint8))[0])
    V = ndi.gaussian_filter(ndi.binary_closing(vent[sub], np.ones((2, 2, 2))).astype(float), 0.8)
    vt, fc, _, _ = measure.marching_cubes(V, 0.5, step_size=2)
    mesh = Poly3DCollection(vt[fc], alpha=1.0, linewidths=0)
    mesh.set_facecolor(RED); mesh.set_edgecolor("none")
    ax.add_collection3d(mesh)
    ax.set_xlim(0, V.shape[0]); ax.set_ylim(0, V.shape[1]); ax.set_zlim(0, V.shape[2])
    ax.set_box_aspect(V.shape); ax.view_init(elev=18, azim=-62); ax.set_axis_off()
    if row == 0:
        ax.set_title("the cavity itself, in 3D", fontsize=12)

    vol = vent.sum() * float(np.prod(nib.load(
        f"{BASE}/deriv/05_synthseg/{sid}_seg.nii.gz").header.get_zooms()[:3])) / 1000
    fig.text(0.012, 0.72 - row * 0.44,
             f"{alias[sid]}  {meta[sid]['Group']}\n{meta[sid]['Age']} y  ·  {vol:.1f} mL",
             fontsize=11, fontweight="bold", rotation=90, ha="center", va="center")

fig.suptitle("Lateral ventricles segmented from T1 with SynthSeg, and the cavity they define",
             fontsize=13.5, y=0.965)
fig.text(0.5, 0.015,
         "Both subjects are controls. Red: FreeSurfer labels 4 and 43. Images are cropped to the brain and no "
         "sagittal view is shown, because a mid-sagittal T1 slice carries the facial profile.",
         ha="center", fontsize=10, color="#495057")
out = f"{HERE}/figures/ventricle_example.png"
fig.savefig(out, dpi=140, facecolor="white", bbox_inches="tight")
print(f"written to {out}")
