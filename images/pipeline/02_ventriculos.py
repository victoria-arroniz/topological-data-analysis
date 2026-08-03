#!/usr/bin/env python3
"""
============================================================================
 Chapter 5 — Stage 2: extract the ventricles from the segmentation
============================================================================

 Replaces 03_limpieza.py, archived in _archivo/scripts_fsl_antiguos/. There,
 the largest connected component had to be kept because the CSF class from
 FAST came out in hundreds of pieces. That is no longer needed: SynthSeg
 gives the anatomical label directly.

 LABELS (FreeSurfer convention)
   4  left lateral ventricle          43  right lateral ventricle
   5  left temporal horn              44  right temporal horn
   14 third ventricle                 15  fourth ventricle

 WHAT COUNTS AS "THE CAVITY" (PREREGISTRO_CAP5.md section 8.5, settled)
 ----------------------------------------------------------------------
 The cavity is labels 4 and 43: the body and atrium of the lateral ventricles.
 All connected components are kept; nothing is filtered by size.

 The temporal horns (5, 44) are out of scope by declaration. They are genuine
 ventricle, but at 1 mm they detach from the body in 15 of the 16 subjects and
 only partially, so where the split falls is set by resolution rather than
 anatomy. Including them leaves the maximum persistence unchanged but lets
 landscape layers 2 to 5 measure how badly the horn fragmented, which
 correlates with ventricle size, the variable of interest.

 Left and right fall in the SAME connected component under 26-connectivity,
 joining near the foramina of Monro. Checked in all 16 subjects.

 This script writes the main definition and the pre-declared sensitivity:

   <s>_vent.nii.gz        labels 4, 43              MAIN
   <s>_vent_full.nii.gz   labels 4, 43, 5, 44       SENSITIVITY

 Both are reported, whatever they show.

 Usage:  python3 pipeline/02_ventriculos.py
============================================================================
"""
import os, csv, glob
import numpy as np, nibabel as nib
from scipy import ndimage as ndi

BASE = os.environ.get("RP", os.path.expanduser("~/rp"))
MAN  = f"{BASE}/T1_data_def/T1.csv"
SEG  = f"{BASE}/deriv/05_synthseg"

MAIN = [4, 43]            # body and atrium — the cavity, per section 8.5
FULL = [4, 43, 5, 44]     # plus the temporal horns — pre-declared sensitivity

meta = {r["Subject"]: r for r in csv.DictReader(open(MAN))}
assert len(meta) == 16, f"expected 16 subjects, the manifest has {len(meta)}"

rows = []
print(f"{'subject':>8} {'group':>8} | {'main_mL':>8} {'n_comp':>7} {'%largest':>9} "
      f"| {'full_mL':>8} {'n_comp':>7} | {'eTIV_mL':>9} {'main/eTIV%':>11}")

struct = np.ones((3, 3, 3))
for s in sorted(meta):
    f = f"{SEG}/{s}_seg.nii.gz"
    assert os.path.exists(f), f"{s}: {f} not found. Run 01_segmentacion.sh first"
    img = nib.load(f)
    lab = np.asarray(img.dataobj).astype(np.int16)
    vox_mL = float(np.prod(img.header.get_zooms()[:3])) / 1000.0

    m_main = np.isin(lab, MAIN)
    m_full = np.isin(lab, FULL)

    # eTIV, from the csv that SynthSeg writes
    vol = list(csv.DictReader(open(f"{SEG}/{s}_vol.csv")))[0]
    etiv = float(vol["total intracranial"]) / 1000.0

    for m, suffix in [(m_main, "vent"), (m_full, "vent_full")]:
        nib.save(nib.Nifti1Image(m.astype(np.uint8), img.affine, img.header),
                 f"{SEG}/{s}_{suffix}.nii.gz")

    lb, n_main = ndi.label(m_main, structure=struct)
    sizes = np.bincount(lb.ravel()); sizes[0] = 0
    pct_largest = 100.0 * sizes.max() / sizes.sum()
    n_full = ndi.label(m_full, structure=struct)[1]
    main_mL, full_mL = m_main.sum() * vox_mL, m_full.sum() * vox_mL

    # sanity check from section 8.5: left and right must share one component
    big = sizes.argmax()
    assert big in np.unique(lb[lab == 4]) and big in np.unique(lb[lab == 43]), \
        f"{s}: labels 4 and 43 are NOT in the same connected component"

    rows.append(dict(subject=s, group=meta[s]["Group"], sex=meta[s]["Sex"],
                     age=meta[s]["Age"],
                     main_mL=f"{main_mL:.2f}", n_components=n_main,
                     pct_in_largest=f"{pct_largest:.1f}",
                     full_mL=f"{full_mL:.2f}", n_components_full=n_full,
                     eTIV_mL=f"{etiv:.1f}",
                     main_pct_eTIV=f"{100*main_mL/etiv:.3f}"))
    print(f"{s:>8} {meta[s]['Group']:>8} | {main_mL:8.2f} {n_main:7d} {pct_largest:8.1f}% "
          f"| {full_mL:8.2f} {n_full:7d} | {etiv:9.1f} {100*main_mL/etiv:11.3f}")

with open(f"{SEG}/volumenes_ventriculos.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

print(f"\n{len(rows)} masks written to {SEG}")
print("Inspect deriv/qc before computing H2. The main definition and the")
print("pre-declared sensitivity are both reported, per section 8.5.")
