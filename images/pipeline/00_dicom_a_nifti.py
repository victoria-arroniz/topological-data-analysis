#!/usr/bin/env python3
"""
============================================================================
 Chapter 5 — Stage 0: DICOM -> NIfTI
============================================================================

 Cohort: 16 subjects (8 Control, 8 PD), all at the baseline visit.

 THE PITFALL THIS SCRIPT SOLVES
 ------------------------------
 The MPRAGE is acquired sagittally, so the volume advances along x, not z.
 Sorting slices by ImagePositionPatient[2] leaves the volume scrambled: in
 this cohort the range of z across the 192 slices spans 0 to 16 mm, and in
 several subjects it is EXACTLY 0, so the ordering is not even defined.

 The fix is to project the positions onto the plane NORMAL, which is the
 cross product of the two ImageOrientationPatient vectors. The slice spacing
 then comes out at 1.0000 mm with zero standard deviation.

 The affine is built from the DICOM header and then verified: each slice's
 ImagePositionPatient is reconstructed from it and compared.

 Usage:  python3 pipeline/00_dicom_a_nifti.py
 Output: deriv/01_nifti/<subject>.nii.gz  and  deriv/01_nifti/geometry.csv
============================================================================
"""
import os, glob, csv, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pydicom
import nibabel as nib

BASE = os.environ.get("RP", os.path.expanduser("~/rp"))
ROOT = f"{BASE}/T1_data_def"
MAN  = f"{ROOT}/T1.csv"
OUT  = f"{BASE}/deriv/01_nifti"
os.makedirs(OUT, exist_ok=True)

manifest = {r["Subject"]: r for r in csv.DictReader(open(MAN))}
assert len(manifest) == 16, f"expected 16 subjects, the manifest has {len(manifest)}"
assert all(r["Visit"] == "BL" for r in manifest.values()), "found visits other than BL"

log = []
for sid, meta in sorted(manifest.items()):
    folder = glob.glob(f"{ROOT}/PPMI*/{sid}/")
    assert len(folder) == 1, f"{sid}: found {len(folder)} folders"
    files = sorted(glob.glob(folder[0] + "**/*.dcm", recursive=True))
    ds = [pydicom.dcmread(f) for f in files]

    series = {d.SeriesInstanceUID for d in ds}
    assert len(series) == 1, f"{sid}: {len(series)} distinct series"

    iop = np.array([float(v) for v in ds[0].ImageOrientationPatient])
    row_dir, col_dir = iop[:3], iop[3:]
    normal = np.cross(row_dir, col_dir)

    pos = np.array([[float(v) for v in d.ImagePositionPatient] for d in ds])
    order = np.argsort(pos @ normal)              # <- the normal, not z
    ds = [ds[i] for i in order]
    pos = pos[order]

    step = np.diff(pos @ normal)
    # 1e-4 mm = 0.1 microns. Below that it is floating-point noise.
    assert step.std() < 1e-4, f"{sid}: non-uniform spacing (sd {step.std():.2e})"

    ps = [float(v) for v in ds[0].PixelSpacing]   # [between rows, between columns]
    delta = (pos[-1] - pos[0]) / (len(ds) - 1)

    vol = np.stack([d.pixel_array.astype(np.float32).T for d in ds], axis=-1)
    slope     = float(getattr(ds[0], "RescaleSlope", 1) or 1)
    intercept = float(getattr(ds[0], "RescaleIntercept", 0) or 0)
    vol = vol * slope + intercept

    A = np.eye(4)
    A[:3, 0] = row_dir * ps[1]
    A[:3, 1] = col_dir * ps[0]
    A[:3, 2] = delta
    A[:3, 3] = pos[0]

    rebuilt = np.array([(A @ np.array([0, 0, k, 1]))[:3] for k in range(len(ds))])
    err = float(np.abs(rebuilt - pos).max())
    assert err < 1e-3, f"{sid}: the affine does not reproduce the positions (error {err:.2e} mm)"

    A[:2, :] *= -1                                # LPS -> RAS
    img = nib.Nifti1Image(vol, A)
    img.header.set_xyzt_units("mm")
    nib.save(img, f"{OUT}/{sid}.nii.gz")

    log.append(dict(subject=sid, group=meta["Group"], sex=meta["Sex"], age=meta["Age"],
                    slices=len(ds), shape="x".join(map(str, vol.shape)),
                    step_mm=f"{step.mean():.4f}", affine_error_mm=f"{err:.2e}",
                    axes="".join(nib.aff2axcodes(A))))
    print(f"{sid:8s} {meta['Group']:8s} {len(ds):4d} slices  step {step.mean():.4f} mm  "
          f"affine error {err:.1e} mm")

with open(f"{OUT}/geometry.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(log[0]))
    w.writeheader(); w.writerows(log)

print(f"\n{len(log)} volumes written to {OUT}")
