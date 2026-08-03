#!/bin/bash
# ============================================================================
#  Chapter 5 — Stage 3: bring the ventricle masks into MNI152 1 mm
# ============================================================================
#
#  WHY THIS STAGE EXISTS
#  ---------------------
#  The primary analysis is measured in MNI space (PREREGISTRO_CAP5.md section 3):
#  each brain is registered with a 12-DOF affine so that head size is normalised
#  and the persistence of H2 comes out in millimetres of MNI. The native-space
#  analysis is the sensitivity arm, and there head size is corrected with the
#  eTIV instead (section 8.3).
#
#  The reference is the 1 mm template, not the 2 mm one: the quantity being
#  measured is a radius in millimetres, and a 2 mm grid would impose a 2 mm
#  grain on it.
#
#  WHERE THE BRAIN MASK COMES FROM
#  -------------------------------
#  There is no BET in this chain any more. The brain mask is built from the
#  SynthSeg labels themselves: every labelled voxel is brain. That is both more
#  accurate than BET and free, since the segmentation is already computed.
#
#  The masks are resampled with NEAREST NEIGHBOUR so they stay binary.
#
#  Input:   deriv/01_nifti/<s>.nii.gz
#           deriv/05_synthseg/<s>_seg.nii.gz
#           deriv/05_synthseg/<s>_vent.nii.gz, <s>_vent_full.nii.gz
#  Output:  deriv/06_mni/<s>_nat2mni.mat, <s>_brain_mni.nii.gz,
#           <s>_vent_mni.nii.gz, <s>_vent_full_mni.nii.gz, volumes_mni.csv
#
#  Usage:  cd "$HOME/rp" && bash pipeline/03_espacio_comun.sh
# ============================================================================

set -e
export LC_NUMERIC=C          # without this awk writes a decimal comma

BASE="${RP:-$HOME/rp}"
case "$BASE" in
  *\ *) echo "ERROR: \$BASE contains spaces ($BASE). FSL will fail."; exit 1 ;;
esac

MAN="$BASE/T1_data_def/T1.csv"
NII="$BASE/deriv/01_nifti"
SEG="$BASE/deriv/05_synthseg"
OUT="$BASE/deriv/06_mni"
REF="$FSLDIR/data/standard/MNI152_T1_1mm_brain.nii.gz"
mkdir -p "$OUT"

for t in flirt fslmaths fslstats; do
  command -v "$t" >/dev/null || { echo "ERROR: $t not found. Load the FSL environment."; exit 1; }
done
[ -f "$REF" ] || { echo "ERROR: reference not found at $REF"; exit 1; }
[ -f "$MAN" ] || { echo "ERROR: manifest not found at $MAN"; exit 1; }

SUBS=$(awk -F'","' 'NR>1 {print $2}' "$MAN" | tr -d '"')
N=$(echo "$SUBS" | wc -w | tr -d ' ')
[ "$N" = "16" ] || { echo "ERROR: expected 16 subjects, found $N"; exit 1; }

echo "subject,vox_nat,mL_nat,vox_mni,mL_mni,det_nat2mni,linear_scale" > "$OUT/volumes_mni.csv"

for s in $SUBS; do
  echo "=== $s ==="
  for f in "$NII/${s}.nii.gz" "$SEG/${s}_seg.nii.gz" "$SEG/${s}_vent.nii.gz"; do
    [ -f "$f" ] || { echo "ERROR: $f not found"; exit 1; }
  done

  # brain mask straight from the segmentation: any labelled voxel is brain
  fslmaths "$SEG/${s}_seg" -bin -fillh "$OUT/${s}_brainmask"
  fslmaths "$NII/${s}"     -mas "$OUT/${s}_brainmask" "$OUT/${s}_brain"

  # 12 degrees of freedom, so scaling can differ along x, y and z
  flirt -in "$OUT/${s}_brain" -ref "$REF" -dof 12 \
        -omat "$OUT/${s}_nat2mni.mat" -out "$OUT/${s}_brain_mni"

  for v in vent vent_full; do
    [ -f "$SEG/${s}_${v}.nii.gz" ] || continue
    flirt -in "$SEG/${s}_${v}" -ref "$REF" -applyxfm \
          -init "$OUT/${s}_nat2mni.mat" -interp nearestneighbour \
          -out "$OUT/${s}_${v}_mni"
  done

  # determinant of the affine: the volume ratio must match it, which is the
  # check that caught the resampling problem in the earlier chain
  det=$(python3 - "$OUT/${s}_nat2mni.mat" <<'PY'
import sys, numpy as np
A = np.loadtxt(sys.argv[1])
print(f"{abs(np.linalg.det(A[:3,:3])):.4f}")
PY
)
  lin=$(python3 -c "print(f'{$det ** (1/3):.4f}')")
  vn=$(fslstats "$SEG/${s}_vent"     -V | awk '{print $1}')
  mn=$(fslstats "$SEG/${s}_vent"     -V | awk '{printf "%.2f", $2/1000}')
  vm=$(fslstats "$OUT/${s}_vent_mni" -V | awk '{print $1}')
  mm=$(fslstats "$OUT/${s}_vent_mni" -V | awk '{printf "%.2f", $2/1000}')
  echo "$s,$vn,$mn,$vm,$mm,$det,$lin" >> "$OUT/volumes_mni.csv"
  echo "    $mn mL native / $mm mL MNI 1 mm   (det $det, linear scale $lin)"
done

echo
echo "Done. Check that mL_mni / mL_nat matches det_nat2mni for every subject,"
echo "then run 04_analisis.py."
