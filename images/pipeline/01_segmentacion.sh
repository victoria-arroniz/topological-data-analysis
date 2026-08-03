#!/bin/bash
# ============================================================================
#  Chapter 5 — Stage 1: segment with SynthSeg
# ============================================================================
#
#  Replaces the earlier chain (BET -> FAST -> registration to MNI 2 mm ->
#  dilated atlas ventricle mask -> largest connected component), which is
#  archived in _archivo/scripts_fsl_antiguos/. The rationale is in
#  PREREGISTRO_CAP5.md section 8.
#
#  WHAT IT DOES
#    mri_synthseg labels 32 structures on the RAW T1. No skull stripping is
#    needed: there is no BET, and therefore the measurement no longer depends
#    on where BET happened to cut (that dependence was r = 0.853 with the
#    birth time of the H2 class).
#
#  OUTPUT, per subject:
#    <s>_seg.nii.gz   one label per voxel
#    <s>_vol.csv      volume in mm3 of each structure, plus the eTIV
#
#  About 2 minutes per subject on CPU. Half an hour for all 16.
#
#  Cite: Billot et al., "SynthSeg: Segmentation of brain MRI scans of any
#  contrast and resolution without retraining", Medical Image Analysis, 2023.
#
#  Usage:  cd "$HOME/rp" && bash pipeline/01_segmentacion.sh
# ============================================================================

set -e
export LC_NUMERIC=C

BASE="${RP:-$HOME/rp}"
case "$BASE" in
  *\ *) echo "ERROR: \$BASE contains spaces ($BASE)."; exit 1 ;;
esac

MAN="$BASE/T1_data_def/T1.csv"
IN="$BASE/deriv/01_nifti"
OUT="$BASE/deriv/05_synthseg"
mkdir -p "$OUT"

command -v mri_synthseg >/dev/null || {
  echo "ERROR: mri_synthseg not found."
  echo "  export FREESURFER_HOME=/Applications/freesurfer/8.2.0"
  echo "  source \$FREESURFER_HOME/SetUpFreeSurfer.sh"
  exit 1
}
[ -f "$FREESURFER_HOME/license.txt" ] || [ -n "$FS_LICENSE" ] || {
  echo "ERROR: FreeSurfer licence not found."; exit 1; }
[ -f "$MAN" ] || { echo "ERROR: manifest not found at $MAN"; exit 1; }

SUBS=$(awk -F'","' 'NR>1 {print $2}' "$MAN" | tr -d '"')
N=$(echo "$SUBS" | wc -w | tr -d ' ')
[ "$N" = "16" ] || { echo "ERROR: expected 16 subjects, found $N"; exit 1; }
echo "subjects in the manifest: $N"

for s in $SUBS; do
  [ -f "$IN/${s}.nii.gz" ] || { echo "ERROR: $IN/${s}.nii.gz not found"; exit 1; }
  if [ -f "$OUT/${s}_seg.nii.gz" ]; then
    echo "=== $s (already done, skipping) ==="
    continue
  fi
  echo "=== $s ==="
  mri_synthseg --i "$IN/${s}.nii.gz" \
               --o "$OUT/${s}_seg.nii.gz" \
               --vol "$OUT/${s}_vol.csv" \
               --threads 8 --cpu --keepgeom
done

echo
echo "Done. Next, 02_ventriculos.py extracts the ventricular masks."
