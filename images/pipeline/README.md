# Chapter 5 processing chain

Cohort: **16 subjects, 8 Control and 8 PD**, SAG 3D MPRAGE, 1.0 mm isotropic,
256×256×192, TR 2300 / TE 2.74 / TI 900 / FA 9. **All at the baseline visit.**
Manifest: `T1_data_def/T1.csv`.

Everything runs from `~/rp`, a space-free symlink to the working folder, because
FSL does not accept paths with spaces.

These scripts used to exist twice, byte for byte: once here and once in a
`pipeline/` directory at the top of the working folder. Since 3 Aug 2026 there is
one copy, this one, and the top-level `pipeline/` is a symlink to it. The usage
lines below are unchanged and still work: `python3 pipeline/04_analisis.py` from
`~/rp` resolves through the link. Edit the files here.

```
ln -s "$HOME/Desktop/Research Project" "$HOME/rp"
export FREESURFER_HOME=/Applications/freesurfer/8.2.0
source $FREESURFER_HOME/SetUpFreeSurfer.sh
cd "$HOME/rp"
```

## Order of execution

| step | what it does | who |
|---|---|---|
| `00_dicom_a_nifti.py` | DICOM → NIfTI, slices ordered by the plane normal | automatic |
| `01_segmentacion.sh` | SynthSeg: 32 structures labelled on the raw T1 | FreeSurfer |
| `02_ventriculos.py` | extracts the ventricular masks from the labels | automatic |
| — | **visual review of all 16 masks** | manual |
| `03_espacio_comun.sh` | registration to MNI152 1 mm, nearest-neighbour resampling of the masks | FSL |
| `04_analisis.py` | H₂ and H₁, landscapes λ₁…λ₅, FPCA, exhaustive permutation test | automatic |
| `05_diagnostico_edad.py` | the age contrast, declared post hoc | automatic |

The step in bold is not optional. A mask is never accepted on the strength of
its number alone. All sixteen were reviewed and none was rejected.

All five stages have been run. What a clone of this repository can reproduce is
not the chain but the analysis: every stage reads the manifest at
`T1_data_def/T1.csv` and the derivatives under `deriv/`, neither of which can be
redistributed, so `results_partC.qmd` is the entry point instead. It recomputes
every number of Chapter 5 from the committed CSVs under `images/output/`.

## What changed on 2 August 2026

The earlier chain (BET, FAST, registration to MNI 2 mm to bring in the atlas
ventricle mask, cropping, and largest connected component) was retired on that
date and is not part of this repository. SynthSeg replaces it. The rationale and
the formal declaration are in `PREREGISTRO_CAP5.md` section 8.

What this buys:

- **No BET.** SynthSeg runs on the raw T1, so the measurement no longer depends
  on where skull stripping happened to cut.
- **No atlas mask** and no dilation, and therefore no truncation.
- **No largest connected component**, and therefore no ventricle discarded.
- **Left and right are separate**, and so are the temporal horns.
- **The eTIV comes included**, replacing the head-size correction based on the
  determinant of the `flirt` affine.

## Output layout

```
deriv/
├── 01_nifti/       converted volumes + geometry.csv
├── 05_synthseg/    <s>_seg.nii.gz, <s>_vol.csv, <s>_vent.nii.gz,
│                   <s>_vent_con3.nii.gz, volumenes_ventriculos.csv
├── 06_mni/         registration to MNI 1 mm
├── 07_tda/         diagrams, landscapes, test
└── qc/             quality-control figures
```

`deriv/02_bet/` and `deriv/03_seg/` belong to the earlier chain and are no
longer used.

## SynthSeg labels that matter here

| id | structure |
|---|---|
| 4 · 43 | left · right lateral ventricle |
| 5 · 44 | left · right temporal horn |
| 14 · 15 | third · fourth ventricle |

The object of this chapter is the body and atrium of the lateral ventricles,
labels **4 and 43**. Labels 5 and 44 are the pre-declared sensitivity analysis;
see the next section.

## What counts as "the cavity" — settled 2 August 2026

**The cavity is labels 4 and 43**: the body and atrium of the lateral
ventricles. All connected components are kept; nothing is filtered by size.

The temporal horns (labels 5 and 44) are **out of scope, by declaration**. They
are genuine ventricle, but at 1 mm they detach from the body in 15 of the 16
subjects, and only partially — about 40 % of the total temporal-horn volume ends
up disconnected, and where the split falls is decided by resolution rather than
anatomy. Including them leaves the maximum persistence unchanged but lets
landscape layers 2 to 5 measure how badly the horn fragmented, which correlates
with ventricle size, the variable of interest.

With labels 4 and 43 the largest component holds between 98.3 % and 100 % of the
mask in all 16 subjects, so fragmentation stops being a factor.

Left and right lateral ventricles fall in the **same** connected component under
26-connectivity, joining near the foramina of Monro. Checked in all 16.

A sensitivity analysis over the full definition (4, 43, 5, 44, all components) is
pre-declared and will be reported alongside the main one. See
`PREREGISTRO_CAP5.md` section 8.5.

## Checks the chain runs on itself

- That the manifest holds 16 subjects and all are baseline.
- That each subject has a single series with uniform slice spacing.
- That the constructed affine reproduces each slice's `ImagePositionPatient`.
- That the FreeSurfer licence is in place before starting.
- `LC_NUMERIC=C` in the shell scripts.
