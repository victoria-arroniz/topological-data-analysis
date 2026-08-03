#!/usr/bin/env python3
"""
Export the Chapter 5 derived data into this repository, anonymised.

WHY THIS SCRIPT EXISTS
----------------------
Parts A and B commit their raw inputs: adjacency matrices and simulated series
carry no personal information. Part C cannot. The imaging data are DICOM from
identified PPMI participants under a Data Use Agreement that does not permit
redistribution, and the subject identifiers are themselves identifying, so even
a directory listing of per-subject files would leak them.

What is committed instead is the topological summary: the birth-death pairs and
the landscapes evaluated on the common grid. These are the same objects that
`funTDA/output/PH/` holds for Part A. They are derived quantities two steps
removed from the image, and they are what `results_partC.qmd` needs to
reproduce every number in the report.

WHAT IS ANONYMISED
------------------
PPMI identifiers are replaced by sub-01 ... sub-16, assigned in order of the
sorted identifier so the mapping is deterministic. The mapping itself is
written to `subject_map.csv`, which is git-ignored and stays on the local
machine.

WHAT METADATA IS KEPT, AND WHAT IS NOT
--------------------------------------
Kept: group and age. Age is needed for the positive control of stage 5 and for
the age-radius correlation, and (group, age) pairs for sixteen subjects drawn
from a cohort of several thousand are what any published table carries.

Dropped: SEX, and two columns no analysis reads. Sex is reported in aggregate
in results_partC.qmd, where the relevant fact is that it is balanced between
groups (5 M / 3 F in control, 4 M / 4 F in PD; Fisher exact p = 1.00), and so
cannot confound the comparison. Carried per subject it would only sharpen the
quasi-identifier without being used by anything: on this cohort (group, sex,
age) makes 14 of 16 rows unique, and (group, age) already makes 14 of 16, so
sex adds re-identification surface for no analytical return.

Usage:  python3 tda-research/images/export_for_repo.py
"""
import os, csv, glob, json, shutil

BASE = os.environ.get("RP", os.path.expanduser("~/rp"))
MAN  = f"{BASE}/T1_data_def/T1.csv"
TDA  = f"{BASE}/deriv/07_tda"
SEG  = f"{BASE}/deriv/05_synthseg"
MNI  = f"{BASE}/deriv/06_mni"
HERE = os.path.dirname(os.path.abspath(__file__))
OUT  = f"{HERE}/output"

os.makedirs(f"{OUT}/diagrams", exist_ok=True)

meta = {r["Subject"]: r for r in csv.DictReader(open(MAN))}
subjects = sorted(meta)
assert len(subjects) == 16, f"expected 16 subjects, found {len(subjects)}"
alias = {s: f"sub-{i+1:02d}" for i, s in enumerate(subjects)}

# the mapping stays local: see .gitignore
with open(f"{HERE}/subject_map.csv", "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["alias", "ppmi_id"])
    for s in subjects:
        w.writerow([alias[s], s])

# --- metadata -------------------------------------------------------------
etiv = {r["subject"]: r for r in csv.DictReader(open(f"{SEG}/volumenes_ventriculos.csv"))}
det = {r["subject"]: r for r in csv.DictReader(open(f"{MNI}/volumes_mni.csv"))}
with open(f"{OUT}/metadata.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=["subject", "group", "age",
                                       "ventricle_mL_main", "n_components",
                                       "eTIV_mL"])
    w.writeheader()
    for s in subjects:
        w.writerow(dict(subject=alias[s], group=meta[s]["Group"],
                        age=meta[s]["Age"],
                        ventricle_mL_main=etiv[s]["main_mL"],
                        n_components=etiv[s]["n_components"],
                        eTIV_mL=etiv[s]["eTIV_mL"]))

# aggregate sex distribution: reported, never carried per subject
sex_tab = {}
for s in subjects:
    key = (meta[s]["Group"], meta[s]["Sex"])
    sex_tab[key] = sex_tab.get(key, 0) + 1
with open(f"{OUT}/sex_distribution.csv", "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["group", "M", "F"])
    for g in ("Control", "PD"):
        w.writerow([g, sex_tab.get((g, "M"), 0), sex_tab.get((g, "F"), 0)])

# --- persistence diagrams -------------------------------------------------
n_dgm = 0
for s in subjects:
    for f in sorted(glob.glob(f"{TDA}/{s}_H*_*.csv")):
        tag = os.path.basename(f).replace(f"{s}_", "")
        shutil.copyfile(f, f"{OUT}/diagrams/{alias[s]}_{tag}")
        n_dgm += 1

# --- landscapes: rename the subject columns -------------------------------
n_land = 0
for f in sorted(glob.glob(f"{TDA}/landscapes_*.csv")):
    rows = list(csv.reader(open(f)))
    rows[0] = [rows[0][0]] + [alias[c] for c in rows[0][1:]]
    with open(f"{OUT}/{os.path.basename(f)}", "w", newline="") as fh:
        csv.writer(fh).writerows(rows)
    n_land += 1

# --- test results: rename the subject list --------------------------------
for name in ["resultados_test.json", "diagnostico_edad.json"]:
    p = f"{TDA}/{name}"
    if not os.path.exists(p):
        continue
    d = json.load(open(p))
    if "subjects" in d:
        d["subjects"] = [alias[s] for s in d["subjects"]]
    if "younger" in d:
        d["younger"] = [alias[s] for s in d["younger"]]
    json.dump(d, open(f"{OUT}/{name}", "w"), indent=2)

print(f"metadata.csv        16 subjects, anonymised (group and age only)")
print(f"sex_distribution.csv  aggregate 2x2 table, no per-subject sex")
print(f"diagrams/           {n_dgm} persistence diagrams")
print(f"landscapes          {n_land} matrices")
print(f"subject_map.csv     written locally, git-ignored")
print(f"\nwritten to {OUT}")
