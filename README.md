# Code and data release: near-duplicate leakage in CT-KIDNEY classification

Companion files for the manuscript "Near-Duplicate Data Leakage Inflates Deep Learning
Performance in Multiclass Kidney Disease Classification from CT". Deposit this folder in a
public repository (for example GitHub with a Zenodo DOI) and cite the DOI in the
Data and code availability statement.

## Contents

| Path | What it is |
|---|---|
| `data/ct_kidney_grouped_split_seed42.csv` | One row per image of the CT-KIDNEY release (12,446 rows): file name, class, MD5, 64-bit dHash and pHash, size, view, DAP group id and the seed-42 split (train 8,711 / val 1,869 / test 1,866). The images themselves are not redistributed; get them from figshare (doi:10.6084/m9.figshare.30342133.v1) and match by MD5. |
| `Obj1_v12.8_pipeline_notebook.ipynb` | The full pipeline (data audit, hashing, grouping, split, training, leakage experiments, calibration, distillation, export) with saved outputs. Training ran on one NVIDIA RTX A4000. |
| `scripts/external_validation.py` | External testing on the two stone-versus-normal cohorts. |
| `scripts/rev1_test_inference.py` | Revision analysis: re-scores the seed-42 RenalDualNet, Swin-T and the FP32/INT8 students on the grouped test split, and runs the direct paired comparison. |
| `scripts/rev2_grouping_and_controls.py` | Revision analysis: injected-duplicate validation of the grouping, independent near-duplicate audit, 5-fold random vs grouped control experiment, external prediction tables. |
| `deploy/` | The shipped research-prototype application (Streamlit) and the INT8 ResNet-18 student. This version has no input-quality check. |
| `results/` | `RESULTS.json`, the per-table CSV files behind every table in the paper, and the frozen package list. |

Model checkpoints of the teacher and baselines (about 2 GB) are not included; they can be
shared on request or added to the repository with Git LFS.

## Reproducing the revision analyses

```
pip install -r results/requirements_frozen.txt
python scripts/rev1_test_inference.py      # needs the checkpoints and the images
python scripts/rev2_grouping_and_controls.py  # needs only the images and the split file
```

Both scripts expect the images under `~/Downloads/archive/CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone/...`;
change the `DATA` constant at the top of each script if the images are elsewhere.

## Licence and use

Code (scripts, notebook code and application): MIT, see `LICENSE`. The split file in
`data/` and the result files in `results/` are derived from the CT-KIDNEY images and are
released under CC BY 4.0, the licence of the source images (Islam et al., Scientific Reports
2022; figshare doi:10.6084/m9.figshare.30342133.v1). The images are not redistributed.

The application is a research prototype for a benchmark-integrity study. It is not a
medical device and must not be used for diagnosis.

## How to cite

Cite the archived release (Zenodo DOI: to be added after the first release) and the
accompanying article. Machine-readable metadata are in `CITATION.cff` and `.zenodo.json`.
