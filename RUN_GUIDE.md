# Running Guide

This document is the operational companion to `README.md`.

- For project scope, method taxonomy, and structure overview, read `README.md`.
- For exact commands, paths, and outputs, use this file.

## 0) Environment and prerequisites

- Python environment with method-specific dependencies (PyTorch and/or TensorFlow).
- Pretrained FCN weights are required by several methods (especially under `Wachter_TimeX_SG/` and `NG/src/cam.py`).

Because each folder is a semi-independent research pipeline, dependency setup is best managed per method.

### Dataset path behavior (all methods)

- Place `UCRArchive_2018/` at the repository root:
  - `./UCRArchive_2018/<Dataset>/<Dataset>_TRAIN.tsv`
  - `./UCRArchive_2018/<Dataset>/<Dataset>_TEST.tsv`
- Dataset loaders now automatically search upward from each script location for `UCRArchive_2018`.
- Optional override: set `UCR_DATA_ROOT` to a custom dataset root.
  - Example: `export UCR_DATA_ROOT=/data/UCRArchive_2018`
- Legacy fallback to `/UCRArchive_2018` is still supported if present.

### Dependency version policy

- Each method folder includes a dedicated `requirements.txt`.
- We use **source-prioritized locking**:
  1. If the original method repository provided pinned versions, we keep those pins.
  2. If no official requirements existed, we pin versions that are broadly compatible with the method's stack and current ecosystem constraints.
- If an environment conflict appears on your machine, treat these as baseline pins and adjust method-locally.

### Pretrained FCN and CAM weights (important)

Several methods in this benchmark require pretrained FCN weights, and some also
require CAM weights (especially NG-CAM and methods reusing FCN checkpoints).

- Weight source: these can be obtained from the original NG repository:  
  https://github.com/e-delaney/Instance-Based_CFE_TSC
- In this benchmark, we reuse/adapt NG-style code to generate weights for
  different datasets under our unified settings.

Recommended workflow:

1. Prepare method dependencies (`NG/requirements.txt` and any method-specific requirements).
2. Generate or collect FCN weights per dataset (e.g., `<dataset>_best_model.hdf5`).
3. Generate CAM weights from the trained FCN (e.g., `cam_train_weights.npy`).
4. Place these artifacts in the paths expected by each script before running CF generation.

If you are reproducing all methods end-to-end, generate FCN/CAM weights first,
then run NG-CAM, Wachter/TimeX/SG, and any other method that depends on these
checkpoints.

---

## 1) CELS / InfoCELS (`CELS-Info_CELS/`)

### Install dependencies

From repository root:

```bash
python3 -m pip install -r CELS-Info_CELS/requirements.txt
```

If you need GPU-enabled PyTorch, install it first from the official PyTorch channel
for your CUDA version, then run the requirements command above.

### Entrypoints

- `main.py` (CELS)
- `main_info.py` (InfoCELS)

### Single-run examples

From repository root:

```bash
python3 CELS-Info_CELS/main.py --dataset Coffee --algo cf --pname CELS_Coffee
python3 CELS-Info_CELS/main_info.py --dataset Coffee --algo cf --pname InfoCELS_Coffee
```

### Batch runs

From `CELS-Info_CELS/`:

```bash
bash run_all.sh
bash run_all_infocels.sh
```

### Outputs

- `CELS/<dataset>/`, `CELS/flip_rates.csv`
- `InfoCELS/<dataset>/`, `InfoCELS/flip_rates.csv`

---

## 2) Glacier (`Glacier/src/Glacier.py`)

### Install dependencies

Option A (recommended, matches original Glacier setup):

```bash
conda env create -f Glacier/environment.yml
conda activate glacier
```

Option B (pip-only):

```bash
python3 -m pip install -r Glacier/requirements.txt
```

### Environment example

```bash
conda env create -f Glacier/environment.yml
conda activate glacier
```

### Single run

From `Glacier/src/`:

```bash
python3 Glacier.py --dataset Coffee --output Glacier_local.csv --w-type local --w-value 0.5 --tau-value 0.5
```

### Glacier 8 variants (paper-aligned naming)

The 8 paper variants are:

- `Glacier-AE-Unc`
- `Glacier-AE-Loc`
- `Glacier-AE-Glob`
- `Glacier-AE-Unif`
- `Glacier-NoAE-Unc`
- `Glacier-NoAE-Loc`
- `Glacier-NoAE-Glob`
- `Glacier-NoAE-Unif`

Code-level mapping:

- `AE / NoAE` -> CSV `method` (`Autoencoder` / `No autoencoder`)
- `Unc / Loc / Glob / Unif` -> CSV `step_weight_type` (`unconstrained` / `local` / `global` / `uniform`)

Each single run evaluates both AE and NoAE.  
So 4 runs (one per `--w-type`) produce all 8 variants:

```bash
cd Glacier/src
python3 Glacier.py --dataset Coffee --output Glacier_variants.csv --w-type local --w-value 0.5 --tau-value 0.5
python3 Glacier.py --dataset Coffee --output Glacier_variants.csv --w-type global --w-value 0.5 --tau-value 0.5
python3 Glacier.py --dataset Coffee --output Glacier_variants.csv --w-type uniform --w-value 0.5 --tau-value 0.5
python3 Glacier.py --dataset Coffee --output Glacier_variants.csv --w-type unconstrained --w-value 0.5 --tau-value 0.5
```

Variant-identification columns:

- `step_weight_type`
- `method`
- `pred_margin_weight`
- `threshold_tau`

### Outputs

- CF arrays under `results/cfs_glacier/`
- summary rows appended to the CSV passed via `--output`

---

## 3) MG (`MG/main.py`)

### Install dependencies

From repository root:

```bash
python3 -m pip install -r MG/requirements.txt
```

From repository root:

```bash
python3 MG/main.py
```

### Outputs

- `res_rk/<dataset>/`
- `res_rk/MG-RF.csv`

---

## 4) NG baselines (`NG/src/`)

### Install dependencies

From repository root:

```bash
python3 -m pip install -r NG/requirements.txt
```

## 4.1) NG_DBA (`dba.py`)

From `NG/src/`:

```bash
python3 dba.py
```

Notes:

- Uses ROCKET + RandomForest and DTW barycenter averaging.
- Uses the shared dataset path resolution described in section 0.

Outputs:

- `/NG/src/results_dba_rocket/`
- `/NG/src/results_dba_rocket/NG-DBA.csv`

## 4.2) NG_CAM (`cam.py`)

From `NG/src/`:

```bash
python3 cam.py
```

Notes:

- Expects pretrained FCN and CAM weights in hardcoded paths.
- `path = '...'` placeholder in script must be set before running.

Outputs:

- `/NG/src/results_cam/`
- `/NG/src/results_cam/all_datasets_cam.csv`

---

## 5) Wachter / TimeX / SG (`Wachter_TimeX_SG/`)

### Install dependencies

From repository root:

```bash
python3 -m pip install -r Wachter_TimeX_SG/requirements.txt
```

Notes:

- The codebase includes a lightweight local `alibi` module under `Wachter_TimeX_SG/alibi`.
- You do not need to install full upstream `alibi` for these local scripts.

### 5.1) Wachter (`mainAlibi.py`)

```bash
cd Wachter_TimeX_SG
python3 mainAlibi.py
```

Outputs:

- `./logs/ALIBI/<DATASET>_res.csv`
- `./logs/ALIBI/<DATASET>_cfs.npy`

### 5.2) TimeX (`mainTimeX.py`)

```bash
cd Wachter_TimeX_SG
python3 mainTimeX.py
```

Outputs:

- `./logs/TimeX/<DATASET>_LOG_0.csv`
- `./logs/TimeX/<DATASET>_cfs0.npy`

### 5.3) SG binary (`mainTS.py`)

```bash
cd Wachter_TimeX_SG
python3 mainTS.py
```

Outputs:

- `./logs/SG_CF/<DATASET>_LOG_.csv`
- `./logs/SG_CF/<DATASET>_cfs.npy`

### 5.4) SG multi-class (`mainTS_mulclass.py`)

```bash
cd Wachter_TimeX_SG
python3 mainTS_mulclass.py
```

Outputs:

- `./logs/SG_CF/<DATASET>_LOG_.csv`
- `./logs/SG_CF/<DATASET>_cfs.npy`

---

## Practical recommendation

Start with one small dataset (for example, `Coffee`) per method to validate environment and paths, then scale to full dataset lists.
