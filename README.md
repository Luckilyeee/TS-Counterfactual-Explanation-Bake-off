# Counterfactual Explanation Bake-off: A Review and Experimental Evaluation for Time Series Classification

This repository accompanies the paper **“Counterfactual Explanation Bake-off: A Review and Experimental Evaluation for Time Series Classification”**, which has been accepted for publication in the Machine Learning Journal.

It provides a benchmark-oriented implementation for counterfactual explanations on univariate time-series classification, covering multiple method families under a unified experimental setting.

For all commands and execution details, see **[RUN_GUIDE.md](./RUN_GUIDE.md)**.

## Paper-aligned benchmark scope

As described in the manuscript, the benchmark covers:

- 9 methods
- 16 models/variants
- 20 UCR datasets
- four key evaluation dimensions:
  - validity (flip rate)
  - proximity (L1/L2/L_inf)
  - sparsity
  - plausibility (IF / LOF / OC-SVM)

Method categories (paper-aligned):

- Instance-based / heuristic: NG-DBA, NG-CAM
- Shapelet-based: MG, SG
- Optimization-based: Wachter, TimeX, Glacier family
- Saliency-based: CELS, InfoCELS

## Repository structure

```text
.
|- CELS-Info_CELS/         # CELS and InfoCELS pipelines (PyTorch)
|- Glacier/                # Glacier framework and variants (TensorFlow)
|  `- src/
|- MG/                     # MG (shapelet + ROCKET/RF)
|- NG/                     # Native-Guide baselines (DBA and CAM)
|  `- src/
|- Wachter_TimeX_SG/       # Wachter, TimeX, and SG family
|- UCRArchive_2018/        # UCR datasets used in experiments
`- README.md
```

## Method-to-script mapping

- **CELS / InfoCELS** (`CELS-Info_CELS/`)
  - `main.py` -> CELS
  - `main_info.py` -> InfoCELS

- **Glacier** (`Glacier/src/Glacier.py`)
  - Glacier family with AE/NoAE and multiple constraint settings

- **MG** (`MG/main.py`)
  - Motif/shapelet-guided baseline

- **NG** (`NG/src/`)
  - `dba.py` -> NG-DBA
  - `cam.py` -> NG-CAM

- **Wachter / TimeX / SG** (`Wachter_TimeX_SG/`)
  - `mainAlibi.py` -> Wachter
  - `mainTimeX.py` -> TimeX
  - `mainTS.py` -> SG (binary)
  - `mainTS_mulclass.py` -> SG (multi-class)

## Method source attribution

Each benchmarked method has publicly available source code. For this repository, we borrowed and modified the original implementations to fit the unified experimental settings used in our paper.

Original method sources:
- **NG-DBA / NG-CAM**: https://github.com/e-delaney/Instance-Based_CFE_TSC
- **MG**: https://github.com/Luckilyeee/motif_guided_cf
- **SG**: https://github.com/Luckilyeee/SG-CF
- **Wachter**: https://github.com/SeldonIO/alibi
- **TimeX**: https://sites.google.com/view/timex-cf
- **Glacier**: https://github.com/zhendong3wang/learning-time-series-counterfactuals
- **CELS**: https://github.com/Luckilyeee/CELS
- **Info-CELS**: https://github.com/Luckilyeee/Info-CELS

## Dependencies and data notes

- Methods are implemented in semi-independent folders; environment setup is best managed per method.
- **Dataset folder location**: place `UCRArchive_2018/` at the repository root (`./UCRArchive_2018`).
- The datasets are sourced from the UCR Time Series Archive \[Dau et al., 2019\].
- All benchmarked method loaders now auto-resolve this folder by walking upward from each script location to find `UCRArchive_2018`.
- Optional override: set `UCR_DATA_ROOT` to an absolute dataset path (for example, `export UCR_DATA_ROOT=/data/UCRArchive_2018`).
- Legacy absolute fallback `/UCRArchive_2018` is still supported for backward compatibility.
- Several methods expect pretrained FCN/CAM weights (notably under `Wachter_TimeX_SG/` and `NG/src/cam.py`); see `RUN_GUIDE.md` for source and generation workflow details.
- This README provides a practical execution map; for methodological details, taxonomy rationale, and full experimental analysis, refer to our paper. 
- `RUN_GUIDE.md` contains extra operational details per method.


## Acknowledgement

We thank the maintainers of the UCR Time Series Archive for making the benchmark datasets publicly available.

## Reference

```bibtex
@article{dau2019ucr,
  title={The UCR time series archive},
  author={Dau, Hoang Anh and Bagnall, Anthony and Kamgar, Kaveh and Yeh, Chin-Chia Michael and Zhu, Yan and Gharghabi, Shaghayegh and Ratanamahatana, Chotirat Ann and Keogh, Eamonn},
  journal={IEEE/CAA Journal of Automatica Sinica},
  volume={6},
  number={6},
  pages={1293--1305},
  year={2019},
  publisher={IEEE}
}
```
