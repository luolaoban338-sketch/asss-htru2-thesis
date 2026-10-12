# ASSS and joint sample–feature selection on HTRU2

Research code for an English master's thesis investigating Adaptive Stable Shapley Selection (ASSS) and joint sample–feature selection. This repository contains the current HTRU2 development implementation, frozen configurations, tests, and selected evidence tables.

**Status:** development validation, not final thesis-wide validation. The code-sharing snapshot was prepared on 12 October 2026. The latest included fixed-budget ablation evidence is dated 11 October 2026. Document integration and final delivery remain pending.

## Start here

1. Read this README and [the scope of this sharing snapshot](docs/SHARING_SCOPE.md).
2. Install Python 3.12 and the pinned dependencies below.
3. Run the integrity check, tests, and short demonstration.
4. Read [the method map](docs/METHOD_MAP.md) and [reproduction guide](docs/REPRODUCING.md) before running experiments.

### Windows PowerShell

From the downloaded or cloned repository directory:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-experiments-lock.txt
.\.venv\Scripts\python.exe scripts/verify_share.py
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe scripts/quickstart.py
```

### Linux / macOS

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-experiments-lock.txt
.venv/bin/python scripts/verify_share.py
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python scripts/quickstart.py
```

The demonstration verifies the HTRU2 data and illustrates the feature-selection modules using an explicitly synthetic utility game. It does not produce a thesis result or start a full experiment. The test suite uses small synthetic problems and saved fixtures; it does not rerun the full HTRU2 experiment.

## Repository map

| Path | Contents |
| --- | --- |
| `src/` | Data loading, partition contracts, preprocessing, row sampling and evaluation |
| `experiments/` | Shapley estimation, corrected scores, feature paths, joint selection, stability and ablation |
| `scripts/` | Experiment entry points, independent audits, replay checks, summaries and sharing utilities |
| `configs/`, `protocols/` | Frozen development settings and the fixed-budget ablation protocol |
| `tests/` | Existing implementation tests, including training/test separation and integrity checks |
| `data/` | Original UCI HTRU2 archive, derived table, provenance and checksums |
| `runs/` | Selected historical inputs used by tests and frozen protocols; **incomplete runs** |
| `qa/` | The historical ablation implementation gate only |
| `evidence/ablation_20261011/` | Selected fixed-budget result tables and an archived numeric-audit report |
| `SOURCE_MANIFEST.json` | SHA-256 inventory linking unchanged files to the source project |

## Method and evidence boundaries

ASSS is the feature-selection component; the joint framework additionally chooses the sample and feature budgets. Feature contributions use training-only utility games. Monte Carlo variance correction, a variability penalty and a redundancy penalty feed a deterministic forward feature path. Joint selection uses inner validation; outer-test outcomes must not tune the method.

The included fixed-budget study uses `rho=0.5` and `k=4`. Its 15 predictive packages and 10 training-only stability resamples were completed and aggregated in the source project. All 135 primary seed-level F1 module contrasts were zero. All eight logical module variants had Nogueira stability and mean Jaccard equal to 1 within each of the three samplers in this fixed-budget setting. Ordered paths sometimes differed despite identical selected four-feature sets.

These are descriptive development results. They do not establish a general module effect, universal robustness, superiority, equivalence or noninferiority. Fixed-budget stability is different from adaptive complete-selection stability. HTRU2 has only eight predictors; this repository does not establish high-dimensional scalability or deployment performance.

## Data and attribution

Lyon, R. (2015). **HTRU2** [Dataset]. UCI Machine Learning Repository. <https://doi.org/10.24432/C5DK6R>.

HTRU2 contains 17,898 rows, eight predictors and 1,639 positive examples. The UCI dataset is distributed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). See [the dataset page](https://archive.ics.uci.edu/dataset/372/htru2), [local data notes](data/README.md), and the original `data/raw/Readme.txt`.

## Software license

No open-source software license has been selected for the thesis code. The dataset license is separate from the code. This snapshot is intended for thesis inspection and agreed research collaboration; do not describe the code as openly licensed.

## 中文说明

这是论文当前 HTRU2 代码的独立分享版本，包含算法、测试、运行配置、数据及部分结果表。原实验目录和完整审计档案保留在本地。先按上面的命令检查文件、运行测试和演示，再阅读复现说明。`runs/` 中仅保留测试及冻结协议需要的部分文件，不代表上传了完整实验记录。当前结果仍属于开发阶段验证。
