# Thesis research: joint sample-feature selection and ASSS

**Latest update: 30 October 2026.** Research materials for *A Unified Framework for Robust Strategic Sample and Feature Selection*. Author: Feiyang Luo; supervisor: Chi Zhao.

## Research materials

| Material | Open |
| --- | --- |
| Code, protocols, manuscript chapters and result extracts | [30 October research update](updates/2026-10-30/) |
| File provenance and hashes | [Research manifest](updates/2026-10-30/PUBLICATION_MANIFEST.json) |

The research update covers work since the 30 September meeting. Traditional HTRU2 and native PowerSHAP five-fold comparisons and all four prespecified synthetic layers have been delivered. WDBC is complete for folds 0-2 only; folds 3-4 are deferred, and Adult remains runtime-gated. A3/A4 synthesis, whole-stage closure and the thesis remain incomplete.

The evidence does not establish overall ASSS superiority. On HTRU2, ASSS matches Mean Shapley and falls below the stronger baselines. The synthetic studies support a specified variance-estimation correction and a correlated-group redundancy effect, while showing mixed or adverse stability-penalty results. These are descriptive development findings.

**Repository structure.** The original repository-root implementation and frozen inputs remain the 12 October sharing snapshot. New research materials are appended under `updates/2026-10-30`; its code is an inspection snapshot, not a complete standalone reproduction environment. Original full experiment archives are retained locally. No experiments were rerun to prepare this publication.

这是公开研究资料入口，访问不需要邀请或关注。可直接查看上面的代码、论文草稿、实验协议与结果资料。关注作者、Star或Watch仅用于后续跟踪，并非查看权限条件。论文尚未完成，结果及其限制均按实际证据报告。

## Run the historical HTRU2 demonstration

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
