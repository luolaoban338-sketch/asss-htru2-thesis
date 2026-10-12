# Running and reproducing the code

Use the repository root as the working directory. The original development environment was Windows with Python 3.12.14. Dependency versions are pinned in `requirements-experiments-lock.txt`. Other platforms have not been shown to reproduce every floating-point result bit for bit.

## 1. Check the supplied files

```text
python scripts/verify_share.py
```

This verifies copied source/evidence SHA-256 values, the raw archive and processed data hashes, and the frozen ablation design references. It performs no fitting and does not modify the evidence. A mismatch is an error to investigate, not a reason to update a historical hash.

## 2. Run the tests and demonstration

```text
python -m unittest discover -s tests -v
python scripts/quickstart.py
```

Tests include synthetic model fits, analytic Shapley checks, selection and leakage contracts, budget accounting, corrupted-cache rejection, and comparisons with selected saved fixtures. Temporary outputs are separate from the historical runs. The demonstration is explicitly synthetic and cannot be cited as an empirical thesis result.

## 3. Optional bounded baseline run

```text
python scripts/run_baselines.py --dataset htru2 --run-id demo-baseline --max-jobs 1
```

This performs at most one new baseline fit and writes `runs/demo-baseline/`. Repeating the command resumes that run and can compute the next missing fit. To compute all 20 baseline jobs in a fresh run, use a new ID and `--max-jobs 20`.

Use new IDs for experiments. Do not use the partial historical baseline/joint fixture directories as destinations. Generated run directories are ignored by Git unless explicitly selected for a later evidence release.

## 4. Research experiment entry points

| Script | Purpose / requirements |
| --- | --- |
| `run_baselines.py` | Full data and three row-selection baselines; accepts a new run ID and a job limit |
| `run_feature_pilot.py` | Feature utility and Shapley development validation; uses the supplied frozen baseline splits |
| `run_joint_pilot.py` | Inner selection and outer evaluation; uses frozen baseline splits and accepts a new run ID |
| `run_stability_pilot.py` | Complete-selection resampling within the first outer training pool |
| `run_contamination.py` | Row-selector contamination study; consult its frozen parent-run requirements |
| `run_joint_contamination.py` | Joint contamination using frozen clean decisions; requires complete clean-run artifacts not supplied here |
| `run_ablation.py` | Historical fixed-budget package runner; preserves the frozen design, implementation gate and main/replay identifiers |

Inspect each script's `--help` and protocol before use. The joint and ablation workflows can require many model fits. The historical ablation runner is not a general command-line estimator and intentionally accepts only its frozen run identifiers. `STOP_ABLATION` prevents accidental ablation execution in this sharing snapshot. Full archival continuation is not part of the quick start; it requires the complete matching archive and unchanged execution environment. Starting a fresh fixed-budget replication should be a separately documented run, not a claim to have resumed the original run.

Audit, summary and replay scripts are supplied for inspection and for use with their required complete run outputs. An audit may refit models; it is not automatically a read-only operation. The quick-start integrity check is the no-fitting alternative.

## 5. Evidence interpretation

The dated files under `evidence/ablation_20261011/` are copied from the source project's completed aggregation. They retain zero contrasts and descriptive stability values. They do not contain the full raw package archive. The final ablation report, revised Results/Discussion and final delivery manifest were pending at the time of this snapshot.
