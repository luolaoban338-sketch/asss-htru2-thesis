# HTRU2 joint-process training contamination

Frozen on 2026-10-03 before new fitting. The user authorized this next stage after completing the English Results chapter. This instantiates the previously planned joint contamination design on HTRU2. It is a development sensitivity experiment because the dataset and clean results have already been examined.

## Fixed decisions

Read each fold's eight decisions from joint-htru2-pilot-20261001-v1. Freeze sampling method, rho, lambda, eta and k to these clean decisions. Record their hashes before fitting. Do not perform an inner search on contaminated data or use any new outer score to select parameters. Rebuild sample identities, contribution estimates, redundancy and feature identities with the frozen budget. Reuse the parent B=4, M=256, 80% contribution subsampling, three utility folds, exact-reference diagnostics and seed derivation unchanged. Only sampling methods selected by at least one of the eight arms need selectors. Shared selectors and identical models count once.

## Perturbation

Use all five original outer folds. Each fold has one clean condition and six contaminated conditions: rates 5% and 10%, each with draws 0, 1, 2. Reuse the sampling-contamination seed root 20260927, signed 10-IQR offset, ceil(rate*n) row count, row-identity sorting, and nested 5%/10% draws. IQR is computed on clean outer training rows. All arms share an identical corrupted training matrix. Change one observed eligible numerical cell per selected row. Labels, identities and original grouping remain fixed.

Perturb before feature-utility splitting. Consequently utility training AND utility validation inherit contamination. The full outer test fold remains clean and participates only in final evaluation. It must not affect selection, IQR or frozen decisions. Persist events and hashes of clean test and contaminated training frames.

## Execution gates

1. Tests must cover frozen decisions, held-out invariance, shared/nested perturbations, contaminated utility validation, and refusal to resume after code/config drift.
2. Recompute fold 0 clean selection and predictions using identical parent seeds; require exact substantive equality with the delivered reference before any contamination.
3. Complete and independently audit all seven fold-0 conditions. Audit events, selected rows, game boundaries, contributions, paths, all final probabilities and metrics. Directly refit representative utility subsets in every condition.
4. Proceed to folds 1-4 according to compute/quota availability, without selecting folds by outcomes. Each fold must pass clean equivalence before its contamination conditions.
5. Replay fold 0 clean plus 5% draw 0 in a separate run. Compare all substantive artifacts exactly, excluding times and derived file hashes. Verify completed-run resume leaves every substantive file hash unchanged.

## Analysis and scope

Report 280 arm-condition records, deduplicated actual fit counts, per-fold/per-draw F1 and ROC-AUC losses (clean minus contaminated), original-feature Jaccard, selected-row Jaccard, actual row fraction, k and cell fraction. Average draws within fold first, then average folds equally. Keep improvements as negative losses. Fold SDs are descriptive. Shared tests, overlapping training folds and common clean references preclude treating 15 draws as independent datasets. No p-values or noninferiority claims. Timings include shared work and do not support separate per-arm speed claims.

New data are kept separate from the delivered migration and English Results chapter until full audit/replay/resume checks pass. Save durable games and a continuation record at quota exhaustion, stop processes, and wait for explicit continuation. No automated wakeup. No further experiment beyond this protocol is authorized by this stage.
