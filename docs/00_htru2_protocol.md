# HTRU2 migration protocol

Approved by the user on 1 October 2026. Frozen before any HTRU2 model fitting. This is a development replication of previously completed stages, not a confirmatory study.

## Data and schema

Use all 17,898 rows in the official UCI HTRU2 archive, with class 1 as positive and the eight columns in original order. Preserve the ARFF names. The Readme reverses the textual order of skewness and kurtosis; record the conflict and avoid individual physical interpretations until verified. No label-based filtering, outlier deletion, feature screening or balancing of validation/test data. SHA256 of the eight predictor values defines duplicate groups, not astronomical identities. There are no exact duplicate vectors. Source identities and observation metadata are unavailable; source-level independence is unverified.

Raw data, processed data, field whitelist, checksums and source attribution are in data/raw and data/audit. All predictor preprocessing, sample selection, contribution estimation, redundancy construction and parameter selection are restricted to the corresponding training partition.

## Frozen methods

Retain the existing classifier, threshold 0.5, four sampling arms, five outer folds, three inner folds, three utility folds and role-specific seeds. Numerical settings are in the five configs/*_pilot_v1.json files. Seeds retain the inherited stage roots, with the dataset name changed to htru2. Full data, random half, ordinary geometry half and robust geometry half are compared on paired folds.

Primary metric: positive-class F1. Also save precision, recall, ROC-AUC, average precision (AP, a stepwise precision-recall summary, not trapezoidal PR-AUC), confusion counts and actual compression. Undefined metrics are explicitly null. No threshold or class-weight tuning. Inner F1 tolerance epsilon=0.01; minimize actual row fraction times k/8, then k, then maximize validation F1, then fixed candidate order. Retain the full-data fallback. No changes based on outer scores.

Baseline: 20 fits. Sample-selector contamination: 140 fits, with 0%, 5%, 10% training-row perturbations; three draws per nonzero rate, fixed IQR offset 10, unchanged test data. This tests specified numerical stress only.

Feature pilot: B=4, 80% training subsamples, M=16, four sampling arms, all 256 feature subsets, 16 games, 12,240 nonempty-subset fits and 48 constant evaluations. Independently check all 8! permutations and direct utility refits. Prespecified descriptive M sweep: 16/64/256/1024 using saved utilities, no additional classifier fitting and no budget adaptation.

Joint validation: B=4, M=256, lambda={0,0.5}, eta={0,0.1}, k=1..8, all four sampling conditions, 128 candidates per inner partition. Five outer folds; eight final comparison arms. Full exact references are diagnostic only; selection uses permutation estimates. All inner decisions are committed before outer evaluation. Shared cached computation does not measure separate per-arm wall times.

Stability: ten complete 80% class-stratified resamples of outer-fold-0 training rows only. Each reruns inner selection and all final feature construction. The outer test pool is physically excluded from the engine. Save Nogueira, Jaccard, feature frequency, selected sets and budgets. All-full/all-empty Nogueira is undefined. Overlapping resamples and pairwise similarities are not independent experimental units.

## Validation and scope

Replay baseline fold 0 (4 fits), contamination fold 0 (28 fits), feature resample 0 (4 games), joint outer fold 0 and stability replicate 0. Audit all main results and reconstruct stored predictions. Verify completed runs resume without changing substantive artifact hashes. Save per-command elapsed time separately from component times.

Inherited generic tests must pass; new tests check HTRU2 schema, raw/processed agreement and AP. New reports must be derived solely from HTRU2 outputs. Old Adult/WDBC results remain separate; COMPAS is archived in the historical project and excluded from the replacement evidence.

The approved scope ends with completion and report QA of these stages. It excludes previously unexecuted joint-process contamination, new datasets and confirmatory final thesis experiments. Word reports require rendered page-by-page visual review before final delivery. Save a checkpoint before quota exhaustion; stop when exhausted and wait for explicit continuation. No scheduled or automatic wakeup.

Source: https://archive.ics.uci.edu/dataset/372/htru2 ; DOI https://doi.org/10.24432/C5DK6R . UCI lists CC BY 4.0; retain attribution and document processing changes.
