# HTRU2 fixed budget module ablation protocol

Version 1 | Frozen design dated 5 October 2026 | Development evidence

## Purpose and scope

This experiment will estimate conditional effects of variance correction, the contribution-variability penalty and the redundancy penalty at an equal retained matrix budget. It follows the completed HTRU2 development and joint-contamination stages. The previously examined outer folds cannot supply a new confirmatory test. No fitting is authorized by this protocol freeze alone.

The scientific design and JSON specification will be frozen together with hashes of the inherited data, splits and implementation. A new runner must later pass the implementation gates and receive a separate code and environment manifest before any fit. Implementation choices may not change the scientific specification. A change to the specification requires a new version and a stated reason before evaluation.

## Questions and prespecified contrasts

The primary question is whether each module changes positive-class F1 when the other modules are enabled and the sampling rule, row identities, feature count and random inputs are paired. Write a configuration as CSR: correction C, stability penalty S and redundancy penalty R, each zero or one. The primary contrasts are F1(111) minus F1(011), F1(111) minus F1(101), and F1(111) minus F1(110). Positive differences favor the enabled module in that conditional comparison. These are algorithmic interventions within this design, not population-causal effects or tests of general superiority.

Secondary outcomes are feature-set changes, contribution diagnostics, fixed-budget resampling stability, ROC-AUC, average precision, precision, recall and confusion counts. Compare 111 with a uniformly sampled four-feature subset at the same rows and feature budget. Report every factorial configuration and sampling stratum, including null and negative results. Do not select a winning stratum for the headline result.

## Frozen data and fitting conditions

Use the unchanged HTRU2 processed data and five outer splits from baseline-htru2-pilot-20261001-v1. Preserve all 17,898 records, eight original variables, labels and row identifiers. No new contamination, dataset, threshold calibration or outer split is introduced. Reuse the inherited logistic model: L2, C=1, intercept, lbfgs, max_iter=5000, tol=0.000001, no class weights, probability threshold 0.5. Fit imputation and scaling only on the rows and variables of each model's training partition. Convergence warnings are failures.

The three sampling strata are random_half, ordinary_half and robust_half, with the inherited stratified sampling implementation and rho=0.5. Set k=4 original variables for every ablation and random-feature control. The retained count is ceil(0.5*n_training), with inherited Hamilton class quotas. Report actual retained cell fraction as n_selected/n_training times 4/8; do not round it to exact 25% in source data. Sampler comparisons have equal row counts but different identities. Within a sampler and replicate, all module variants and the random-feature control share exactly the same selected rows, final model seed and one training-only redundancy matrix. Fit feature columns in their original canonical order.

The design selects rho=0.5, k=4, lambda=0.5 and eta=0.1 as explicit controlled settings, not as optima. The nonzero penalties are inherited grid values and ensure the corresponding modules can be active. The fixed four-feature budget is chosen to expose differences at a common half-feature budget. There is no inner budget search, tuning grid, full-data fallback or performance-triggered budget relaxation in this stage.

## Module definitions and shared calculations

Within each selector partition, construct B=4 stratified 80% contribution subsamples using the inherited sampling order and three utility folds. Estimate contributions using M=256 independent uniform feature permutations per game. Fit utilities with sampling inside each utility-training partition, as in the inherited implementation. Validation records remain complete. Share subsamples, utility splits, selected rows, permutation sequences and cached utility values across all eight module labels.

For feature j, let mu_j be the mean of the B estimated contributions, v_j their sample variance with denominator B-1, and a_j the mean Monte Carlo variance of the contribution mean. With C=1 use t_j=max(0,v_j-a_j); with C=0 use t_j=v_j. Record negative pre-truncation values and the number truncated. With S=1 use score_j=mu_j-0.5*sqrt(t_j); with S=0 use score_j=mu_j. Retain negative scores. Do not replace a zero variance with a fitted floor.

Use the inherited absolute Spearman association R_ij on the selected training rows. For a feature set A, the greedy objective is sum(score_j for j in A) minus eta/8 times sum(R_ij for i<j in A). Set eta=0.1 when R=1 and zero otherwise. Add the highest-increment variable at each step, resolve exact ties by original column order, and retain the first four variables even if increments are negative. Record the full eight-variable path for diagnostics.

|CSR|Variance used|lambda|eta|Role|
|---|---|---|---|---|
|111|Corrected and truncated|0.5|0.1|All enabled reference|
|011|Observed|0.5|0.1|Remove correction|
|101|Ignored|0|0.1|Remove stability penalty|
|110|Corrected and truncated|0.5|0|Remove redundancy penalty|
|001|Ignored|0|0.1|Must equal 101|
|100|Ignored|0|0|Mean contribution only|
|010|Observed|0.5|0|Uncorrected penalty without redundancy|
|000|Ignored|0|0|Must equal 100|

Eight logical labels produce at most six distinct feature paths. Preserve all labels in outputs and deduplicate identical selected-set/model fits. Assert exact equality of 001 with 101 and of 000 with 100. Equivalent labels and shared fitted models are not additional replications. If enabled modules yield the same path, record that null intervention rather than increasing their coefficients.

The equal-budget random-feature control samples four of the eight original columns uniformly without replacement with a dedicated shared seed; reuse the same feature draw across the three sampling strata for a fold and algorithm seed. Additional context arms are each half-row sampler with all eight features, and all rows with all eight features. These context arms have different matrix budgets and are excluded from claims about equal-budget module effects.

## Predictive and stability schedules

Predictive schedule: use each of five frozen outer folds with three root seeds, 2026100501, 2026100502 and 2026100503. Each fold-seed pair evaluates all three samplers. This gives 45 selector instances, 180 contribution games, 360 logical ablation records, 45 random-feature records, 45 half-row/all-feature records and 15 full-data records, or 465 logical records. At most 375 distinct final fits are needed before further deduplication. Save all selector outputs and selected identities before fitting or scoring outer predictions. No outer outcomes may enter contributions, redundancy or selection.

Stability schedule: use only the original outer-fold-0 training pool. Generate ten stratified 80% resamples using roots 2026100510 through 2026100519, with ceil(0.8*n_pool) and inherited Hamilton quotas. For each resample, rebuild all three sampling strata, B contribution games and all feature paths, using algorithm root 2026100501. This gives 30 selector instances and 120 games. Save 240 logical ablation selections and 30 random-feature selections; no final classifier or outer-test prediction is made. Recompute random feature draws per resample and share each draw across samplers. This measures fixed-budget selection stability, not the previous complete-workflow stability that repeated inner budget selection. Resampling also changes stochastic inputs through the coordinate key, so this is not a decomposition of data and algorithm randomness.

Top-level seeds use UTF-8 canonical JSON arrays with no whitespace, SHA256, first four digest bytes interpreted big-endian, modulo 2147483647. Use coordinates [protocol_id, phase, root_seed, fold_or_resample, sampler_or_shared, role, index]. Roles are data_resample, final_rows, contribution_subsample, game, permutation, final_model and random_features. The index is contribution replicate b=0,1,2,3 for contribution_subsample, game and permutation; otherwise it is zero. Separate b values must use separate permutation streams; do not share permutations across contribution replicates. Contribution subsample and game keys use sampler_or_shared="shared", preserving common contribution pools and utility split seeds across samplers; final_rows and permutation keys use the sampler name. Final model and random-feature keys use "shared". Data resamples use their dedicated root and "shared". Main phases are "predictive" and "stability"; replays reuse the original phase and all original keys, in a separate run directory.

In top-level JSON keys, root_seed, fold_or_resample and index are integers; all other fields are strings. Predictive fold indices are 0 through 4 and stability resample indices are 0 through 9. For stability data_resample use that resample's dedicated root; for all subsequent stability keys use algorithm root 2026100501. For predictive keys use the corresponding predictive root. These types and coordinate assignments are part of the seed contract.

Within each game, retain the existing FeatureUtility seed_for derivation: join the tuple fields with a literal vertical bar, encode UTF-8, take the first four SHA256 digest bytes as an unsigned big-endian integer without a modulo. Derive utility split from [game_seed,"utility_split"], utility sampling from [game_seed,utility_fold,"utility_sampler"], and utility model from [game_seed,utility_fold,"utility_model"]. Enumerate the ledger before fitting, record both key formats, and reject collisions between distinct keys within a derivation namespace. Module flags never enter a shared key. Preserve group-disjoint utility folds; stop if the inherited singleton feature-fingerprint groups have changed rather than silently treating a grouped dataset as independent rows.

## Outcomes and reporting rules

For every sampler, compute paired F1 contrasts within each outer fold and seed, average the three seeds within each fold, then average the five folds equally. Report the five fold-level mean differences, their mean, descriptive SD, range and positive/zero/negative counts using a 1e-12 zero tolerance. Report individual-seed results in source data. A cross-sampler descriptive mean may be shown only with equal weights over the three prespecified strata and alongside all stratum results. The five folds and fifteen fold-seed values are dependent, not independent datasets; do not compute a superiority p-value, confidence interval or noninferiority claim from them.

Report absolute F1 as well as paired contrasts. A loss greater than 0.01 relative to the matched full-data arm is a descriptive warning count, not a selection constraint or a success criterion. Secondary metrics must not replace F1 as the primary outcome after inspection. Do not omit failed configurations or replace seeds. A numerical failure blocks a complete summary until resolved under a recorded implementation correction or explicitly reported as an incomplete run.

For stability, compute Nogueira stability, feature frequencies, selected-set sizes and mean pairwise Jaccard separately for each sampler and each logical variant across the ten resamples. The 45 pairwise similarities are dependent. Report differences from 111 descriptively; do not compare this fixed-k stability directly as an improvement over the historical adaptive-k experiment. Report feature-set Jaccard between paired module variants, path equality, corrected-variance clipping counts and changes of selected features in both schedules.

Enumerate all 256 utility subsets in each eight-variable game as an exact diagnostic reference. Exact contributions must not replace permutation estimates or set feature paths for evaluation. Report contribution RMSE, maximum absolute error and exact-prefix agreement at k=4 without adjusting M. These checks evaluate the current estimator and do not make this procedure a high-dimensional benchmark.

## Computation budget and stopping rules

Each game requires at most 255 nonempty subsets times three utility folds, or 765 logistic fits, plus three constant evaluations. Shared game caches are reused by all module labels. Counts below are upper bounds before final-model deduplication and exclude no hidden audit work.

|Work package|Games|Maximum logistic fits|
|---|---|---|
|Predictive main schedule|180|137700 utility plus 375 final|
|Stability main schedule|120|91800 utility|
|Independent predictive replay fold 0 seed 2026100501|12|9180 utility plus 25 final|
|Independent stability replay resample 0|12|9180 utility|
|Audits and implementation validation|0 new games|10000 maximum|
|Total planned upper bound|324|258260|

The all-stage hard cap is 260000 logistic-fit attempts, including failed attempts, independent replay and audit work. Main and replay games additionally require 972 constant evaluations. The 1740-fit difference between the planned bound and cap is contingency, not authority to add experimental conditions. Reserve at least 300 audit utility refits, one nonempty subset-fold from every main game, plus reconstruction of every final prediction from stored coefficients and preprocessing. Keep the remaining audit reserve for boundary, formula and identity checks. Stop before any operation that would exceed the cap; changing the cap requires a new protocol version.

Use one numerical thread per process and at most two concurrent workers. The classifier-fit cap excludes KMeans and encoder fits; count those operations separately and record their wall times along with selection and diagnostic work. Do not infer per-arm speedups from shared caches. There is no promised wall-clock duration. Checkpoint each completed contribution game and final arm atomically. At quota exhaustion or explicit stop, finish only the current atomic operation, save hashes and counters, stop all workers, and wait for explicit user continuation. Resume requires unchanged design, code, configuration, data, environment and completed-output hashes. No scheduled wakeup is allowed.

## Acceptance gates and delivery

Before the main schedule, verify all inherited hashes; validate the JSON counts and seed ledger; audit disjoint train, validation and test identities; and run deterministic analytic tests for all module formulas, tie-breaking, negative increments and the two required label equivalences. Validate the all-enabled adapter against inherited corrected_scores and forward_path on saved arrays. Audit the first predictive fold-seed package before processing the remaining packages. These gates validate implementation, not whether the result is favorable.

After execution, require complete logical records, exact paired row budgets, independent refits, prediction reconstruction, exact-reference diagnostics, the two prespecified separate replays and an unchanged-resume check. Preserve selected rows and features, contribution estimates, permutations, raw and corrected variance, paths, coefficients, probabilities, metrics, fit counters, failures and environment metadata. Produce audited per-unit CSVs, descriptive summaries, an English report, numeric checks, individually inspected rendered pages and a delivery hash manifest. A result may pass delivery with zero or negative module effects; it may not pass with missing verification or silent parameter changes.

The frozen design has no experimental results. The next authorized execution should implement and verify this protocol in a separate run, ablation-htru2-fixedbudget-20261005-v1. If implementation changes any scientific choice above, retain this design and create version 2 rather than overwrite it. Cross-dataset validation, contamination ablations, adaptive budget retuning and isolated end-to-end cost comparisons remain outside this stage.

