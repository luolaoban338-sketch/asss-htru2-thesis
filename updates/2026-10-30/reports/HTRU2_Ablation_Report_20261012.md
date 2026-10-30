> Public research copy prepared on 2026-10-30. This is a dated report, not a current stage-status statement. Operational account-use material was omitted where present; scientific results and audit limitations are retained. See the publication manifest for the original source hash and removed source-line ranges.

# HTRU2 Fixed Budget Ablation Report

English research report | 12 October 2026

The fixed-budget HTRU2 ablation found no change in the final four-feature set or predictive F1 after the prespecified module removals. Some full selection paths changed, and contribution estimates did not consistently recover the exact ordered prefix. This report explains those distinct outcomes and the limits of their interpretation.

## Design and predictive performance

The experiment used the unchanged 17,898-record HTRU2 development dataset with eight numerical predictors [1]. Five frozen outer folds and three algorithm seeds produced 15 predictive packages. Within each of three sampling strata, all module variants shared selected rows, utility inputs, permutation sequences, final model seeds and a training-only redundancy matrix. The row fraction was fixed at rho=0.5 and the feature budget at k=4. There was no inner budget search or full-data fallback.

Four contribution subsamples (B=4) used M=256 permutations per utility game. Enabled penalties were lambda=0.5 for contribution variability and eta=0.1 for redundancy. The inherited logistic-regression model and probability threshold of 0.5 were unchanged; preprocessing was fitted on each model's training partition. The eight logical CSR labels, a random-four-feature control and a half-row all-feature arm were evaluated within each sampler, together with a full-data reference.

Table 1. Predictive performance and retained matrix size

|Sampling|Feature rule|F1|Fold SD|Cells %|Warnings|
|---|---|---|---|---|---|
|Random half|All module variants|0.8601|0.0102|25.001|11/15|
|Random half|Random four|0.8299|0.0308|25.001|12/15|
|Random half|All eight|0.8707|0.0088|50.001|0/15|
|Ordinary half|All module variants|0.8592|0.0101|25.001|9/15|
|Ordinary half|Random four|0.8289|0.0312|25.001|11/15|
|Ordinary half|All eight|0.8716|0.0123|50.001|0/15|
|Robust half|All module variants|0.8594|0.0111|25.001|10/15|
|Robust half|Random four|0.8296|0.0317|25.001|12/15|
|Robust half|All eight|0.8711|0.0101|50.001|0/15|
|Full rows|All eight|0.8743|0.0091|100.000|0/15|

Each F1 first averages three seeds within each outer fold, then weights five folds equally. SD is the sample SD of the five fold means. Warnings count fold-seed losses greater than 0.01 against the paired full-data arm; these 15 units are dependent. The threshold is descriptive and was not a selection constraint. All eight logical module configurations share the reported values within a sampler. Actual mean matrix fractions are approximately 25.0007% for four features and 50.0014% for all eight, because row counts are rounded upward.

<!--page-->

## Module contrasts were zero at the selected budget

All 135 primary seed-level F1 contrasts were exactly zero: three module removals across three samplers, five folds and three seeds. All eight logical configurations selected the same four columns in each predictive comparison, so final-model caching did not conceal distinct selected models. Table 2 reports the full logical factorial result, including the prescribed aliases.

Table 2. Every logical module configuration at the fixed budget

|CSR|Random F1|Ordinary F1|Robust F1|
|---|---|---|---|
|111|0.8601|0.8592|0.8594|
|011|0.8601|0.8592|0.8594|
|101|0.8601|0.8592|0.8594|
|110|0.8601|0.8592|0.8594|
|001|0.8601|0.8592|0.8594|
|100|0.8601|0.8592|0.8594|
|010|0.8601|0.8592|0.8594|
|000|0.8601|0.8592|0.8594|

CSR denotes variance correction C, contribution-variability penalty S and redundancy penalty R. The exact implementation aliases 001=101 and 000=100 are retained for completeness; they are not additional independent evidence. Each primary contrast 111 minus 011, 101 or 110 was exactly zero in all 15 fold-seed units within every sampler.

The all-enabled configuration had a higher average F1 than the paired random-four control in every sampling stratum (Table 3). However, one fold favored the control under random-half sampling and one under robust-half sampling. The comparator demonstrates that an equal feature count does not by itself reproduce the observed scoring-rule outcome. Because every scoring variant produced the same set, this comparison cannot attribute the average difference to correction, the variability penalty or redundancy.

Table 3. Paired F1 differences from the random four feature control

|Sampling|Mean difference|Fold SD|Minimum|Maximum|Positive folds|
|---|---|---|---|---|---|
|Random half|0.0302|0.0236|-0.0024|0.0554|4/5|
|Ordinary half|0.0303|0.0236|0.0007|0.0568|5/5|
|Robust half|0.0297|0.0241|-0.0024|0.0540|4/5|

Differences are all-enabled 111 minus random_four, with seeds averaged within folds before aggregation. Negative minimum values occurred in fold 2 for random-half and robust-half sampling. These descriptive contrasts do not identify an incremental benefit of any module or establish statistical superiority.

<!--page-->

## Stable selected sets described the fixed budget only

All eight module variants selected Profile_mean, Profile_stdev, Profile_skewness and Profile_kurtosis in every fixed-budget stability resample and sampler. Their Nogueira stability and mean pairwise Jaccard similarity were both 1.0000 (Table 4). The four remaining local columns had selection frequency zero. Column names identify the stored data schema; the documented descriptor-order discrepancy limits their physical interpretation.

Table 4. Fixed budget feature selection stability

|Sampling|Feature rule|Nogueira|Mean Jaccard|Mean k|
|---|---|---|---|---|
|Random half|All module variants|1.0000|1.0000|4.0|
|Random half|Random four|0.0667|0.3833|4.0|
|Ordinary half|All module variants|1.0000|1.0000|4.0|
|Ordinary half|Random four|0.0667|0.3833|4.0|
|Robust half|All module variants|1.0000|1.0000|4.0|
|Robust half|Random four|0.0667|0.3833|4.0|

Each row describes ten stratified 80% resamples of outer-fold-0 training data. Jaccard averages 45 dependent pairs. Every one of the eight logical module variants has the same stability summary in each sampler. Shared random feature draws give identical control summaries across samplers; these are paired controls, not three independent replications.

Each resample contained 11,455 rows from the outer-fold-0 training pool, with 5,728 rows retained by each half-row selector. Selection and contributions were rebuilt at the frozen budgets. No outer-test record entered the stability pool, and no final predictive classifier or outer-test prediction was produced by these stability packages.

This experiment differs from the earlier complete-workflow stability evaluation, which repeated inner budget selection and final feature construction. That earlier joint rule had Nogueira stability 0.4974 and mean Jaccard 0.7352. Fixed-budget resampling removes variation from budget search and asks a narrower question. Its perfect observed set agreement does not supersede the earlier result or establish a stability advantage for the penalty. The data resamples overlap, and data and algorithm randomness are not separately identified [2].

<!--page-->

## Paths and approximation retained measurable variation

The absence of selected-set changes did not imply that every intermediate ranking or full greedy path was invariant. Table 5 separates those quantities for each primary removal, while Table 6 compares the estimated paths with exact diagnostics.

Table 5. Module removals could alter paths without altering selected sets

|Phase|Comparison|Set changes|Full path changes|
|---|---|---|---|
|Predictive|111 vs 011|0/45|7/45|
|Predictive|111 vs 101|0/45|12/45|
|Predictive|111 vs 110|0/45|3/45|
|Stability|111 vs 011|0/30|5/30|
|Stability|111 vs 101|0/30|8/30|
|Stability|111 vs 110|0/30|1/30|

A full path is the ordered greedy sequence across all eight variables. A selected set is the unordered first four variables. Predictive totals are 0/135 set changes and 22/135 full-path changes; stability totals are 0/90 and 14/90. Equal selected sets therefore do not imply equal complete paths.

Table 6. Contribution approximation and ordered prefix diagnostics

|Phase|Games|Mean RMSE|Max error|Exact prefix|Clipped variance|
|---|---|---|---|---|---|
|Predictive|180|0.012148|0.057886|194/360|213/360|
|Stability|120|0.012635|0.059218|148/240|132/240|

RMSE compares M=256 permutation estimates with exact contributions within each utility game; the reported mean averages games. Exact-prefix agreement compares the ordered first four entries of estimated and exact greedy paths across logical module labels, including aliases. It is not selected-set agreement. Clipped-variance counts cover selector-feature entries once per selector; negative corrected variance was truncated at zero. These denominators are computational diagnostics, not independent statistical samples.

Exact contributions were retained as diagnostics and did not replace the prescribed permutation estimates used for selection. Variance clipping likewise describes an estimator operation; it is not evidence that a variable is intrinsically stable. The observed ordered-prefix discrepancies leave approximation sensitivity unresolved outside the selected-set result.

<!--page-->

## Verification and interpretation

The completed main schedule contained 25 independently audited packages, 465 predictive records, 270 stability selections and 300 contribution games. The separate predictive and stability replays matched 28 and 18 substantive artifacts, respectively. The aggregate audit independently recomputed the saved summaries with tolerance 1e-12, including 31 absolute summaries, 180 seed contrasts, 12 fold-aggregate contrasts, 27 stability summaries, 216 feature frequencies and 1,215 dependent Jaccard pair values.

The 11 October unchanged-resume check visited all 25 main and two replay packages through their cache branch and added zero operations. It preserved 2,049 run artifacts, including 2,010 JSON files, 38 frozen code copies and the budget database. Before document writing, these artifacts and earlier deliveries were verified again. The logistic ledger remained at 248,344 successful attempts under the 260,000 cap; encoder and KMeans counts were 250,070 and 2,808. These totals include the ablation stage's audits and replays, and no failed or unfinished operations were recorded. Document preparation added no fits.

The results support a local conclusion about this fixed configuration: the tested module interventions did not change the final feature set or F1. They do not establish equivalence, noninferiority, universal absence of module effects, or robustness to corruption. The lower absolute F1 and repeated descriptive loss warnings relative to full data remain relevant even when the selected set is perfectly stable. No budget, coefficient or threshold was retuned after these outcomes.

These previously examined development folds provide neither an independent confirmatory dataset nor a test of source-level astronomical generalization. Eight-variable exact diagnostics do not establish high-dimensional scalability. Fit counts document computational work but do not establish a per-rule runtime advantage because caches and shared utility evaluations are part of the implementation.

The source tables are retained in qa/ablation_20261011/full_snapshot. The immutable run identifier is ablation-htru2-fixedbudget-20261005-v1. Detailed report-cell checks and page inspections are recorded separately from the experimental evidence.

# References

[1] Lyon, R. HTRU2 [Dataset]. UCI Machine Learning Repository (2015). https://doi.org/10.24432/C5DK6R. Dataset documentation: https://archive.ics.uci.edu/dataset/372/htru2 (accessed 12 October 2026).

[2] Nogueira, S., Sechidis, K. and Brown, G. On the Stability of Feature Selection Algorithms. Journal of Machine Learning Research 18(174), 1-54 (2018). https://jmlr.org/papers/v18/17-514.html (accessed 12 October 2026).
