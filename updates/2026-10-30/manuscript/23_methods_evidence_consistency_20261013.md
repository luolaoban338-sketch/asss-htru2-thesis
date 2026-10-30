# HTRU2 Methods and Evidence Consistency Review

English research report and Methods companion | 13 October 2026

The reviewed Results and Discussion remain consistent with the saved development evidence. Four Methods clarifications are supplied below to complete their procedural coverage. The fixed-budget findings do not establish an advantage for any individual scoring module.

## Review decision and scope

The review compared the original Methods, the two reports delivered on 12 October, frozen protocols, execution code and audited summaries. It covered 24 methodological and interpretation checks. No unresolved numerical contradiction or unsupported affirmative claim was identified within those materials. This is a source-consistency review, not a new independent experimental audit.

The original Methods describes adaptive joint selection, sampling-only contamination and complete-workflow stability [S1]. The latest chapters also contain joint-process contamination and fixed-budget ablation [S2, S3]. The Methods companion on pages 3 to 5 supplies those missing descriptions. It is intended to accompany the original Methods when assembling the thesis.

Table 1. Documentation findings and their resolution

|Finding|Why the distinction matters|Resolution in this companion|
|---|---|---|
|Fixed-budget ablation omitted|The earlier inner search and fallback do not apply to this stage.|Fixed rho and k, module definitions, controls and schedules are specified.|
|Joint contamination omitted|The earlier 140-fit sampling stress test does not describe the joint experiment.|Clean decisions stay fixed; selected identities are rebuilt after perturbation.|
|Exact-prefix convention implicit|The comparison does not remove every estimated quantity.|Exact contributions retain the estimated Monte Carlo variance in this diagnostic.|
|Concurrency scope implicit|The original four-task allowance differs from the ablation limit.|The ablation permits two workers, with one numeric thread each.|

All four findings are resolved in the companion text. The original Methods and the two 12 October documents remain preserved as dated evidence. No experiment code, budget, coefficient, prediction threshold or result was changed.

## Terms used throughout the review

Complete-workflow stability repeats inner budget selection and final feature construction. Fixed-budget stability rebuilds selection while holding rho=0.5 and k=4 constant. The contribution-variability penalty is the S module; its name does not assert stable selected sets. A selected set is unordered, whereas an ordered prefix retains feature order.

<!--page-->

## Findings that constrain the thesis conclusions

Table 2. Evidence and permissible interpretation

|Evidence checked|Verified observation|Interpretation retained|
|---|---|---|
|Primary module contrasts|All 135 predictive F1 differences are zero.|Local invariance at the tested budgets; no equivalence or general no-effect claim.|
|Fixed-budget stability|All eight variants have Nogueira=1 and mean Jaccard=1 within every sampler.|Perfect observed agreement across these ten resamples, without a demonstrated penalty benefit.|
|Complete-workflow stability|Earlier joint Nogueira=0.4974 and mean Jaccard=0.7352.|A distinct adaptive procedure; its variation is not superseded by fixed-budget stability.|
|Full path changes|22/135 predictive and 14/90 stability primary comparisons change paths; none change selected sets.|Set invariance does not imply identical ordered paths.|
|Random-four control|One fold favors the control under random-half sampling and one under robust-half sampling.|Higher average F1 does not imply an advantage in every fold or an individual module benefit.|
|Loss warnings|All-enabled counts are 11/15, 9/15 and 10/15 for random, ordinary and robust sampling.|Descriptive losses above 0.01 remain visible; no budget relaxation or noninferiority claim.|
|Approximation diagnostics|Exact ordered prefixes agree in 194/360 predictive and 148/240 stability label comparisons.|These are ordered-prefix diagnostics, not selected-set agreement rates or independent samples.|
|Computational evidence|248,344 successful logistic attempts against a 260,000 cap.|Counts include ablation audits and replays; they do not establish end-to-end speedup.|

Sources are the dated ablation snapshot and its independent aggregate audit [S5], the earlier stability summary [S6], and the unchanged ledger [S7]. Five outer folds, repeated seeds and overlapping resamples provide dependent development evidence. They do not provide independent datasets for confirmatory inference.

The reviewed chapters retain negative control differences, prediction-loss warnings, approximation error, source-identity limitations and the descriptor-order conflict. Proposed explanations for invariant sets remain hypotheses. Neither robustness to corruption nor high-dimensional scalability is established by the fixed-budget experiment.

The terminology scan flagged two names for the S penalty, intentional adaptive/fixed-budget distinctions and contextual numeric formatting. Each flag was reviewed against its surrounding text. The companion adopts the precise penalty name and preserves the distinction between the experimental designs.

<!--page-->

## Methods companion for the fixed budget design

### Design and shared inputs

The subsequent ablation used the unchanged HTRU2 data and five outer folds from the earlier development evaluation. Within each fold, three algorithm seeds evaluated random-half, ordinary-half and robust-half sampling. The root seeds were 2026100501, 2026100502 and 2026100503. These previously examined folds supplied development evidence only.

The requested row fraction was rho=0.5, and each ablation or random-feature arm retained k=4 of eight variables. Row counts used ceil(0.5 times training size), with the inherited stratified allocation. Actual matrix fraction was selected rows divided by training rows, multiplied by 4/8. This stage had no inner budget search, full-data fallback or outcome-dependent budget relaxation [S3].

All module variants within a sampler shared selected rows, contribution inputs, permutation sequences, final-model seeds and one training-only redundancy matrix. Contribution subsamples and utility split seeds were also shared across samplers. Permutation streams differed across contribution replicates and samplers. Each random-four control used a shared feature draw across samplers within the same fold and algorithm seed.

The inherited logistic model used L2 regularization, C=1, an intercept, lbfgs, at most 5,000 iterations and tolerance 10^-6, without class weights. Positive predictions required probability at least 0.5. Imputation and scaling were fitted on each model's selected training rows and variables. Convergence warnings invalidated the affected fit. Each utility game used three validation folds, retaining complete validation partitions.

### Contributions and module interventions

Four stratified 80% contribution subsamples were drawn from the current selector partition. Each game used M=256 independent uniform feature permutations. Row selection occurred inside each utility-training fold. Mean positive-class F1 across its three validation folds defined the subset utility. The empty subset used the selected training labels' positive-class proportion as a constant probability.

Let mu_j denote the mean estimated contribution across the four subsamples. Let v_j denote their sample variance, and a_j the mean estimated Monte Carlo variance of the contribution mean. For correction C=1, t_j=max(0, v_j-a_j); for C=0, t_j=v_j. With penalty S=1, score_j=mu_j-0.5 sqrt(t_j); otherwise, score_j=mu_j. Negative corrected variances were recorded before truncation.

The greedy objective summed selected scores and subtracted eta/8 times the sum of pairwise absolute Spearman associations. Eta was 0.1 for redundancy R=1 and zero otherwise. Ties followed original variable order, and negative increments did not terminate the path. Selection retained the first four variables; the complete eight-variable path was saved [S4].

<!--page-->

## Methods companion for controls and stability

### Logical comparisons and diagnostic conventions

The eight CSR labels were 111, 011, 101, 110, 001, 100, 010 and 000. Correction has no effect when S=0, giving exact aliases 001=101 and 000=100. Both aliases remained in logical outputs; identical selected models were cached. Final feature columns followed canonical data order. All selector records were saved before outer-model fitting.

Primary contrasts were F1(111)-F1(011), F1(111)-F1(101) and F1(111)-F1(110), computed within paired sampler, fold and seed units. The equal-budget control selected four variables uniformly without replacement. Half-row all-feature arms and a full-row all-feature reference provided context at different matrix budgets. Five folds and three seeds produced 465 logical predictive records [S3, S5].

Exact Shapley values from all 256 subsets were retained for diagnostics. They did not replace the permutation estimates used for selection. The exact-prefix diagnostic substituted exact contributions while retaining the same estimated Monte Carlo variance array. It recomputed scores and paths under that convention, then compared the ordered first four entries. It therefore is not a fully noise-free selection oracle [S4].

### Fixed budget and complete workflow stability

Fixed-budget stability used ten stratified 80% resamples from outer-fold-0 training data. Resample roots were 2026100510 through 2026100519, with algorithm root 2026100501. The coordinate-derived stochastic inputs changed across resamples. Each resample contained 11,455 rows, and each half-row selector retained 5,728 rows. Outer test records were excluded [S3, S5].

Every resample rebuilt row selection, four contribution games per sampler, redundancy and all module paths at the fixed budgets. Random feature draws were refreshed between resamples and shared across samplers. The schedule yielded 240 module selections and 30 control selections. These packages fitted utility classifiers but no final predictive classifier, and generated no outer-test predictions.

The earlier complete-workflow experiment instead repeated inner budget selection before rebuilding the final selectors. Its outputs can vary through sampling-rule and feature-budget decisions as well as feature construction. The two experiments therefore measure different procedures. Data resampling and algorithmic randomness were not separately identified in either design [S1, S6].

Both designs report Nogueira stability, feature frequencies and mean pairwise Jaccard similarity. Ten resamples yield 45 dependent pairs for each summary. All-full or all-empty selections have undefined Nogueira values. The overlapping resamples do not support population-stability or independent-replication claims.

<!--page-->

## Methods companion for contamination and interpretation

### Joint process contamination with frozen decisions

The joint contamination experiment froze each clean fold's sampling rule, row fraction, penalty coefficients and feature count. It rebuilt selected row identities, contributions, redundancy and feature identities without repeating inner budget search. This distinguishes it from the earlier sampling-only stress test [S1, S8].

Each outer fold included one clean condition and three draws at each of 5% and 10% contamination. Each affected row received a randomly signed ten-IQR offset in one eligible numerical predictor. IQR values came from clean outer training data, and each 5% event set was nested within its corresponding 10% set. All arms shared the same perturbed matrix.

Perturbation occurred before utility splitting, so utility-training and utility-validation data could both contain corrupted values. Labels and outer test records remained unchanged. Five folds and seven conditions produced 35 conditions and 280 arm-condition records. Contaminated summaries averaged draws within folds before equally weighting folds. Loss was clean minus contaminated F1, retaining negative values for improvements [S8].

### Aggregation and reproducibility

Ablation summaries averaged three seeds within each outer fold and then weighted five folds equally. Fold standard deviations were descriptive. An F1 loss above 0.01 against paired full data generated a warning only. This warning differed from the earlier adaptive search's inner feasibility constraint. Neither rule guaranteed outer noninferiority.

The ablation permitted at most two workers, with one numeric thread per worker [S3]. Separate run identifiers preserved manifests, selected identities, contributions, predictions and audit evidence. Its ledger includes main execution, audits and replays; earlier stage counts may exclude separate audits and replays. Shared caches prevent interpreting these totals as isolated per-rule runtime comparisons.

Reproducibility checks support the recorded computation within its development scope. They do not establish general superiority, equivalence, robustness, scalability or an end-to-end speed advantage. The absence of astronomical source identities and the descriptor-order discrepancy continue to limit source-level and physical interpretation.

### Source index and completion boundary

S1 Original Methods and complete-selection protocol: docs/10_htru2_methods_en.md and docs/11_complete_selection_stability_protocol.md. S2 Reviewed chapters: docs/20_ablation_report_en_20261012.md through docs/22_htru2_discussion_en_20261012.md.

S3 Frozen ablation protocol and config: protocols/ablation-htru2-fixedbudget-v1-20261005/. S4 Implementation: scripts/run_ablation.py; experiments/ablation.py, shapley.py, feature_path.py and feature_utility.py.

S5 Audited snapshot: qa/ablation_20261011/full_snapshot/ and independent_aggregate_checks.json. S6 Earlier complete-workflow summary: runs/stability-htru2-pilot-20261001-v1/summary.json. S7 Ablation ledger: runs/ablation-htru2-fixedbudget-20261005-v1/fit_budget.sqlite3.

S8 Joint contamination protocol and runner: docs/14_joint_contamination_protocol.md and scripts/run_joint_contamination.py. All paths are relative to the active HTRU2 workspace. The 13 October review is complete; final stage archive and delivery remain a separate 14 October scope.
