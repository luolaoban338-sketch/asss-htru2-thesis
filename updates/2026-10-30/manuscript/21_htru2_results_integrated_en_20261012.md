# Results

The development evaluation used 17,898 HTRU2 records and eight numerical predictors, including 1,639 positive-class records (9.16%) [1]. Local checks found no missing predictor values or duplicate predictor vectors. Predictive summaries are equal-weight means across five frozen outer folds; fold standard deviations are descriptive, not standard errors. All comparisons used the same logistic-regression settings and probability threshold of 0.5. F1 denotes positive-class F1, and losses below are reference F1 minus evaluated F1.

## Joint selection compressed the training matrix with an inconsistent F1 trade off

Joint selection retained 35.00% of the training matrix cells and achieved mean F1 of 0.8638, compared with 0.8743 for full data (Table 1). Its mean paired difference from full data was -0.0105. It retained half of the training rows and an average of 5.6 variables. The prespecified loss tolerance of 0.01 was exceeded in two of five outer folds. Compression therefore did not consistently preserve the intended held-out F1 tolerance.

Table 1. Predictive performance and compression across five outer folds.

|Selection rule|Mean F1|F1 SD|Mean k|Cells (%)|Loss >0.01|
|---|---|---|---|---|---|
|Full data|0.8743|0.0091|8.00|100.00|0/5|
|Sample selection only|0.8732|0.0121|8.00|50.00|0/5|
|Feature selection only|0.8663|0.0091|5.60|70.00|1/5|
|Joint selection|0.8638|0.0052|5.60|35.00|2/5|
|Sequential selection|0.8653|0.0094|6.00|37.50|1/5|
|Joint: random candidates|0.8667|0.0093|5.80|36.25|1/5|
|Joint: ordinary candidates|0.8646|0.0039|6.00|37.50|2/5|
|Joint: robust candidates|0.8635|0.0056|6.00|37.50|2/5|

Cells denotes the retained row fraction multiplied by the retained fraction of original variables; k is the retained variable count. Loss is the paired full-data F1 minus the selection-rule F1. Restricted joint rules limit the available sampling candidates while retaining the full-data fallback. All rules share the same inner candidate evaluations.

Joint selection used ordinary geometry sampling in folds 0 and 1, random sampling in fold 2, and robust geometry sampling in folds 3 and 4. Its paired F1 differences from full data were -0.0093, -0.0142, -0.0021, -0.0063 and -0.0208. Sequential selection retained 37.50% of cells with mean F1 of 0.8653 and one tolerance violation. Joint selection retained slightly fewer cells but had lower mean F1; these comparisons do not establish noninferiority or a general advantage.

In the separate sampling baseline, random-half, ordinary geometry-half and robust geometry-half selection achieved mean F1 of 0.8732, 0.8739 and 0.8695. None exceeded the 0.01 loss tolerance in an outer fold. That baseline used its own stage-specific seeds: its random-half arm is distinct from the sample-only decision in Table 1 despite their equal rounded means. Adding feature selection changed the trade-off, but this comparison does not isolate a module effect.

## Joint selection was less stable under complete workflow resampling

Ten complete workflows on stratified 80% resamples of the first outer training pool repeated inner budget selection and final feature construction. Joint selection had Nogueira stability of 0.4974 and mean pairwise Jaccard similarity of 0.7352, both lower than sequential selection at 0.6549 and 0.8519 (Table 2). Outer test records were excluded.

Table 2. Feature-selection stability across ten complete workflows.

|Selection rule|Nogueira|Jaccard|Mean k|Cells (%)|
|---|---|---|---|---|
|Full data|Undefined|1.0000|8.00|100.00|
|Sample selection only|Undefined|1.0000|8.00|50.00|
|Feature selection only|0.5838|0.7537|5.30|66.25|
|Joint selection|0.4974|0.7352|5.60|36.25|
|Sequential selection|0.6549|0.8519|6.10|38.13|
|Joint: random candidates|0.4763|0.7543|5.90|38.13|
|Joint: ordinary candidates|0.4902|0.7905|6.20|40.00|
|Joint: robust candidates|0.3941|0.7687|6.30|40.63|

Jaccard values average 45 dependent pairwise comparisons. Nogueira stability is undefined for the full-data and sample-only rules because all eight variables are retained in every workflow. A Jaccard value of 1 for these rules does not indicate successful variable discrimination. Compression values refer to these stability resamples, separately from the outer-fold evaluation in Table 1.

Joint selection retained between two and seven variables and a mean matrix fraction of 36.25% in these resamples. It chose random-half sampling in five workflows, ordinary geometry in one, robust geometry in three and full-row sampling in one. These compression values belong to the stability experiment and differ from the outer-fold evaluation in Table 1. Resamples overlapped and data and algorithm seeds changed together. The experiment describes combined workflow variation; it neither separates its sources nor supplies an additional prediction estimate because no final classifier was fitted.

## Contamination reduced F1 under the frozen joint decisions

The joint stress test evaluated 35 conditions and 280 arm-condition records: one clean condition and three draws at each of 5% and 10% contamination in each outer fold. Each affected training row received a randomly signed offset of ten clean-training interquartile ranges in one eligible numerical predictor. The 5% events were nested within 10% for a draw, and all arms shared the corrupted matrix. Labels and outer test records stayed unchanged. Corruption preceded utility splitting, so utility-training and utility-validation records could both be affected.

The sampling rule, row fraction, penalty coefficients and feature count were inherited from the clean fold. Selected identities, contributions and redundancy were rebuilt, without rerunning inner budget search. Joint mean F1 fell from 0.8638 to 0.8281 at 5% and 0.8167 at 10% (Table 3). At 10%, full data achieved 0.8228, giving a joint-minus-full difference of -0.0061. The joint clean-to-contaminated loss was 0.0470, with descriptive fold SD of 0.0053; a smaller decrease from a lower clean score does not establish better contaminated performance.

Table 3. Mean F1 and paired losses across five outer folds.

|Selection rule|Clean|5%|10%|Loss 5%|Loss 10%|
|---|---|---|---|---|---|
|Full data|0.8743|0.8341|0.8228|0.0402|0.0515|
|Sample selection only|0.8732|0.8339|0.8216|0.0393|0.0516|
|Feature selection only|0.8663|0.8316|0.8185|0.0347|0.0478|
|Joint selection|0.8638|0.8281|0.8167|0.0356|0.0470|
|Sequential selection|0.8653|0.8315|0.8172|0.0337|0.0480|
|Joint random candidates|0.8667|0.8315|0.8172|0.0352|0.0495|
|Joint ordinary candidates|0.8646|0.8323|0.8172|0.0323|0.0474|
|Joint robust candidates|0.8635|0.8274|0.8171|0.0361|0.0463|

Each contaminated entry averages three draws within each fold, then averages the five folds equally. Loss is clean F1 minus contaminated F1 for the same rule and fold. Negative losses denote improvements. Clean entries use one condition per fold.

The robust, ordinary and random restricted joint rules achieved mean F1 of 0.8171, 0.8172 and 0.8172 at 10%. These inherited decisions did not show a consistent robust-candidate advantage. In the separate sampling-only stress test, the corresponding 10% values were 0.8210 for robust half, 0.8229 for ordinary half and 0.8254 for random half (Table S2). The two stages used distinct selection procedures and seeds; their similar labels must not be treated as identical arms.

ROC-AUC and average precision (AP) supplement the thresholded F1 results (Tables S3 and S4). At 10%, joint ROC-AUC was 0.9708 versus 0.9698 for full data, whereas AP was 0.9033 versus 0.9072. AP was computed descriptively from audited predictions and was not used for selection. No threshold or budget was revised after observing these results.

## Fixed budgets allowed selected identities to change

Joint selection retained 35.00% of cells throughout the stress test, but its mean feature-set Jaccard similarity to the clean selection was 0.7746 at 5% and 0.7524 at 10%. Selected-row similarities were 0.4671 and 0.4666 (Table 4). Thus, unchanged budgets did not imply unchanged selections.

Table 4. Similarity to each fold's clean selection and retained matrix size.

|Selection rule|Feature J 5%|Feature J 10%|Row J 5%|Row J 10%|Cells %|
|---|---|---|---|---|---|
|Full data|1.0000|1.0000|1.0000|1.0000|100.00|
|Sample selection only|1.0000|1.0000|0.6007|0.5993|50.00|
|Feature selection only|0.7524|0.7524|1.0000|1.0000|70.00|
|Joint selection|0.7746|0.7524|0.4671|0.4666|35.00|
|Sequential selection|0.7714|0.7714|0.6007|0.5993|37.50|
|Joint random candidates|0.7841|0.7619|1.0000|1.0000|36.25|
|Joint ordinary candidates|0.7714|0.7714|0.3351|0.3322|37.50|
|Joint robust candidates|0.8190|0.7857|0.3331|0.3355|37.50|

J denotes Jaccard similarity, the intersection divided by the union of the selected sets. Each perturbed selection is compared with the corresponding clean fold. Cells is the retained row fraction multiplied by the retained fraction of the eight original variables; it is unchanged by construction within each frozen fold.

Random sampling can retain identical rows under predictor corruption because row identities and seeds stay fixed. Restricted joint rules also retain a full-data fallback. These design properties can produce high Jaccard values without demonstrating resistance of predictions to contamination. This stress test concerns specified numerical perturbations, not a validated model of astronomical measurement noise.

## Exact diagnostics verified calculations but did not establish path recovery

In 16 standalone utility games, independent exact calculations agreed to a maximum absolute error of 9.29e-14. Increasing the permutation budget from 16 to 1024 reduced mean contribution RMSE from 0.044124 to 0.007057 (Table S1). At the frozen joint budget M=256, mean RMSE was 0.012773 and the top feature matched the exact reference in 9 of 16 games. The diagnostic sweep did not change the evaluation budget.

Across 320 joint-workflow games, M=256 produced mean RMSE of 0.012388 and maximum absolute error of 0.064981. Estimated and exact greedy paths agreed in full order in 105 of 320 cases; prefix sets agreed in 2283 of 2560 comparisons. Exact paths were diagnostic only. Variance correction truncated negative estimates in 18 of 32 standalone feature-score entries at M=256. This truncation is not evidence of feature stability, and the eight-variable games do not establish high-dimensional scalability.

## Fixed budget ablation left selected sets and F1 unchanged

The subsequent factorial ablation held rho=0.5 and k=4 fixed within three sampling strata. CSR denotes variance correction C, contribution-variability penalty S and redundancy penalty R. Enabled penalties were lambda=0.5 and eta=0.1, with B=4 contribution subsamples and M=256 permutations per game. All module variants shared the same selected rows, utility inputs and random streams within a paired comparison. This design evaluated conditional module interventions without repeating inner budget search.

All 135 primary seed-level contrasts, F1(111) minus F1(011), F1(101) or F1(110), were exactly zero. All eight logical configurations selected the same four-feature set within each predictive comparison. The implementation aliases 001=101 and 000=100 were retained as checks, not additional evidence. Table 5 reports absolute performance and the paired random-four control.

Table 5. Fixed budget predictive comparisons across five outer folds

|Sampling|Module F1|Fold SD|Random four F1|Paired difference|Warnings|
|---|---|---|---|---|---|
|Random half|0.8601|0.0102|0.8299|0.0302|11/15|
|Ordinary half|0.8592|0.0101|0.8289|0.0303|9/15|
|Robust half|0.8594|0.0111|0.8296|0.0297|10/15|

Module F1 applies to all eight logical CSR configurations. The difference is 111 minus random_four. Seeds are averaged within folds before equal fold weighting. Warnings count dependent fold-seed units with full-data F1 loss greater than 0.01; the threshold is descriptive. All four-feature arms retain approximately 25.0007% of cells. The full-data reference has mean F1 0.8743.

The all-enabled rule exceeded the random-four control on average in every sampler, but random-half and robust-half sampling each had one negative fold difference. This control comparison does not identify a benefit of any individual module because all scoring variants produced the same selected sets. Half-row all-feature references yielded mean F1 of 0.8707, 0.8716 and 0.8711 for random, ordinary and robust sampling. The four-feature means remained below the full-data mean, and the loss warnings were preserved without budget relaxation.

## Fixed budget set stability did not imply identical paths

Ten stratified 80% resamples of the outer-fold-0 training pool rebuilt selection at the fixed budgets. All eight module variants selected the four local Profile columns in every sampler and resample. Nogueira stability and mean Jaccard were both 1.0000 (Table 6). No final predictive classifier was fitted, and no outer-test predictions were generated. The data and algorithm seeds varied together, so their contributions to variation were not isolated.

Table 6. Fixed budget stability across ten training resamples

|Sampling|Module Nogueira|Module Jaccard|Control Nogueira|Control Jaccard|
|---|---|---|---|---|
|Random half|1.0000|1.0000|0.0667|0.3833|
|Ordinary half|1.0000|1.0000|0.0667|0.3833|
|Robust half|1.0000|1.0000|0.0667|0.3833|

Module columns apply to all eight logical variants; control denotes random_four. All sets contain four of eight variables. Each Jaccard entry averages 45 dependent pairs. Random feature draws are shared across samplers. These resamples rebuild selection at fixed budgets, unlike the complete-workflow resamples in Table 2.

Selected sets were invariant in all primary removal comparisons, yet full eight-feature paths changed in 22 of 135 predictive comparisons and 14 of 90 stability comparisons. Estimated and exact ordered four-feature prefixes agreed in 194 of 360 predictive and 148 of 240 stability logical-label comparisons. These are ordered-prefix diagnostics, not selected-set agreement rates. Across 180 predictive and 120 stability games, mean contribution RMSE was 0.012148 and 0.012635, respectively. Exact references did not replace the frozen selection estimates.

Fixed-budget stability describes a narrower procedure than the adaptive complete workflows in Table 2. It does not show that the contribution-variability penalty improved stability, because disabling it produced the same sets. Nor does it remove the F1 tolerance failures or the selection changes observed under contamination. The full logical factorial table, path breakdown and approximation diagnostics are reported in the accompanying HTRU2 Fixed Budget Ablation Report dated 12 October 2026.

## Verification supported reproduction within the development scope

Independent audits, prespecified replays and unchanged-resume checks passed for the delivered stages. All clean joint-contamination folds exactly reproduced the parent selections, coefficients, probabilities and metrics. The migration delivery recorded 73 passed implementation tests; the subsequent contamination delivery recorded 79. Verification counts for the joint-contamination stage are retained in Table S5. The ablation added a separately accounted ledger of 248,344 successful logistic attempts, including its audits and replays. All 25 main packages and both prescribed replays passed; aggregate checks and cache-only resume preserved 2,049 run artifacts with zero new operations. No experimental fits were performed to integrate these chapters.

The earlier five main stages required 748345 logistic fits, and joint contamination added 428400 contribution fits and 210 final fits. These stage counts include exact diagnostics and exclude separate audits and replays. Baseline median pipeline times were 0.049 s for full data, 0.038 s for random half, 0.318 s for ordinary geometry half and 0.204 s for robust geometry half. Shared caches and overlapping execution prevent separate end-to-end speed claims for the joint and sequential rules.

The outer training folds, stability resamples and contamination draws are dependent. These results describe a previously examined development dataset, with no significance or confirmatory performance claim. The absence of astronomical source identities limits source-level validation, and conflicting descriptor-order documentation limits physical interpretation of selected variables. These boundaries apply to all integrated comparisons.
