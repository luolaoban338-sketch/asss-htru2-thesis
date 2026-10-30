> Public research copy prepared on 2026-10-30. This is a dated report, not a current stage-status statement. Operational account-use material was omitted where present; scientific results and audit limitations are retained. See the publication manifest for the original source hash and removed source-line ranges.

# A4 Correlated Groups Method Validation

English research progress report | 25 October 2026 | A4 2a

The bounded correlated groups LR experiment completed eight independent generated datasets and 48 nested selectors. For the prespecified comparison 111 minus 110 at size two, mean group coverage changed by +0.145833, same group excess by -0.291667 and noise count by +0.000000. Mean test F1 changed by +0.019557. The result is described by the complete paired structural and prediction differences below; individual endpoints are not treated as independent confirmations. The two XOR layers remain pending.

## Frozen design and primary comparison

Each dataset contains 600 training rows, 2000 independent test rows and six predictors. The signal groups are {X1, X2} and {X3, X4}, each with population correlation 0.9; X5 and X6 are noise. With Z1=(X1+X2)/sqrt(3.8) and Z2=(X3+X4)/sqrt(3.8), the generating probability is sigmoid[1.5(Z1+Z2)/sqrt(2)]. All four grouped variables enter the true model. Selecting two representatives does not recover all four signal variables.

Each dataset has six nested selector resamples of 480 training rows, with four contribution samples of 384 rows and two utility folds. The fixed learner is L2 logistic regression with C=1, lbfgs, tolerance 0.000001 and at most 5000 iterations. Standardization uses each fit's training rows only. Final models use the same 480 selector rows and threshold 0.5. Test data and exact diagnostics are accessed after hashed selection commitments.

In a CSR label, C denotes Monte Carlo variance correction, S the stability penalty and R the redundancy penalty. The primary comparison is 111 minus 110 at k=2, holding C and S fixed. Selection uses B=4, M=64, lambda=0.5 and eta=0.1. All eight logical variants, k=1/2/3, random control and all features remain recorded; 001=101 and 000=100 are structural aliases.

## Primary endpoint means

|Endpoint|111 mean|110 mean|Paired mean change|
|---|---|---|---|
|Group coverage|0.864583|0.718750|+0.145833|
|Same group excess|0.270833|0.562500|-0.291667|
|Noise count|0.000000|0.000000|+0.000000|
|Test F1|0.709639|0.690081|+0.019557|
|Nogueira|0.643750|0.637500|+0.006250|
|Mean Jaccard|0.697222|0.683333|+0.013889|

Means first aggregate the six nested resamples within each dataset, then weight the eight datasets equally. The independent sample size is eight. The six resamples, 20 diagnostic streams and 15 Jaccard pairs do not enlarge it.

<!--page-->

## All eight primary structural contrasts

Every value is 111 minus 110 at k=2. H counts covered signal groups, E counts within-group selections beyond one representative and N counts selected noise variables. Group coverage G=H/2 is 0, 0.5 or 1, rather than a binary indicator that both groups were recovered. Each selected set satisfies H+E+N=2.

|Seed|Group coverage|Same group excess|Noise count|
|---|---|---|---|
|2026102401|+0.166667|-0.333333|+0.000000|
|2026102402|+0.250000|-0.500000|+0.000000|
|2026102403|+0.000000|+0.000000|+0.000000|
|2026102404|+0.166667|-0.333333|+0.000000|
|2026102405|+0.166667|-0.333333|+0.000000|
|2026102406|+0.250000|-0.500000|+0.000000|
|2026102407|+0.000000|+0.000000|+0.000000|
|2026102408|+0.166667|-0.333333|+0.000000|

|Endpoint|Mean|Median|Range|Positive zero negative|
|---|---|---|---|---|
|Group coverage|+0.145833|+0.166667|+0.000000 to +0.250000|6 / 2 / 0|
|Same group excess|-0.291667|-0.333333|-0.500000 to +0.000000|0 / 2 / 6|
|Noise count|+0.000000|+0.000000|+0.000000 to +0.000000|0 / 8 / 0|

Positive coverage differences, negative excess differences and negative noise differences are favorable directions. The three endpoints are constrained by the fixed selected-set size. For each dataset, integer sums over the six pairs satisfy D_H+D_E+D_N=0, with coverage change D_H/12 and excess and noise changes D_E/6 and D_N/6. They are not three independent pieces of evidence.

Across 48 paired selectors, 15 changed the selected set, 19 changed the complete feature path and 14 changed at least one structural count. Paired-selector counts were 1 for within-group substitutions and 1 for nonidentical selected sets with unchanged structural counts. Noise entered in 0 pairs and left in 0. These nested event counts explain the response and are not additional independent replicates.

The result is described by the complete paired structural and prediction differences below; individual endpoints are not treated as independent confirmations.

The corrected scores and first selected feature match between 111 and 110 in every pair, because the redundancy penalty is zero at an empty prefix. The independent audit also reconstructs the second-step score, eta/6 times the absolute Spearman association and the selected feature. These records explain path changes without introducing a new endpoint or retuning the penalty.

<!--page-->

## Prediction and conditional stability

F1 columns show absolute dataset means and paired changes. Nogueira and Jaccard columns show 111 minus 110 at k=2, computed from the six complete selected sets within each dataset. Feature identity changes within a correlated signal group can alter feature-level stability while preserving group coverage.

|Seed|111 F1|110 F1|F1 change|Nogueira change|Jaccard change|
|---|---|---|---|---|---|
|2026102401|0.724889|0.704237|+0.020652|+0.100000|+0.088889|
|2026102402|0.716226|0.684921|+0.031305|-0.200000|-0.155556|
|2026102403|0.708025|0.708025|+0.000000|+0.000000|+0.000000|
|2026102404|0.706842|0.684320|+0.022523|+0.200000|+0.177778|
|2026102405|0.720054|0.701024|+0.019030|+0.200000|+0.177778|
|2026102406|0.747530|0.706283|+0.041247|-0.050000|+0.000000|
|2026102407|0.637057|0.637057|+0.000000|+0.000000|+0.000000|
|2026102408|0.716485|0.694784|+0.021701|-0.200000|-0.177778|

|Endpoint|Mean|Median|Range|Positive zero negative|
|---|---|---|---|---|
|Test F1|+0.019557|+0.021177|+0.000000 to +0.041247|6 / 2 / 0|
|Nogueira|+0.006250|+0.000000|-0.200000 to +0.200000|3 / 2 / 3|
|Mean Jaccard|+0.013889|+0.000000|-0.177778 to +0.177778|3 / 3 / 2|

Integer TP, FP and FN counts gave 6 positive, 2 exactly zero and 0 negative dataset-level F1 differences. These classifications use rational fractions from the prediction labels. Display rounding and the audit tolerance are not equivalence or noninferiority criteria.

Stability directions use exact rational calculations from the saved selected sets. The frozen summary verification failed on a floating-point direction count. Independent reconciliation proved two exact zeros: the R Jaccard difference for seed 2026102406 and the S Nogueira difference for seed 2026102408. Their floating residues were about 5.55e-17 and 1.11e-16. This report uses the reconciled values; the original summaries, failure and frozen source remain unchanged. Numerical tolerance was not used to erase a genuine nonzero difference.

Nogueira stability is based on feature-selection indicators; mean Jaccard averages 15 overlapping pairs. The sampling and model coordinates follow the frozen namespace, and random-path coordinates are fixed across resamples. Stability is conditional on that design. A fixed random subset can be perfectly stable without recovering signal; all-feature Nogueira stability is undefined and is retained as such.

Structural improvement and prediction improvement are separate outcomes. Any coverage gain must be read together with excess, noise and F1. Reducing duplicate representatives by selecting noise is not evidence of useful group recovery. No significance test, confidence interval, noninferiority threshold or post hoc favorable-seed selection is added.

<!--page-->

## Secondary stability penalty contrast

The prespecified secondary comparison 111 minus 101 at k=2 isolates S. It does not replace the primary R comparison. The columns below retain all eight datasets, including exact zeros and adverse directions. Coverage, excess and noise retain their correlated-group definitions; weak-signal omission from the signal-noise layer is not applicable here.

|Seed|Coverage|Excess|Noise|F1|Nogueira|Jaccard|
|---|---|---|---|---|---|---|
|2026102401|+0.00000|+0.00000|+0.00000|+0.00000|+0.00000|+0.00000|
|2026102402|+0.00000|+0.00000|+0.00000|-0.00191|+0.20000|+0.13333|
|2026102403|+0.00000|+0.00000|+0.00000|+0.00489|+0.40000|+0.35556|
|2026102404|+0.41667|-0.83333|+0.00000|+0.05671|-0.25000|-0.22222|
|2026102405|+0.00000|+0.00000|+0.00000|+0.00000|+0.00000|+0.00000|
|2026102406|+0.00000|+0.00000|+0.00000|+0.00016|-0.30000|-0.20000|
|2026102407|+0.00000|+0.00000|+0.00000|+0.00000|+0.00000|+0.00000|
|2026102408|-0.16667|+0.33333|+0.00000|-0.02191|+0.00000|+0.00000|

|Endpoint|Mean|Median|Range|Positive zero negative|
|---|---|---|---|---|
|Group coverage|+0.031250|+0.000000|-0.166667 to +0.416667|1 / 6 / 1|
|Same group excess|-0.062500|+0.000000|-0.833333 to +0.333333|1 / 6 / 1|
|Noise count|+0.000000|+0.000000|+0.000000 to +0.000000|0 / 8 / 0|
|Test F1|+0.004743|+0.000000|-0.021907 to +0.056715|3 / 3 / 2|
|Nogueira|+0.006250|+0.000000|-0.300000 to +0.400000|2 / 4 / 2|
|Mean Jaccard|+0.008333|+0.000000|-0.222222 to +0.355556|2 / 4 / 2|

The same paired training rows, utility tables, final training sample and heldout data are used within each comparison. Means, medians, ranges and sign counts refer to eight dataset-level values. Positive stability or F1 changes alone cannot establish improved group recovery. The full logical-arm and selected-size outputs are preserved in the source summaries for traceability; they were not used to choose a new primary setting.

<!--page-->

## Secondary correction diagnostics

C diagnostics target the sample variance of exact Shapley contributions across the same four realized games, with denominator B minus one. They do not estimate an independently known population variance. Twenty streams per selector are separate from the main selection stream; M16 uses the prefix of its corresponding M64 stream. Exact values are computed after selection and never enter it.

|M|Estimator|Mean bias|Mean MSE|Clipping frequency|
|---|---|---|---|---|
|16|Uncorrected|+1.996e-03|1.032e-05|Not applied|
|16|Raw corrected|-1.056e-04|4.512e-06|Not applied|
|16|Clipped corrected|+5.550e-04|2.827e-06|0.53038|
|64|Uncorrected|+5.238e-04|7.615e-07|Not applied|
|64|Raw corrected|-1.288e-06|3.439e-07|Not applied|
|64|Clipped corrected|+1.400e-04|2.451e-07|0.39670|

Bias is estimator minus finite-B exact target; MSE is mean squared error. Each is averaged over nested resamples, streams and features before the eight dataset summaries are weighted equally. Clipping frequency counts negative raw corrections replaced by zero. The following values are corrected MSE minus uncorrected MSE, so negative values favor correction for this diagnostic error.

|Seed|M16 raw|M16 clipped|M64 raw|M64 clipped|
|---|---|---|---|---|
|2026102401|-8.798e-06|-1.073e-05|-5.967e-07|-7.196e-07|
|2026102402|-2.186e-08|-2.547e-08|-1.344e-09|-1.520e-09|
|2026102403|-8.305e-06|-1.083e-05|-6.342e-07|-7.712e-07|
|2026102404|-7.099e-06|-8.849e-06|-4.442e-07|-5.634e-07|
|2026102405|-8.137e-06|-1.054e-05|-6.117e-07|-7.422e-07|
|2026102406|-2.163e-08|-2.632e-08|-1.310e-09|-1.472e-09|
|2026102407|-6.459e-06|-9.280e-06|-5.546e-07|-7.184e-07|
|2026102408|-7.623e-06|-9.661e-06|-4.963e-07|-6.135e-07|

At M=16, raw correction reduced MSE in 8 datasets and increased it in 0; clipped correction reduced MSE in 8 and increased it in 0. At M=64, raw correction reduced MSE in 8 datasets and increased it in 0; clipped correction reduced MSE in 8 and increased it in 0.

Raw and clipped corrections are both retained because clipping changes bias as well as squared error. Lower diagnostic MSE does not itself demonstrate improved feature selection, prediction, population variance calibration or contamination robustness. These diagnostics require no additional model fits.

<!--page-->

## Operation accounting and verification

|Recorded operation|Attempts or events|Scope|
|---|---|---|
|Classifier fits|25,421|Current package cap 26404|
|Encoder fits|25,421|Current package cap 26404|
|KMeans and new probes|0 / 0|Neither belongs to this layer|
|Direct classifier refits|336|48 selectors with 7 fixed targets|
|Failed or unresolved attempts|0|All operation rows succeeded|
|Cumulative A4 fits per kind|50,864|Includes 25443 prior attempts per kind; cap 112000|

|Verification|Status|Evidence|
|---|---|---|
|Direct refits|Pass|All 336 fixed probabilities labels preprocessing and F1 checks|
|Cold replay|Pass|514 independent prediction and preprocessing pairs|
|Unchanged resume|Pass|127765 files; zero new operations or predictions|
|Independent audit|Pass|48 selections and all 8 dataset summaries reconstructed|
|Historical evidence|Pass|Original sources prior deliveries and earlier STOPs preserved|
|Worker closure|Pass|Finite fitting workers exited and SQLite sidecars absent|

Recorded fitting invocation intervals total 13713.2 seconds for the eight main datasets, 349.0 seconds for direct audits and 549.1 seconds for cold replay. Main and replay intervals include class initialization; the direct-audit interval starts after initialization. These are measured invocation intervals, excluding independent read-only verification, reporting and archive work, not end-to-end runtime or a scalability result.

The timing gate inherits the verified v2 maximum-of-16 measurements per learner and its initialization inventory. No new probe or warm-up fit was introduced. The inherited LR timing came from the signal-noise training data and is not a correlated-group-specific benchmark. The original failed timing gate and its 32 successful fit pairs remain preserved and counted; the later v2 run contributes a further 25411 pairs.

## Interpretation and remaining scope

This experiment addresses the redundancy penalty under one frozen correlated-group generator, one learner and one parameter setting. It contains no L1 comparison and no sample-compression factor. The earlier five-fold HTRU2 comparison, in which ASSS matched Mean Shapley and had lower mean F1 than L1, remains part of the thesis evidence. Synthetic mechanism results do not reverse that comparison or establish overall superiority.

Only the correlated_groups_LR layer is completed here. XOR RF and LR remain pending; A3 wider work and native PowerSHAP remain separately gated, and A2 WDBC folds 3 and 4 remain deferred. Scientific inputs and earlier STOP markers are preserved. The final delivery manifest and archive receipt separately record ZIP integrity, restoration checks and the closed checkpoint; this report does not certify those later archive operations.
