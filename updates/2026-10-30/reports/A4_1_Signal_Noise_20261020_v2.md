> Public research copy prepared on 2026-10-30. This is a dated report, not a current stage-status statement. Operational account-use material was omitted where present; scientific results and audit limitations are retained. See the publication manifest for the original source hash and removed source-line ranges.

# A4 Signal and Noise Method Validation

English research progress report | 20 October 2026 | A4.1 version 2

The first A4 package completed the frozen signal_noise_LR layer: eight independently generated datasets and all 48 nested selectors, followed by direct audits, cold replay and an unchanged resume. The S penalty changed mean Nogueira stability by -0.118750 and mean test F1 by -0.002076 across the eight datasets. The weak-signal omission contrast was +0.104167. Nogueira stability was lower in 4 datasets and unchanged in 4; F1 was lower in 4 and unchanged in 4. Neither endpoint improved in any dataset under this comparison. The other three synthetic layers remain pending, so this delivery does not complete A4.

## Frozen evaluation design

Each dataset has 600 training rows, 2,000 test rows and six independent standard-normal predictors. The generating probability is sigmoid(1.5 X1 + 0.5 X2); X1 and X2 are signal and X3 to X6 are noise. Each of six selector resamples contains 480 training rows and four contribution games of 384 rows, with two utility folds per game.

The common learner is L2 logistic regression with C=1, lbfgs, tolerance 0.000001 and a 5,000-iteration limit. Every fitted standardizer uses only its fit's training rows. F1 uses probability threshold 0.5. Final models use the same 480 selector rows, and test data are accessed only after hashed selection commitments.

In a CSR label, C is Monte Carlo variance correction, S is the stability penalty and R is the redundancy penalty. Selection uses B=4 games, M=64 permutations, lambda=0.5 and eta=0.1. The primary contrast 111 minus 101 isolates the stability penalty at selected-set size two. All eight logical CSR variants, sizes one to three, a random path and all six features remain in the archived results. The aliases 001=101 and 000=100 are not independent arms.

## Primary endpoint means

|Endpoint|111 mean|101 mean|Paired mean change|
|---|---|---|---|
|Nogueira|0.800000|0.918750|-0.118750|
|Mean Jaccard|0.822222|0.927778|-0.105556|
|Signal recall|0.895833|0.947917|-0.052083|
|Weak signal omission|0.208333|0.104167|+0.104167|
|Noise count|0.208333|0.104167|+0.104167|
|Test F1|0.725656|0.727732|-0.002076|

All values first aggregate within each dataset, then average over the eight independent datasets. Recall is the fraction of the two signal variables selected. Omission is the fraction of resamples omitting X2. Noise is the mean number of selected noise variables. Positive changes favor 111 for stability, recall and F1; negative changes favor 111 for omission and noise. These are descriptive observations under the frozen synthetic design.

<!--page-->

## All eight stability penalty contrasts

Every value is 111 minus 101 at size two. Seeds identify independent dataset realizations; the six nested resamples within a dataset are not independent observations.

|Seed|Nogueira|Jaccard|Recall|Omission|Noise|F1|
|---|---|---|---|---|---|---|
|2026102401|-0.40000|-0.35556|-0.16667|+0.33333|+0.33333|-0.00540|
|2026102402|+0.00000|+0.00000|+0.00000|+0.00000|+0.00000|+0.00000|
|2026102403|+0.00000|+0.00000|+0.00000|+0.00000|+0.00000|+0.00000|
|2026102404|-0.05000|-0.04444|-0.08333|+0.16667|+0.16667|-0.00285|
|2026102405|-0.25000|-0.22222|-0.08333|+0.16667|+0.16667|-0.00454|
|2026102406|-0.25000|-0.22222|-0.08333|+0.16667|+0.16667|-0.00383|
|2026102407|+0.00000|+0.00000|+0.00000|+0.00000|+0.00000|+0.00000|
|2026102408|+0.00000|+0.00000|+0.00000|+0.00000|+0.00000|+0.00000|

|Endpoint|Mean|Median|Range|Positive / zero / negative|
|---|---|---|---|---|
|Nogueira|-0.11875|-0.02500|-0.40000 to +0.00000|0 / 4 / 4|
|Mean Jaccard|-0.10556|-0.02222|-0.35556 to +0.00000|0 / 4 / 4|
|Signal recall|-0.05208|-0.04167|-0.16667 to +0.00000|0 / 4 / 4|
|Weak signal omission|+0.10417|+0.08333|+0.00000 to +0.33333|4 / 4 / 0|
|Noise count|+0.10417|+0.08333|+0.00000 to +0.33333|4 / 4 / 0|
|Test F1|-0.00208|-0.00142|-0.00540 to +0.00000|0 / 4 / 4|

Nogueira stability is computed from selection indicators across the six resamples. Mean Jaccard averages their 15 pairwise set similarities. Those 15 pairs share resamples and cannot increase the independent sample size. The contrast signs are computed from unrounded values; displayed zeros and rounding are not used to classify direction.

Stability is interpreted jointly with recovery and prediction. A stable selection that repeatedly omits X2 or includes noise is not a demonstrated benefit. The fixed random path can have conditional stability equal to one without recovering signal, because its algorithmic randomness is held constant across the resamples. No significance test, confidence interval or selection of favorable seeds is introduced.

The prespecified secondary sizes and CSR arms remain available in the complete dataset summaries. Their retention does not make this primary table an optimization step or authorize choosing a better-looking setting after observing the results.

<!--page-->

## Correction diagnostics for the finite B target

The correction analysis targets the sample variance of exact Shapley contributions across the same four realized games. It does not estimate an independently established population variance. Exact contributions are computed only after the selector is committed and are never used to select features.

For each nested selector, 20 diagnostic streams independently generate permutation estimates within each of the four games. M=16 uses the prefix of its corresponding M=64 stream. These streams are separate from the main selection stream. The uncorrected estimator is the sample variance of estimated contributions. Raw correction subtracts the mean estimated Monte Carlo variance; clipped correction replaces negative raw values by zero.

|M|Estimator|Mean bias|Mean MSE|Clipping frequency|
|---|---|---|---|---|
|16|Uncorrected|+1.040e-03|3.842e-06|Not applied|
|16|Raw corrected|+2.078e-05|1.880e-06|Not applied|
|16|Clipped corrected|+2.679e-04|1.416e-06|0.39462|
|64|Uncorrected|+2.529e-04|3.212e-07|Not applied|
|64|Raw corrected|-1.773e-06|2.021e-07|Not applied|
|64|Clipped corrected|+4.059e-05|1.740e-07|0.23038|

Bias is the estimator minus the finite-B exact target, averaged within a dataset over resamples, streams and features. MSE averages the squared errors over those same nested records. The table then averages the eight dataset summaries equally. Clipping frequency is the fraction of negative raw corrected values, before replacement by zero.

At M=16, raw correction reduced MSE in 8 of eight datasets and clipping reduced MSE in 8. At M=64, raw correction reduced MSE in 8 of eight datasets and clipping reduced MSE in 8. The complete dataset-level MSE contrasts on the next page preserve the direction and magnitude of every realization.

Lower Monte Carlo estimation error under this finite-B comparison does not establish improved feature selection or prediction. Clipping changes the estimator and may trade reduced squared error against a changed bias. The raw and clipped versions are therefore both reported. The stream, feature and resample dimensions remain nested diagnostic records, and no additional model fits are required for these calculations.

<!--page-->

## All eight correction MSE contrasts

Each value is corrected MSE minus uncorrected MSE. Negative values favor correction for this error measure. Raw and clipped correction are shown at both prespecified permutation counts.

|Seed|M16 raw|M16 clipped|M64 raw|M64 clipped|
|---|---|---|---|---|
|2026102401|-2.931e-06|-3.720e-06|-2.214e-07|-2.659e-07|
|2026102402|-3.070e-06|-3.805e-06|-1.722e-07|-2.124e-07|
|2026102403|-5.517e-09|-6.733e-09|-2.092e-10|-2.434e-10|
|2026102404|-2.563e-08|-3.381e-08|-9.832e-10|-1.345e-09|
|2026102405|-7.062e-09|-8.979e-09|-4.258e-10|-4.441e-10|
|2026102406|-3.623e-06|-4.418e-06|-2.015e-07|-2.490e-07|
|2026102407|-4.602e-06|-5.574e-06|-2.676e-07|-3.394e-07|
|2026102408|-1.428e-06|-1.845e-06|-8.842e-08|-1.089e-07|

|Estimator|Mean|Median|Range|Positive / zero / negative|
|---|---|---|---|---|
|M16 Raw|-1.962e-06|-2.180e-06|-4.602e-06 to -5.517e-09|0 / 0 / 8|
|M16 Clipped|-2.426e-06|-2.783e-06|-5.574e-06 to -6.733e-09|0 / 0 / 8|
|M64 Raw|-1.191e-07|-1.303e-07|-2.676e-07 to -2.092e-10|0 / 0 / 8|
|M64 Clipped|-1.472e-07|-1.607e-07|-3.394e-07 to -2.434e-10|0 / 0 / 8|

The ranges and sign counts summarize the eight independent datasets separately for each estimator and permutation count. They are descriptive and do not convert 20 streams, six features or six resamples into independent replication. Arithmetic and sign checks use full precision.

These results address a known-truth signal and noise generator with the frozen LR learner. They do not establish contamination robustness, general method superiority, population variance calibration, performance on correlated groups or performance on XOR. The current layer also does not isolate a benefit of the redundancy penalty. The correlated-group comparison and learner-specific XOR layers remain necessary parts of the frozen A4 design.

<!--page-->

## Accounting and verification

|Recorded operation|Attempts or events|Scope|
|---|---|---|
|Classifier fits|25,411|First package cap 26,436|
|Encoder fits|25,411|First package cap 26,436|
|KMeans fits|0|Cap zero|
|Failed or unresolved attempts|0|No failed started or interrupted rows|
|Direct audit classifier refits|336|48 selectors times 7|
|RF forest fits and internal trees|16 / 1,024|Training only probes|
|Fitted cache hits|2,050|No new fit|

|Verification|Status|Evidence|
|---|---|---|
|Timing probes|Pass|32 training only LR and RF fits; frozen timing gates passed|
|Direct refits|Pass|336 fixed target probabilities labels F1 and preprocessing matched|
|Cold replay|Pass|510 fresh prediction and preprocessing pairs matched|
|Unchanged resume|Pass|127717 files unchanged; zero new operations|
|Independent audit|Pass|48 selectors and eight dataset summaries independently reconstructed|
|Historical evidence|Pass|Historical evidence prior deliveries and original STOP hashes preserved|

The attempt ledger includes the 32 training-only LR and RF timing probes, main selectors, 336 fixed direct audits and the prescribed first-seed q0 cold replay. Encoder and classifier operations are counted separately. RF forests and their internal trees are disclosed as different units; neither is treated as equivalent work to an LR fit.

Recorded phase elapsed times were 4.1 s for probes, 1466.8 s for all main selectors, 25.3 s for direct audits and 31.8 s for cold replay. Their sum was 1528.0 s. This total excludes document preparation, independent read-only verification and archival. It is not a general scalability result.

<!--page-->

## Timing revision and continuation scope

The version 2 probe measured 1.273291 seconds of library initialization separately from counted fit pairs. The model-operation projection was 0.537015 hours. Adding initialization for the frozen 41 full-stage invocations gave 0.551516 hours, below the 12-hour warning. The RF component was 0.497685 model hours plus initialization for 11 invocations, giving 0.501575 hours below its six-hour warning. The RF initialization allowance is part of the total allowance, not added to it again.

These projections use the maximum of all 16 counted probe durations per learner and the complete frozen learner inventories. They are feasibility estimates, not observed completion of later layers or steady-state throughput measurements. Probe, main and replay intervals include initialization; the direct-audit interval starts after initialization. These are recorded phase intervals, not end-to-end wall time. Scientific caps remain 112,000 classifier and 112,000 encoder attempts for cumulative A4 accounting, with a 26,436 bound for each in this version 2 package.

The first run's failed runtime gate remains preserved. It completed 32 classifier and 32 encoder probes and started no main selector. Those attempts remain in the historical accounting and are excluded from the version 2 counts above. The combined counts across both versions are 25,443 classifier and 25,443 encoder attempts.

The user separately authorized a prospective timing correction, a new execution freeze and completion of all eight datasets while preserving the global reserve. Version 2 loads the required libraries before probe timing. It performs no uncounted warmup fits and drops no probe observation. The earlier failed projection is not overwritten or reclassified as passed. Version 2 feasibility and results belong to this separately frozen execution.

The correlated_groups_LR, xor_RF and xor_LR layers remain pending. A3 folds 1 to 4 remain pending; WDBC A2 folds 3 and 4 remain deferred; Adult remains blocked by its runtime gate; native PowerSHAP remains separately gated. The current milestone does not complete the thesis. Code and compact reports remain on C; new run artifacts and delivery archives use F. Final checkpoint and restoration evidence are recorded separately after report review.
