> Dated public research copy, prepared 11 November 2026. The source date and scientific findings are retained. Current progress is stated in the update README: Adult confirmation timing passed on 10 November; its main experiment has not run. Local source links are shown as source references because this is a curated inspection snapshot. One account-usage paragraph is omitted; all scientific results and failed/interrupted-operation caveats remain.

# WDBC five fold development comparison

5 November 2026 | Logistic regression | Outer development folds 0 to 4

All five planned WDBC outer folds have completed the frozen comparison and independent audit. Across folds, the joint procedure retains 14.69% of training cells and gives mean F1 0.9582, compared with 0.9649 for the full-data arm (paired mean difference -0.0067). The joint procedure exceeds the 0.01 outer F1 loss tolerance in 2 of five folds. This closes the WDBC five-fold main comparison; Adult, full R09 and the thesis remain open.

Table 1 Equal weight means across five development folds

| Arm | Mean F1 | Mean AUC | Rows % | Mean k | Cells % | Breaches |
| --- | --- | --- | --- | --- | --- | --- |
| Full | 0.9649 | 0.9956 | 100.00 | 30.00 | 100.00 | 0/5 |
| Sample only | 0.9685 | 0.9946 | 50.09 | 30.00 | 50.09 | 2/5 |
| Feature only | 0.9596 | 0.9929 | 100.00 | 9.80 | 32.67 | 3/5 |
| Joint | 0.9582 | 0.9916 | 50.09 | 8.80 | 14.69 | 2/5 |
| Sequential | 0.9583 | 0.9923 | 50.09 | 13.00 | 21.71 | 2/5 |
| Joint random | 0.9585 | 0.9911 | 60.09 | 12.00 | 26.03 | 3/5 |
| Joint ordinary | 0.9497 | 0.9914 | 50.09 | 9.40 | 15.69 | 2/5 |
| Joint robust | 0.9567 | 0.9934 | 70.04 | 11.80 | 24.02 | 3/5 |

Each fold has equal weight. Rows and cells are percentages of its complete outer training matrix; k counts original variables out of 30. Cell retention is calculated within each fold as its actual row fraction multiplied by k/30, then averaged. The product of the two separate means is not substituted.

Table 2 Newly completed fold 4 performance and retained matrix

| Arm | F1 | AUC | Rows % | k | Cells % | Breach |
| --- | --- | --- | --- | --- | --- | --- |
| Full | 0.9351 | 0.9825 | 100.00 | 30 | 100.00 | No |
| Sample only | 0.9474 | 0.9808 | 50.00 | 30 | 50.00 | No |
| Feature only | 0.9474 | 0.9760 | 100.00 | 18 | 60.00 | No |
| Joint | 0.9351 | 0.9733 | 50.00 | 11 | 18.33 | No |
| Sequential | 0.9351 | 0.9733 | 50.00 | 11 | 18.33 | No |
| Joint random | 0.9474 | 0.9760 | 100.00 | 18 | 60.00 | No |
| Joint ordinary | 0.9351 | 0.9733 | 50.00 | 11 | 18.33 | No |
| Joint robust | 0.9231 | 0.9771 | 50.00 | 25 | 41.67 | Yes |

Fold 4 contains 456 training rows and 113 held-out rows. The outer loss tolerance is exceeded by 1 of its seven compressed arms. Folds 0 to 3 retain their previously accepted results; only fold 4 adds new model fits.

<!--page-->

## Paired differences and tolerance violations

A positive difference below favors the named compressed arm over the full-data arm in the same fold. An asterisk marks full F1 minus arm F1 greater than 0.01 plus the frozen numerical tolerance of 1e-12. Flags use unrounded values, so a displayed difference near -0.0100 can require the source precision.

Table 3 F1 differences from full data within each fold

| Arm | Fold 0 | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Mean |
| --- | --- | --- | --- | --- | --- | --- |
| Sample only | -0.0101* | +0.0277 | +0.0000 | -0.0123* | +0.0123 | +0.0035 |
| Feature only | -0.0196* | +0.0277 | -0.0356* | -0.0118* | +0.0123 | -0.0054 |
| Joint | -0.0212* | +0.0000 | -0.0123* | +0.0000 | +0.0000 | -0.0067 |
| Sequential | -0.0212* | +0.0000 | -0.0118* | +0.0000 | +0.0000 | -0.0066 |
| Joint random | -0.0204* | +0.0000 | -0.0123* | -0.0118* | +0.0123 | -0.0064 |
| Joint ordinary | -0.0309* | +0.0000 | -0.0454* | +0.0000 | +0.0000 | -0.0153 |
| Joint robust | -0.0212* | +0.0277 | -0.0356* | +0.0000 | -0.0120* | -0.0082 |

The inner validation rule selects a small admissible matrix using its prespecified loss tolerance. The outer observations evaluate that selection procedure on complete held-out rows. Inner admissibility therefore does not establish an outer noninferiority guarantee, and the mean does not erase violating folds.

Table 4 Joint minus sequential within each fold

| Fold | F1 difference | AUC difference | Rows difference pp | Cells difference pp |
| --- | --- | --- | --- | --- |
| 0 | +0.0000 | +0.0000 | +0.00 | +0.00 |
| 1 | +0.0000 | +0.0000 | +0.00 | +0.00 |
| 2 | -0.0006 | -0.0023 | +0.00 | -25.05 |
| 3 | +0.0000 | -0.0013 | +0.00 | -10.02 |
| 4 | +0.0000 | +0.0000 | +0.00 | +0.00 |

The mean joint-minus-sequential F1 difference is -0.0001; the mean cell-retention difference is -7.02 percentage points. Negative cell differences mean that the joint procedure retains fewer training cells. The fold-level contrasts show whether this compression accompanies a performance loss.

The five outer training sets overlap. These are descriptive development comparisons rather than five independent replications or a confirmatory test. No statistical superiority, general robustness or Monte Carlo accuracy claim follows from these tables.

<!--page-->

## Complete records for folds 0 to 2

The two adjacent appendix tables contain all 40 arm-by-fold records, grouped by outer fold. No arm is omitted because it duplicates another arm's final model or violates the loss tolerance.

Table 5 All arm records for the first three folds

| Fold | Arm | F1 | AUC | Rows % | k | Cells % | Breach |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | Full | 0.9804 | 0.9994 | 100.00 | 30 | 100.00 | No |
| 0 | Sample only | 0.9703 | 0.9959 | 50.11 | 30 | 50.11 | Yes |
| 0 | Feature only | 0.9608 | 0.9969 | 100.00 | 8 | 26.67 | Yes |
| 0 | Joint | 0.9592 | 0.9966 | 50.11 | 8 | 13.36 | Yes |
| 0 | Sequential | 0.9592 | 0.9966 | 50.11 | 8 | 13.36 | Yes |
| 0 | Joint random | 0.9600 | 0.9959 | 50.11 | 12 | 20.04 | Yes |
| 0 | Joint ordinary | 0.9495 | 0.9956 | 50.11 | 11 | 18.37 | Yes |
| 0 | Joint robust | 0.9592 | 0.9966 | 50.11 | 8 | 13.36 | Yes |
| 1 | Full | 0.9333 | 0.9959 | 100.00 | 30 | 100.00 | No |
| 1 | Sample only | 0.9610 | 0.9962 | 50.11 | 30 | 50.11 | No |
| 1 | Feature only | 0.9610 | 0.9966 | 100.00 | 6 | 20.00 | No |
| 1 | Joint | 0.9333 | 0.9918 | 50.11 | 8 | 13.36 | No |
| 1 | Sequential | 0.9333 | 0.9918 | 50.11 | 8 | 13.36 | No |
| 1 | Joint random | 0.9333 | 0.9891 | 50.11 | 9 | 15.03 | No |
| 1 | Joint ordinary | 0.9333 | 0.9918 | 50.11 | 8 | 13.36 | No |
| 1 | Joint robust | 0.9610 | 0.9966 | 100.00 | 6 | 20.00 | No |
| 2 | Full | 0.9880 | 1.0000 | 100.00 | 30 | 100.00 | No |
| 2 | Sample only | 0.9880 | 1.0000 | 50.11 | 30 | 50.11 | No |
| 2 | Feature only | 0.9524 | 0.9967 | 100.00 | 7 | 23.33 | Yes |
| 2 | Joint | 0.9756 | 0.9977 | 50.11 | 10 | 16.70 | Yes |
| 2 | Sequential | 0.9762 | 1.0000 | 50.11 | 25 | 41.76 | Yes |
| 2 | Joint random | 0.9756 | 0.9977 | 50.11 | 10 | 16.70 | Yes |
| 2 | Joint ordinary | 0.9425 | 0.9977 | 50.11 | 10 | 16.70 | Yes |
| 2 | Joint robust | 0.9524 | 0.9967 | 100.00 | 7 | 23.33 | Yes |

Breach denotes the outer F1 loss relative to the same fold's full-data arm. Actual row fractions may differ slightly from a nominal half because integer row counts and class-stratified sampling are retained.

<!--page-->

## Complete records for folds 3 and 4

The two adjacent appendix tables contain all 40 arm-by-fold records, grouped by outer fold. No arm is omitted because it duplicates another arm's final model or violates the loss tolerance.

Table 6 All arm records for the final two folds

| Fold | Arm | F1 | AUC | Rows % | k | Cells % | Breach |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 3 | Full | 0.9880 | 1.0000 | 100.00 | 30 | 100.00 | No |
| 3 | Sample only | 0.9756 | 1.0000 | 50.11 | 30 | 50.11 | Yes |
| 3 | Feature only | 0.9762 | 0.9983 | 100.00 | 10 | 33.33 | Yes |
| 3 | Joint | 0.9880 | 0.9987 | 50.11 | 7 | 11.69 | No |
| 3 | Sequential | 0.9880 | 1.0000 | 50.11 | 13 | 21.71 | No |
| 3 | Joint random | 0.9762 | 0.9967 | 50.11 | 11 | 18.37 | Yes |
| 3 | Joint ordinary | 0.9880 | 0.9987 | 50.11 | 7 | 11.69 | No |
| 3 | Joint robust | 0.9880 | 1.0000 | 50.11 | 13 | 21.71 | No |
| 4 | Full | 0.9351 | 0.9825 | 100.00 | 30 | 100.00 | No |
| 4 | Sample only | 0.9474 | 0.9808 | 50.00 | 30 | 50.00 | No |
| 4 | Feature only | 0.9474 | 0.9760 | 100.00 | 18 | 60.00 | No |
| 4 | Joint | 0.9351 | 0.9733 | 50.00 | 11 | 18.33 | No |
| 4 | Sequential | 0.9351 | 0.9733 | 50.00 | 11 | 18.33 | No |
| 4 | Joint random | 0.9474 | 0.9760 | 100.00 | 18 | 60.00 | No |
| 4 | Joint ordinary | 0.9351 | 0.9733 | 50.00 | 11 | 18.33 | No |
| 4 | Joint robust | 0.9231 | 0.9771 | 50.00 | 25 | 41.67 | Yes |

Breach denotes the outer F1 loss relative to the same fold's full-data arm. Actual row fractions may differ slightly from a nominal half because integer row counts and class-stratified sampling are retained.

The restricted joint arms use full rows in 3 of their 15 arm-fold records. These fallbacks are allowed by the frozen rule. Consequently, arm names alone do not establish equal sampling budgets; R08 requires matched realized row fractions, feature modules and paired records, or a separately frozen comparison.

Candidate evaluations are shared across several decision rules. Their stored operation totals describe this execution, not separate end-to-end timings for each method. The WDBC results extend the observed row-column compression tradeoff to a second real dataset while preserving these limits.

<!--page-->

## Audit resources and remaining evidence

Table 7 Recorded classifier and preprocessing attempts

| Kind | Before fold 4 | Added | Cumulative | Fold 4 cap |
| --- | --- | --- | --- | --- |
| Classifier | 1,714,186 | 343,364 | 2,057,550 | 358,192 |
| Encoder | 1,714,792 | 343,485 | 2,058,277 | 358,312 |
| KMeans | 1,040 | 208 | 1,248 | 208 |

The new fold contains 64 contribution games, 16 selectors and eight outer decision arms. Its independent audit adds 200 classifier and 200 encoder refits: three utility refits per game and one final refit per arm. Across the five main folds, 1,000 classifier audit refits are verified. The accepted fold 0 cold replay and its 200 audit refits are reverified and reused, giving 1,200 main-plus-replay audit refits; no new cold replay is claimed.

Attempt counts include failures and any successful work not committed before an interruption. The historical classifier attempt 1726048 remains failed after the October 16 interruption, with actual completion and duration unknown. No additional failed or unfinished attempts are accepted. The independent checks preserve the original ledger prefix, earlier fold evidence and frozen scientific inputs.

The first invocation was deliberately stopped during random-half selection after the preceding full-sample selector had committed, with 38 of 64 games and 9 of 16 selectors committed. Its original worker receipt retains SystemExit(75), and ledger session 16 retains status failed with 2,610.152 seconds. This is an execution-control outcome, distinct from a failed model fit. The stopped snapshot had charged 204,540 classifier, 204,602 encoder and 104 KMeans attempts and retained only the historical failed fit 1726048. One manually reviewed continuation then completed the same fold under the original account and model-attempt ceilings. The original receipt and counter baseline remain unchanged. Successful but uncommitted work and any work repeated on continuation remain included in the full attempt totals above.


Cache recovery has passed with fitting and prediction disabled and zero new ledger operations. The frozen run and ledger remain on C. New archive and restoration files use the configured F store. Document numeric and individual-page review, followed by all-member archive CRC, size, SHA-256 and restoration checks, complete the delivery evidence.

### Remaining evidence

WDBC now has all five planned main folds. This completion does not close the entire A2 cross-dataset stage. Adult main comparison remains runtime-gated by the original training-side projection of 243.82 hours. Full R09, the equal-sampling-budget evidence and the complete thesis remain open. The accepted A3 HTRU2 and A4 synthetic closures retain their defined scopes. Work stops after the authorized fold 4 report and archive.

Evidence: qa/a2_20261105/daily_summary.json, audit_wdbc_main_direct.json, unchanged_resume.json and preservation.json; configs/comparison.json; protocols/freeze_manifest.json. The delivery manifest and external archive receipt provide exact source hashes.
