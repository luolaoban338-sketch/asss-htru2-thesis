> Dated public research copy, prepared 11 November 2026. The source date and scientific findings are retained. Current progress is stated in the update README: Adult confirmation timing passed on 10 November; its main experiment has not run. Local source links are shown as source references because this is a curated inspection snapshot.

# A Unified Framework for Robust Strategic Sample and Feature Selection

7 November 2026

Revision applying review items REV01 to REV05

This stage manuscript brings together the Methods, Results and Discussion for the completed HTRU2 and WDBC development comparisons, the HTRU2 feature baselines and the four synthetic diagnostic settings. The central finding is a measured tradeoff between training-matrix compression and predictive performance. Joint selection retains about 35% of HTRU2 training cells and 15% of WDBC training cells, while exceeding the outer F1 loss tolerance in two of five folds on each dataset. The review decisions concern the interpretation of these tradeoffs, the remaining Adult and matched-sampling evidence, and the scope required for the thesis.

The review evidence index (source: `outputs/thesis_revision_adult_prep_20261107/exports/thesis_T5_20261106/support/Evidence_and_exhibit_index.md>`) connects these sections to their source records. The remaining evidence register (source: `outputs/thesis_revision_adult_prep_20261107/exports/thesis_T5_20261106/support/Missing_evidence_and_review_decisions.md>`) identifies which claims still need results or an author and advisor decision. HTRU2 feature-comparison and synthetic-diagnostic closures retain their defined scope; the full R09 requirement and thesis remain open.

## Related work

Reducing a training matrix involves two decisions: which observations to retain and which original variables to measure or model. These decisions have different objectives in the literature. A row subset may approximate the training objective, improve a validation criterion, or identify valuable observations. A feature subset may provide a sparse predictor, reduce redundancy, or retain variables with sufficient measured importance. The distinction matters for this thesis because a smaller matrix alone does not establish acceptable predictive loss. The relevant comparison concerns the retained matrix, the performance criterion used during selection, and the performance subsequently observed on held-out data.

Training-set selection can be formulated around either optimization fidelity or validation performance. CRAIG constructs weighted subsets whose gradients approximate those of the full training set. Its subset-size objective is tied to a gradient-error tolerance, and its convergence analysis depends on the stated optimization assumptions. GLISTER instead formulates subset selection as a bilevel problem: parameters are learned from a training subset, while the outer objective evaluates a separate validation set under a cardinality budget. Its practical algorithm alternates subset updates and parameter learning. These objectives explain why the term “representative sample” should be qualified: representing a full gradient and improving validation loss are different selection criteria. Neither procedure, as formulated in these papers, selects a subset of input variables. [1,2]

Feature selection has a similarly important distinction between model sparsity and an explicitly controlled feature budget. The lasso constrains the absolute magnitudes of model coefficients and can set some coefficients to zero; the constraint controls shrinkage rather than fixing the number of selected variables. The original paper also considers extension beyond least-squares regression. By contrast, minimum-redundancy maximum-relevance selection, or mRMR, builds a feature set by balancing information about the response against dependence among selected features. Its incremental criterion and subsequent wrapper stage distinguish relevance ranking from evaluation by a predictive model. For the present comparison, a fixed number of variables chosen from an L1 coefficient ranking must therefore be described as a specified ranking procedure, rather than assumed to reproduce the lasso's natural support. [3,4]

Shapley-based methods differ in their players and utility functions. Data Shapley assigns value to training observations by averaging their marginal effects across training-data coalitions, conditional on the learning algorithm and performance measure. Its Monte Carlo and gradient approximations address the cost of repeated coalition evaluation. SAGE assigns importance to input features through a loss-based measure of predictive power. In its model-based formulation, missing features are integrated out from a trained predictor; this differs from retraining a separate predictor for each feature subset. Consequently, a shared Shapley formula does not make row valuation, feature attribution, and sample-conditioned feature selection interchangeable. Their results must be interpreted through the coalition construction, the utility, and the approximation actually used. [5,6]

PowerSHAP turns feature attribution into a selection rule by comparing observed feature impacts with those of an added random variable. Repeated fits and out-of-sample absolute SHAP values support this comparison, while its automatic mode uses a power calculation to determine further iterations. The resulting feature count is determined by the rule rather than fixed in advance. This makes native PowerSHAP a useful comparison for a different decision policy, but it does not establish an equal-column-budget comparison with a selector constrained to retain four variables. The distinction should remain visible in both the methods and the results tables, alongside the selector's own fitting cost. [7]

Joint instance and feature selection also has direct precedents. Mac Parthaláin and Jensen formulate simultaneous selection using fuzzy-rough bireducts. Their algorithm alternates feature retention and instance removal to satisfy discernibility clauses. García-Pedrajas and colleagues subsequently combine sequential forward feature search with instance selection in SI(FS)². Its published description establishes simultaneous reduction, and a retrieved passage of the institutional pre-proof defines a fitness combining nearest-neighbour performance and data reduction. That passage measures reduction using the product of retained row and feature fractions. Thus, neither selecting both axes nor expressing matrix size through this product is, by itself, a defensible novelty claim for the present work. The accessible SI(FS)² passage supports this limited comparison; it does not substitute for a complete audit of that implementation. [8,9]

This thesis is positioned around a narrower empirical question: under a common, prespecified predictive-loss tolerance, how do row selection, feature selection, joint selection, and sequential selection trade performance against retained matrix size? In the established protocol, feature-subset utilities are evaluated after a sampler has selected training observations. Contribution estimates therefore depend on the sampling condition. Inner validation selects a feasible configuration, and an outer fold evaluates the resulting procedure on complete held-out rows. This design provides a common basis for studying the coupling between the two decisions. It does not imply that the proposed contribution score is universally preferable to sparse regression, attribution-based selection, or previous joint selectors.

Several comparisons must remain separate to preserve that interpretation. Adaptive compression policies answer a different question from fixed-budget feature comparisons. Restricted joint policies that can return to all training rows do not, by themselves, establish an equal-sampling-budget comparison. Furthermore, sharing candidate evaluations across decision rules reduces duplicated computation but cannot measure each rule's standalone runtime. The accompanying experiments therefore need to report actual row fractions, feature counts, performance differences, and tolerance violations, together with the scope of reused computation. Stability and perturbation studies provide additional diagnostics under their specified resampling and noise settings; they do not convert development-fold comparisons into independent confirmation. These boundaries allow the literature to motivate the evaluation without treating prior methods' guarantees or benchmark results as evidence for this thesis's own performance.

{{< pagebreak >}}

# Methods

## HTRU2 training observations variables and prediction

The selection problem asks how much of a training matrix a classifier can discard while its inner-validation positive-class F1 remains within 0.01 of the full-data reference. Outer folds separately evaluate the realized loss. The framework chooses both the observations that supply training evidence and the original variables through which the classifier sees them. If it retains a fraction of the rows and k of the original eight variables, their product (row fraction × k/8) gives the fraction of training cells passed to the final fit. Every held-out observation still receives a prediction.

The [HTRU2 dataset, DOI 10.24432/C5DK6R](https://doi.org/10.24432/C5DK6R) contains 17,898 pulsar candidates, including 1,639 positives, with eight numerical measurements per candidate. Exact predictor vectors defined groups for splitting. No duplicate vectors occurred, but the table lacks astronomical source identifiers. The experiment therefore evaluates held-out candidates. It uses the archived column order throughout because the dataset descriptions disagree on the order of skewness and kurtosis.

## Feature contributions conditional on selected rows

We compared all training rows with three half-sample alternatives. The half-samplers rounded the retained count up and allocated places to each class by the largest-remainder rule. One sampled randomly within classes. The other two clustered each class and selected observations by their within-cluster distance ranks. Ordinary geometry used means and standard deviations for scaling. Robust geometry used medians and interquartile ranges, with clipping at the first and ninety-ninth percentiles. All geometry was estimated on training rows.

Adaptive Stable Shapley Selection (ASSS) values a feature by its marginal change in mean positive-class F1 across three utility-validation folds. Within each utility fold, the selected training rows, validation rows and model seed remain fixed across feature subsets. Each nonempty subset receives a fresh classifier, with imputation and scaling fitted only on its selected training rows and variables. The empty subset uses a constant probability equal to the positive-class proportion in those selected training rows. Changing the training sample can therefore change the measured value of a feature.

## HTRU2 feature path and budget selection

For the HTRU2 adaptive comparison, five outer folds measured prediction. Within each outer training fold, three inner folds compared budgets. Inside each inner training partition, each of four stratified 80% subsamples defined a contribution game using three utility-validation folds. For each game, 256 random feature permutations estimated marginal F1 contributions. All 256 possible feature subsets yielded a separate exact diagnostic. Only the permutation estimates entered budget selection.

The three switches C, S and R specify a feature path. C subtracts estimated permutation noise from the variance across the four contribution estimates and truncates negative values to zero. S subtracts 0.5 times the resulting standard deviation from the mean contribution when enabled. In the ablation, disabling C sends the uncorrected variance to S. R penalizes redundant pairs by 0.1/8 times their absolute Spearman correlation on selected training rows. At each step the algorithm adds the variable with the largest score increment. Ties follow the original column order, and all eight prefixes remain eligible.

Four sampling choices, two values for each penalty and eight feature counts yielded 128 candidates per inner partition. Feasible candidates had mean inner F1 at least equal to the full-data reference minus 0.01. Among these, the procedure minimized retained cells, then k, then favored higher F1 and finally the earlier candidate in a fixed order. The full-data candidate supplied a fallback. Each outer training pool rebuilt the selected rows and features before evaluation on its complete test fold.

## Comparators and measurement

The four principal rules allowed neither choice, row choice alone, feature choice alone, or both choices together. A fifth rule chose the sampler with all features first, then chose features within that sampler using the original F1 cutoff. For these comparisons, the final classifier was L2 logistic regression with regularization parameter C = 1, an intercept, the lbfgs solver, a maximum of 5,000 iterations, optimization tolerance 10⁻⁶, no class weighting and a probability threshold of 0.5. This regularization parameter is distinct from the contribution-noise correction switch C. We report equal-weight fold means for positive-class F1 and area under the receiver operating characteristic curve (AUC). The outer training folds overlap. The 0.01 cutoff is an inner selection criterion and an outer descriptive check. Methods and verification protocol (source: `outputs/thesis_revision_adult_prep_20261107/thesis_htru2_20261001/docs/10_htru2_methods_en.md>`)

## WDBC extension and dataset specific settings

The Wisconsin Diagnostic Breast Cancer dataset (WDBC) contributes 569 observations and 30 numerical original variables. Malignant diagnosis is encoded as the positive class (212 observations), and benign diagnosis as the negative class (357 observations). Its five outer folds contain 455 training and 114 test observations in folds 0 to 3, and 456 training and 113 test observations in fold 4. Duplicate predictor vectors define split groups; all 569 WDBC vectors are distinct. These groups describe predictor identity and do not establish person-level independence.

The WDBC comparison uses four stratified 80% resamples, 64 permutations per contribution game, three utility folds and three inner folds. It evaluates every feature-path prefix from one to 30 variables, with the same four sampler choices and penalty grids of 0 or 0.5 for contribution variability and 0 or 0.1 for redundancy. It retains the L2 logistic predictor, classification threshold, inner F1 tolerance and deterministic budget ordering. Unlike the eight-variable HTRU2 analysis, WDBC exceeds the frozen limit of eight variables for exact enumeration. The recorded skip therefore supplies no high-dimensional approximation-accuracy result.

Eight WDBC decision arms share the same frozen partitions, model seeds and deduplicated candidate scores. Full, sample only, feature only and joint define the four principal comparisons. Sequential selection fixes a sampler before selecting a feature budget. The three restricted joint arms constrain the half-sampler to random, ordinary geometry or robust geometry, while allowing fallback to all training rows. Actual row fractions must therefore be reported alongside the arm labels. Frozen WDBC protocol (source: `outputs/thesis_revision_adult_prep_20261107/thesis_crossdataset_20261014/docs/protocol_en.md>`) and comparison configuration (source: `outputs/thesis_revision_adult_prep_20261107/thesis_crossdataset_20261014/configs/comparison.json>`).

{{< pagebreak >}}

# Results

## RQ1 HTRU2 prediction at a smaller training matrix

Joint selection kept 35.00% of training cells and lost 0.0105 F1 on average against full data. The largest losses occurred in folds 1 and 4. Row selection alone halved the matrix with a mean loss of 0.0011 and stayed within tolerance in all five folds. Feature selection alone kept 70.00% of cells and exceeded tolerance in fold 4.

Table 1. HTRU2 prediction and retained training cells by outer fold

| Fold | Rule | F1 | AUC | Rows (%) | k | Cells (%) | Loss |
|-----|--------------|--------|--------|----------|----|----------|---------|
| 0 | Full | 0.8692 | 0.9750 | 100.0000 | 8 | 100.0000 | 0.0000 |
| 0 | Sample only | 0.8630 | 0.9748 | 50.0000 | 8 | 50.0000 | +0.0062 |
| 0 | Feature only | 0.8616 | 0.9762 | 100.0000 | 5 | 62.5000 | +0.0076 |
| 0 | Joint | 0.8598 | 0.9752 | 50.0000 | 6 | 37.5000 | +0.0093 |
| 1 | Full | 0.8836 | 0.9741 | 100.0000 | 8 | 100.0000 | 0.0000 |
| 1 | Sample only | 0.8874 | 0.9725 | 50.0000 | 8 | 50.0000 | -0.0037 |
| 1 | Feature only | 0.8810 | 0.9745 | 100.0000 | 6 | 75.0000 | +0.0027 |
| 1 | Joint | 0.8694 | 0.9735 | 50.0000 | 6 | 37.5000 | +0.0142† |
| 2 | Full | 0.8707 | 0.9792 | 100.0000 | 8 | 100.0000 | 0.0000 |
| 2 | Sample only | 0.8664 | 0.9797 | 50.0035 | 8 | 50.0035 | +0.0043 |
| 2 | Feature only | 0.8649 | 0.9804 | 100.0000 | 5 | 62.5000 | +0.0059 |
| 2 | Joint | 0.8687 | 0.9797 | 50.0035 | 5 | 31.2522 | +0.0021 |
| 3 | Full | 0.8639 | 0.9766 | 100.0000 | 8 | 100.0000 | 0.0000 |
| 3 | Sample only | 0.8639 | 0.9764 | 50.0035 | 8 | 50.0035 | 0.0000 |
| 3 | Feature only | 0.8567 | 0.9732 | 100.0000 | 6 | 75.0000 | +0.0073 |
| 3 | Joint | 0.8576 | 0.9771 | 50.0035 | 6 | 37.5026 | +0.0063 |
| 4 | Full | 0.8841 | 0.9747 | 100.0000 | 8 | 100.0000 | 0.0000 |
| 4 | Sample only | 0.8855 | 0.9752 | 50.0000 | 8 | 50.0000 | -0.0013 |
| 4 | Feature only | 0.8673 | 0.9745 | 100.0000 | 6 | 75.0000 | +0.0169† |
| 4 | Joint | 0.8634 | 0.9765 | 50.0000 | 5 | 31.2500 | +0.0208† |
| Mean | Full | 0.8743 | 0.9759 | 100.0000 | 8.0 | 100.0000 | 0.0000 |
| Mean | Sample only | 0.8732 | 0.9757 | 50.0014 | 8.0 | 50.0014 | +0.0011 |
| Mean | Feature only | 0.8663 | 0.9758 | 100.0000 | 5.6 | 70.0000 | +0.0081 |
| Mean | Joint | 0.8638 | 0.9764 | 50.0014 | 5.6 | 35.0010 | +0.0105 |

Loss is full-data F1 minus arm F1. A dagger (†) marks a loss above 0.01 before rounding. Rows and cells are percentages of the outer training matrix. The last four rows give equal-weight fold means. Fold records (source: `outputs/thesis_revision_adult_prep_20261107/thesis_htru2_20261001/runs/joint-htru2-pilot-20261001-v1/outer_results.csv>`) · Source table and checks (source: `outputs/thesis_revision_adult_prep_20261107/exports/thesis_T1_20261103/HTRU2_four_cell.md>`)

{{< pagebreak >}}

## RQ1 WDBC prediction and compression

The completed WDBC comparison extends the measured compression tradeoff to a second real dataset. Joint selection retained 14.69% of training cells and had mean F1 0.9582, compared with 0.9649 for full data. Its mean paired F1 difference was -0.0067. Folds 0 and 2 exceeded the 0.01 outer loss tolerance. Sample-only selection had a higher mean F1 than full data, but it also exceeded the loss tolerance in two individual folds.

Table 2 WDBC equal weight means across five development folds

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

Rows and cells are percentages of each complete outer training matrix; k counts original variables out of 30. Cell retention is calculated from the actual row fraction times k/30 within each fold and then averaged. Breaches use the unrounded full-data-minus-arm F1 loss above 0.01 plus 1e-12. Table 1 and Table 2 compare different datasets separately; their scores are not pooled.

## RQ3 WDBC joint and sequential decisions

Joint selection retained 7.02 percentage points fewer cells than sequential selection on average, with a mean F1 difference of -0.0001. Folds 2 and 3 account for the additional compression. The five paired contrasts reveal that the almost equal mean F1 accompanies different feature budgets rather than identical decisions throughout.

Table 3 WDBC joint minus sequential differences

| Fold | F1 difference | AUC difference | Rows difference pp | Cells difference pp |
| --- | --- | --- | --- | --- |
| 0 | +0.0000 | +0.0000 | +0.00 | +0.00 |
| 1 | +0.0000 | +0.0000 | +0.00 | +0.00 |
| 2 | -0.0006 | -0.0023 | +0.00 | -25.05 |
| 3 | +0.0000 | -0.0013 | +0.00 | -10.02 |
| 4 | +0.0000 | +0.0000 | +0.00 | +0.00 |

Negative cell differences indicate additional compression by the joint procedure. Percentage point differences are denoted pp. Full arm-by-fold values and F1 differences from full data remain in the accepted five-fold report (source: `outputs/thesis_revision_adult_prep_20261107/thesis_crossdataset_20261014/exports/A2_WDBC_Fivefold_20261105.md>`). All 40 arm-fold records were independently checked from saved probabilities and selected rows for this integration.

The restricted joint arms used all training rows in three of 15 arm-fold records: the joint robust arm in folds 1 and 2, and the joint random arm in fold 4. These allowed fallbacks prevent treating the arm names as evidence of equal sampling budgets. The five outer training sets overlap, and the shared candidate search does not measure each method's standalone runtime.

{{< pagebreak >}}

## RQ2 HTRU2 changes under different training rows

With all eight features held fixed, the random, ordinary-geometry and robust-geometry half-samplers had mean F1 of 0.8732, 0.8739 and 0.8695, respectively. All three retained the same class counts within a fold. In fold 0, for example, each retained 645 positives and 6,514 negatives from 1,290 positives and 13,028 negatives. Performance could thus vary despite equal class coverage. These fixed-sampler runs used their own seeds. The sample-only arm in Table 1 chose among samplers on the inner folds.

The pulse-profile mean illustrates how sampling affected a feature's measured contribution. In a training-only diagnostic on fold 0, the exact Shapley contribution of this variable had a standard deviation across four subsamples of 0.00183 for random half, 0.00095 for ordinary half and 0.00102 for robust half. At 16 permutations per game the corresponding noise-corrected estimates were 0.03752, zero and zero. A zero here records truncation of an estimate. The exact contributions still varied across the four subsamples. Contribution diagnostic (source: `outputs/thesis_revision_adult_prep_20261107/thesis_htru2_20261001/runs/feature-htru2-pilot-20261001-v1/feature_summary.csv>`)

Ten stratified resamples of the first outer training pool repeated the entire inner search. Joint selection yielded Nogueira stability of 0.4974 and mean pairwise Jaccard similarity of 0.7352. Sequential selection yielded 0.6549 and 0.8519. These workflows varied both data and algorithm seeds, and the 45 Jaccard pairs shared resamples. The two measures describe the reproducibility of the selected feature sets under that combined variation. [Stability measure](https://jmlr.org/papers/v18/17-514.html) · Resampling results (source: `outputs/thesis_revision_adult_prep_20261107/thesis_htru2_20261001/runs/stability-htru2-pilot-20261001-v1/summary.json>`)

## RQ3 Joint and sequential choices on HTRU2

Sequential selection kept six variables in every fold, retained 37.50% of cells and achieved mean F1 of 0.8653. Joint selection used five variables in folds 2 and 4. With fewer cells, joint selection scored 0.0067 higher in fold 2 and 0.0039 lower in fold 4 than sequential selection. In folds 0, 1 and 3 the cell budgets matched, with joint-minus-sequential F1 differences of 0.0000, −0.0111 and +0.0010. Sequential selection exceeded the full-data loss tolerance only in fold 4. Both rules consulted the same 1,920 logical inner candidate-score records (128 candidates × three inner folds × five outer folds). This count is not the number of distinct fitted models, because candidates may reuse cached evaluations. These records identify paired prediction and compression differences. Separate execution times for the two rules were not measured. Paired decisions (source: `outputs/thesis_revision_adult_prep_20261107/thesis_htru2_20261001/runs/joint-htru2-pilot-20261001-v1/joint_sequential.csv>`)

{{< pagebreak >}}

## RQ4 Perturbations and fixed budget ablations

At 5% and 10% training contamination, one predictor per affected row shifted by ten clean-training interquartile ranges in a random direction. Three draws per fold used nested event sets. Labels and test rows stayed clean. The joint rule kept its clean sampler, budgets and penalties but rebuilt selected identities. After averaging draws within folds, joint F1 fell from 0.8638 clean to 0.8281 at 5% and 0.8167 at 10%. Full data reached 0.8228 at 10%.

With half the rows and four features fixed, removing C, S or R left F1 unchanged in all 135 paired comparisons, although 22 full feature orders changed. Against full data, all-enabled losses exceeded 0.01 in 11/15 random, 9/15 ordinary and 10/15 robust fold-seed pairs. In ten fixed-budget resamples all variants retained the same four pulse-profile variables. Nogueira stability and mean Jaccard were both 1. Exact-prefix checks retained the estimated Monte Carlo variance. Each of at most two ablation workers used one numerical thread. Ablation and stress-test methods (source: `outputs/thesis_revision_adult_prep_20261107/thesis_htru2_20261001/docs/23_methods_evidence_consistency_20261013.md>`)

## Feature baselines on HTRU2

With all training rows, all-enabled ASSS (ASSS111) and mean Shapley selected the same four pulse-profile variables in every fold. Their logistic predictors had identical fold scores, with means of 0.861801 for F1, 0.973623 for AUC and 0.915579 for average precision. [A1]

At four features, L1 ranking (mean F1 0.872498) and forward cross-validation (0.867402) exceeded ASSS111 in every fold. Random-forest impurity ranking (0.863574) exceeded it in three. ASSS111 beat random selection in only two folds, despite its mean paired advantage of 0.002699. [A2]

All eight features gave mean F1 of 0.874328. Native PowerSHAP retained 4, 6, 6, 6 and 4 features. Its mean F1 of 0.868810 exceeded ASSS111 by 0.007009, with higher F1 in four folds. Both the selection rule and the feature count varied. [A3]

The eight methods shared training pools, test rows and final-model seeds within folds. The overlapping folds came from previously examined data. ASSS111 and mean Shapley also shared contribution estimates, so their timings cannot establish independent end-to-end costs. [A4]

{{< pagebreak >}}

## Synthetic tests of the three selection modules

Known signal identities allowed separate tests of variance estimation, weak-signal retention, group coverage and interaction recovery. Each of four settings used eight independently generated datasets with six predictors, 600 training observations and 2,000 test observations per dataset. Results first averaged six resamples within a dataset and then weighted datasets equally. The two XOR learners shared all eight datasets and all 48 resampling schedules. [B1]

At both 16 and 64 permutations per game, subtracting estimated permutation noise reduced the mean squared error of the variance estimate on every dataset in each setting. Its nonnegative version also reduced error. In the signal-and-noise logistic model at 64 permutations, mean squared error fell from 3.212 × 10⁻⁷ uncorrected to 2.021 × 10⁻⁷ after subtraction and 1.740 × 10⁻⁷ after clipping. The target was the sample variance of exact Shapley contributions across the four realized games. [B2]

Selection used 64 permutations per game and a primary budget of two features. In the logistic signal-and-noise setting, adding the contribution-variability penalty S, with C and R enabled, increased the weak-signal omission rate by 0.104167 on average. Nogueira stability fell on four datasets and tied on four, for a mean change of −0.118750. [B3]

In the two-group setting, adding R with C and S enabled increased mean signal-group coverage by 0.145833. Six datasets gained coverage and two tied. The selected noise-feature count was unchanged on all eight datasets. With two available places, selecting fewer members from a group already represented made room for the other group. [B4]

The all-enabled random-forest workflow recovered both interacting XOR features in every resample and achieved mean test F1 of 0.943483. Logistic regression recovered both in a mean fraction of 0.083333 of resamples and achieved F1 of 0.494482. Each learner both valued feature subsets and predicted on test data. The contrast therefore applies to the paired workflows as a whole. [B5]

{{< pagebreak >}}

# Discussion

On HTRU2 the joint rule discarded nearly two thirds of the training matrix, but two outer folds lost more F1 than the inner selection rule allowed. Row selection alone came closer to the full-data predictor. Holding the number of features fixed changed the question. Removing any single ASSS module left the four-feature set unchanged, yet L1 ranking and forward selection used the same four places to obtain higher F1 in every fold. The agreement between ASSS111 and mean Shapley therefore depends on which variables their rules ranked highly. The number of places alone cannot account for it. [D1]

The synthetic studies identify where the selection rule can succeed or fail. Subtracting permutation noise improved estimation of the variance across the four measured games. Penalizing the remaining variation nevertheless made a weak signal easier to omit. In a different design with correlated signal groups, penalizing redundancy allowed the second available place to cover a different group. Each result answers a distinct question about the path from an estimated contribution to a selected feature. None establishes a predictive benefit from that module on HTRU2. [D2]

Two comparisons would distinguish explanations that remain compatible with the observations. Holding the utility learner fixed on XOR while varying the predictor, or reversing those roles, would separate selection from prediction. A prospective study on a second real dataset, with the same rows, feature budget and final model for competing selectors, would test whether the HTRU2 ordering persists. [D3]

The second-dataset extension would require a separately prespecified matched-budget feature-selector comparison on WDBC. The current R09 proposal first addresses HTRU2 with shared random half-samples and k=4. Neither comparison has been executed, and the completed WDBC adaptive-policy comparison addresses a different endpoint.

## Cross dataset interpretation and remaining evidence

Both HTRU2 and WDBC show that joint selection can produce a smaller final training matrix with a measurable predictive tradeoff. Their outer tolerance violations mean that inner feasibility is not an outer noninferiority guarantee. The different feature dimensions and permutation budgets also prevent attributing cross-dataset differences solely to a sampler or contribution module. The supported result is the observed behavior of each frozen workflow in its own development comparison.

The contribution-conditioned design relates to prior feature and row valuation methods through the definition of the game, rather than through a claim that joint reduction itself is new [5,6,8,9]. Fixed-count feature rankings, native PowerSHAP and adaptive row-column policies make different selection decisions. Their comparisons are informative when the budget, sampling condition and final learner are stated explicitly. The same distinction explains why the all-row HTRU2 feature baselines do not complete the proposed comparison under shared half-samples.

WDBC now has all five planned outer folds. Adult main execution remains absent because its training-side runtime projection was 243.82 hours, above the 48-hour warning threshold. The original three-dataset target therefore remains incomplete. The historical threshold-based Shapley baseline also remains unresolved: the signed report states a minimum contribution constraint and K=5, but the numerical threshold and matching original executable were not recovered. The prospective R09 comparison uses a common random-half sampler and k=4; it is a proposed new protocol, not a reconstruction of that missing historical rule.

The next comparisons should resolve specific uncertainties. A matched-sampling feature baseline would test whether the observed HTRU2 selector ordering persists under the same retained rows. A reduced Adult design requires a new frozen timing gate before it can test transfer to mixed numerical and categorical variables. Separate full-process timing is needed to assess practical speed, since shared search and fewer final training cells do not establish an end-to-end speedup.



{{< pagebreak >}}

# References


1. Mirzasoleiman, B., Bilmes, J. and Leskovec, J. (2020). Coresets for Data-efficient Training of Machine Learning Models. *Proceedings of the 37th International Conference on Machine Learning*, PMLR **119**, 6950–6960. [Official proceedings and paper](https://proceedings.mlr.press/v119/mirzasoleiman20a.html).
2. Killamsetty, K., Sivasubramanian, D., Ramakrishnan, G. and Iyer, R. (2021). GLISTER: Generalization based Data Subset Selection for Efficient and Robust Learning. *Proceedings of the AAAI Conference on Artificial Intelligence* **35**(9), 8110–8118. [DOI: 10.1609/aaai.v35i9.16988](https://doi.org/10.1609/aaai.v35i9.16988).
3. Tibshirani, R. (1996). Regression Shrinkage and Selection via the Lasso. *Journal of the Royal Statistical Society: Series B (Methodological)* **58**(1), 267–288. [DOI: 10.1111/j.2517-6161.1996.tb02080.x](https://doi.org/10.1111/j.2517-6161.1996.tb02080.x).
4. Peng, H., Long, F. and Ding, C. (2005). Feature Selection Based on Mutual Information: Criteria of Max-Dependency, Max-Relevance, and Min-Redundancy. *IEEE Transactions on Pattern Analysis and Machine Intelligence* **27**(8), 1226–1238. [DOI: 10.1109/TPAMI.2005.159](https://doi.org/10.1109/TPAMI.2005.159).
5. Ghorbani, A. and Zou, J. (2019). Data Shapley: Equitable Valuation of Data for Machine Learning. *Proceedings of the 36th International Conference on Machine Learning*, PMLR **97**, 2242–2251. [Official proceedings and paper](https://proceedings.mlr.press/v97/ghorbani19c.html).
6. Covert, I., Lundberg, S. M. and Lee, S.-I. (2020). Understanding Global Feature Contributions With Additive Importance Measures. *Advances in Neural Information Processing Systems* **33**. [Official proceedings and paper](https://proceedings.neurips.cc/paper/2020/hash/c7bf0b7c1a86d5eb3be2c722cf2cf746-Abstract.html).
7. Verhaeghe, J., Van Der Donckt, J., Ongenae, F. and Van Hoecke, S. (2023). Powershap: A Power-Full Shapley Feature Selection Method. *Machine Learning and Knowledge Discovery in Databases: ECML PKDD 2022*, LNCS **13713**, 71–87. [DOI: 10.1007/978-3-031-26387-3_5](https://doi.org/10.1007/978-3-031-26387-3_5). Publication year is 2023; the conference and arXiv preprint are from 2022.
8. Mac Parthaláin, N. and Jensen, R. (2013). Simultaneous feature and instance selection using fuzzy-rough bireducts. *2013 IEEE International Conference on Fuzzy Systems*, 1–8. [DOI: 10.1109/FUZZ-IEEE.2013.6622500](https://doi.org/10.1109/FUZZ-IEEE.2013.6622500); [institutional full text](https://pure.aber.ac.uk/ws/files/2450694/F_1392.pdf).
9. García-Pedrajas, N., Romero del Castillo, J. A. and Cerruela-García, G. (2021). SI(FS)²: Fast simultaneous instance and feature selection for datasets with many features. *Pattern Recognition* **111**, 107723. [DOI: 10.1016/j.patcog.2020.107723](https://doi.org/10.1016/j.patcog.2020.107723); [institutional pre-proof](https://helvia.uco.es/xmlui/bitstream/10396/26740/3/si_fs.pdf). Full-text access was partial during this review.
