# Thesis progress report: 1-29 October 2026

30 October 2026 | Feiyang Luo | Supervisor: Chi Zhao

## Thesis Progress Report

A Unified Framework for Robust Strategic Sample and Feature Selection



Feiyang Luo   |   Supervisor: Chi Zhao

Reporting interval: 1-29 October 2026   |   Meeting: 30 October 2026

**Current position.** The experiments now support a more specific account of when the framework helps and when it fails. Compression is measurable; a general predictive or stability advantage has not been established.

### 1. Progress since the previous meeting

The previous presentation, dated 30 September, focused on COMPAS and preliminary joint sample-feature selection. Its pending work concerned complete-workflow stability, runtime, contamination, additional datasets and conventional feature-selection baselines. This report covers the work completed after that presentation. [E1]

| Workstream | New progress | Current boundary |
| --- | --- | --- |
| HTRU2 validation | Five-fold compression study; complete-workflow stability; training-contamination stress test | Development evidence on one dataset |
| Fixed-budget ablation | Eight C/S/R combinations; predictive and stability checks | No observed F1 increment from the modules |
| Cross-dataset extension | WDBC folds 0-2; Adult runtime feasibility probe | WDBC 2 folds deferred; Adult main run unstarted |
| A3 comparisons | Traditional selectors and native PowerSHAP: five folds each | No overall ASSS advantage |
| A4 mechanisms | Signal/noise LR, correlated-group LR, XOR RF and XOR LR delivered | Combined stage synthesis remains pending |

### 2. Main findings

**Compression has a performance cost.** HTRU2 joint selection retained 35% of training matrix cells, with mean F1 0.8638 versus 0.8743 for full data. Two of five folds exceeded the 0.01 loss reference. [E2]

**The stronger baselines matter.** ASSS obtained mean F1 0.861801 in A3, equal to Mean Shapley and below L1 top-four (0.872498) and native PowerSHAP (0.868810). Native PowerSHAP selected four or six features, so it is not an equal-size comparison. [E5-E6]

**Module effects are conditional.** Variance correction C reduced the specified diagnostic MSE; redundancy penalty R helped in correlated groups; stability penalty S did not consistently improve stability or prediction. [E7-E10]

### Purpose of this meeting

Agree on the defensible thesis contribution and the priority of the remaining work: consolidate the completed evidence, explain the adverse S results, and decide whether further external validation is needed. This report adds no new model fitting or confirmatory statistical analysis.

Reading guide: Sections 3-4 revisit the full framework; Sections 5-7 isolate selector comparisons and mechanisms. These stages use different designs and should not be pooled into one performance estimate.

## 3. Full-framework validation on HTRU2

### Method and evaluation design

The thesis studies training sample-feature compression with nested validation. ASSS (Adaptive Stable Shapley Selection) is the feature-selection component: it estimates contributions through subset retraining on training-derived samples, corrects variance estimates, and constructs a greedy path with contribution-variability and redundancy penalties. Validation and test sets are evaluated in full; they are not subsampled. [E2-E3]

HTRU2 contains 17,898 observations, eight numerical predictors and 1,639 positives (9.16%). The full-framework experiment used five frozen outer folds, a common logistic-regression configuration and a decision threshold of 0.5. Inner validation selected compression budgets. The following results are equal-fold descriptive means.

### Compression and prediction

| Method | Mean F1 | Cells retained | Mean features | Loss > 0.01 |
| --- | --- | --- | --- | --- |
| Full data | 0.8743 | 100% | 8.0 | 0 / 5 |
| Sample only | 0.8732 | 50% | 8.0 | 0 / 5 |
| Feature only | 0.8663 | 70% | 5.6 | 1 / 5 |
| Joint selection | 0.8638 | 35% | 5.6 | 2 / 5 |
| Sequential selection | 0.8653 | 37.5% | 6.0 | 1 / 5 |

Cells retained measures the selected training matrix relative to the full training matrix. It is a compression measure, not a measured runtime speedup. Loss is full-data F1 minus method F1; 0.01 is a descriptive reference, not a validated noninferiority margin.

### Complete-workflow stability

Ten overlapping stratified 80% resamples of outer-fold-0 training data repeated the inner budget decision and feature construction. Joint selection had Nogueira stability 0.4974 and mean Jaccard 0.7352; sequential selection had 0.6549 and 0.8519. Thus, the adaptive joint workflow did not show greater feature-set stability in this check. No outer-test prediction was used in this stability package.

### Training-only contamination

| Method | Clean F1 | 5% contamination | 10% contamination |
| --- | --- | --- | --- |
| Full data | 0.8743 | 0.8341 | 0.8228 |
| Joint selection | 0.8638 | 0.8281 | 0.8167 |
| Sequential selection | 0.8653 | 0.8315 | 0.8172 |

The stress test fixed clean-fold strategies and budgets, then rebuilt selected identities after perturbing training predictors. It did not repeat inner budget selection. Labels and outer-test rows were unchanged.

A corrupted training row received a signed ten-IQR offset in one numerical predictor. Joint selection lost less F1 from its own clean baseline at 10% contamination (0.0470 versus 0.0515 for full data), but its contaminated F1 remained lower. This does not establish superior robustness or a realistic model of astronomical measurement noise.

Source: [E2], Tables 1-4 and associated methods. The 0.8638 joint result belongs to adaptive sample-feature compression; the 0.861801 A3 result on page 4 belongs to a separate fixed-k selector comparison.

## 4. Ablation and cross-dataset extension

### Fixed-budget ablation revealed an uninformative selected-set regime

The HTRU2 ablation fixed the sample ratio at 0.5 and feature count at four, with B=4 contribution samples and M=256 permutations. Eight logical C/S/R combinations were evaluated with shared paired inputs. C denotes Monte Carlo variance correction, S the contribution-variability penalty and R the redundancy penalty. [E3]

**All 135 primary seed-level F1 differences were exactly zero.** Within each predictive comparison, all eight variants selected the same four-feature set. Removing modules changed the full greedy path in 22/135 predictive and 14/90 stability comparisons, without changing the selected set. All variants achieved Nogueira and mean Jaccard of 1 in the fixed-budget stability check.

These findings show no observed incremental predictive benefit at the frozen HTRU2 operating point. They do not prove mathematical equivalence or universal irrelevance of the modules. Perfect stability under a fixed budget also does not replace the weaker complete-workflow stability result, where the budget decision was repeated.

All-enabled mean F1: random-half 0.8601; ordinary-half 0.8592; robust-half 0.8594. Full-data reference: 0.8743. The fixed-budget stage retained approximately 25.0007% of cells. These settings differ from the adaptive joint experiment on page 2.

### WDBC remains an incomplete development comparison

Only outer folds 0, 1 and 2 have been delivered. Folds 3 and 4 are deferred. The observed-fold means below describe those three folds only. [E4]

| Method | Mean F1 | Mean F1 loss | Loss > 0.01 |
| --- | --- | --- | --- |
| Full data | 0.9672 | 0.000000 | 0 / 3 |
| Sample only | 0.9731 | -0.005870 | 1 / 3 |
| Feature only | 0.9581 | 0.009158 | 2 / 3 |
| Joint selection | 0.9560 | 0.011184 | 2 / 3 |
| Sequential selection | 0.9562 | 0.010990 | 2 / 3 |

Loss uses the unrounded observed-fold means. A favorable average can coexist with a fold-level violation. B=4 and M=64 were prospective cost-control settings; approximation adequacy at 30 predictors has not been established.

### Adult was stopped at the runtime feasibility gate

A bounded 200-fit training-only probe projected 243.82 hours against a 48-hour warning threshold. This was a projection with safety margins, not an observed completed runtime. The Adult main experiment was not run and the official test set was not used. [E4]

The 17 October priority revision therefore emphasized conventional baselines and controlled synthetic diagnostics. Current evidence supports neither broad cross-dataset superiority nor a general computational-efficiency claim.

Sources: [E3] fixed-budget ablation and methods consistency companion; [E4] WDBC delivery, state/resume records and current experiment plan.

## 5. A3: completed five-fold selector comparisons

### Common downstream evaluation

Traditional selectors retain four of eight HTRU2 predictors; all-eight is contextual. All final A3 models use the complete outer-training pool (rho=1); training resamples are internal to contribution estimation. Final evaluation uses the same outer partitions, common L2 logistic regression and threshold 0.5. Native PowerSHAP uses CatBoost for selection and the common LR downstream. Entries are equal-fold means. [E5-E6]

| Selector | Features | Mean F1 | ROC AUC | Mean AP |
| --- | --- | --- | --- | --- |
| ASSS 111 | 4 | 0.861801 | 0.973623 | 0.915579 |
| Mean Shapley | 4 | 0.861801 | 0.973623 | 0.915579 |
| L1 coefficient | 4 | 0.872498 | 0.976793 | 0.922885 |
| RF MDI | 4 | 0.863574 | 0.975345 | 0.921699 |
| Forward CV | 4 | 0.867402 | 0.973314 | 0.922980 |
| Random four | 4 | 0.859102 | 0.973102 | 0.914849 |
| All features (context) | 8 | 0.874328 | 0.975932 | 0.922558 |
| Native PowerSHAP | 4 or 6 | 0.868810 | 0.976313 | 0.923112 |

Mean Shapley is the existing mean_shapley100_alias000 arm, not an additional independently implemented comparator. AUC is ROC AUC; AP is average precision.

### What the comparison establishes

ASSS and Mean Shapley had identical F1 in all five folds. The mean paired ASSS-minus-comparator F1 difference was -0.010697 for L1, -0.001773 for RF MDI, -0.005601 for Forward CV and +0.002699 for Random four. ASSS therefore did not outperform the stronger conventional baselines on this development dataset.

Native PowerSHAP was completed on 29 October. Its selected feature counts were [4, 6, 6, 6, 4], with native iterations [10, 13, 16, 10, 12]. ASSS minus native PowerSHAP averaged -0.00700941 F1, with one positive and four negative fold differences. In two folds the ASSS deficit exceeded 0.01.

### Interpretation and comparability limits

Native PowerSHAP is an authentic implementation with its own stopping and selection behavior; feature counts were neither padded nor truncated. Its advantage cannot be attributed solely to selection quality at a common k, because both selected size and the selection learner differ. The common downstream LR and data partitions improve comparability but do not eliminate these differences.

The five folds are previously examined development partitions of one dataset, with overlapping training sets. No confirmatory superiority, equivalence or noninferiority inference is made. Native feature-screening p-values do not constitute p-values for predictive comparisons between methods.

Sources: [E5] five_fold_metrics and paired_f1_contrasts; [E6] records, native_actual_k, native_iterations and exact mean_paired_F1_contrast. Numerical displays are rounded; source aggregates retain full precision.

## 6. A4: mechanism-specific evidence

### A design that can distinguish the modules

Each layer used eight independent generated datasets, each with 600 training and 2,000 independently generated test rows. Six nested selection resamples per dataset produced 48 selectors per layer. The primary feature count was k=2, with B=4 and main M=64. The independent unit is the generated dataset, not a resample, feature, permutation stream or pair of sets. [E7-E10]

C subtracts estimated Monte Carlo variance from observed contribution variance and clips at zero. S subtracts 0.5 times contribution standard deviation from the mean. R penalizes accumulated absolute Spearman redundancy, scaled by eta/d with eta=0.1. The S contrast is 111 minus 101; the R contrast is 111 minus 110. Structural aliases are not independent methods.

### C reduced finite-budget diagnostic error

| Layer (M=64) | Uncorrected MSE | Raw-corrected | Clipped-corrected |
| --- | --- | --- | --- |
| Signal/noise LR | 3.212e-7 | 2.021e-7 | 1.740e-7 |
| Correlated-group LR | 7.615e-7 | 3.439e-7 | 2.451e-7 |
| XOR RF | 6.125e-7 | 3.797e-7 | 3.154e-7 |
| XOR LR | 1.912e-7 | 1.712e-7 | 1.694e-7 |

Dataset means of diagnostic MSE. Raw and clipped corrections reduced MSE in all eight datasets within each layer at M=16 and M=64. This targets the finite-B sample variance of exact contributions over the same four realized games, not an established population variance.

Twenty diagnostic streams per selector were separate from the selection stream. Exact contributions were evaluated after commitments and did not enter selection. Lower diagnostic error is evidence for this correction mechanism; it does not, by itself, imply better selected subsets or prediction.

### S was adverse in the signal/noise LR setting

The generator used sigmoid(1.5 X1 + 0.5 X2) with four noise predictors. Adding S changed mean Nogueira stability by -0.118750, signal recall by -0.052083 and test F1 by -0.002076. Each endpoint worsened in four datasets and tied in four. Weak-signal omission and noise count each increased by 0.104167. This is evidence against a general stability-benefit claim for S. [E7]

### R helped represent correlated signal groups

With two correlated signal groups (within-group correlation 0.9), adding R increased group coverage by 0.145833 and F1 by 0.019557: six datasets improved and two tied. Noise count was unchanged. Feature-level stability remained mixed (Nogueira: three increases, two ties, three decreases). Coverage measures representatives of the two groups, not recovery of all four signal variables. [E8]

Sources: [E7-E10] C_diagnostics; [E7] S_primary_111_minus_101_k2; [E8] independently reconciled R_primary_111_minus_110_k2. Structural coverage endpoints are constrained and are not independent confirmations.

## 7. XOR, reproducibility and remaining decisions

### XOR exposed dependence on the learner

The outcome was the XOR of the signs of X1 and X2, with 5% independent label flips and four noise predictors. RF and LR shared the same eight datasets and all 48 nested partitions. RF used 64 trees, depth 4 and minimum leaf size 5; LR contained no interaction terms and was not retuned. [E9-E10]

| ASSS 111, k=2 | Joint recovery | Mean F1 | ROC AUC | Balanced acc. |
| --- | --- | --- | --- | --- |
| RF pipeline | 1.000000 | 0.943483 | 0.948825 | 0.943191 |
| LR pipeline | 0.083333 | 0.494482 | 0.508074 | 0.514212 |

The RF/LR difference is descriptive for one generator and two frozen configurations. Four A4 layers do not represent 32 independent datasets: the XOR layers reuse the same eight datasets.

All CSR variants had identical RF outcomes at k=2. Strong absolute RF performance therefore does not establish an individual benefit from C, S or R. The result shows that detecting interacting signals depends on what the learner can represent. It does not establish general RF superiority.

### Reproducibility checks and preserved failures

The deliveries retained frozen seeds, training-only selection, independent reconstruction, direct refits, cold replay and zero-operation resume checks. Each synthetic layer included 336 prescribed direct refits and one complete selector replay. Native PowerSHAP included a full fold-0 replay and five independent LR refits. These checks support computational reproducibility, not independent scientific confirmation.

Two reporting issues remain visible in the evidence history. The correlated-group frozen summary miscounted two secondary stability directions because of floating-point residuals; a separate exact reconciliation supplied the report. The earlier PowerSHAP training-gate audit omitted native float32 semantics; the failure was retained, and the formal auditor was prospectively revised and then passed directly. Neither failure was overwritten as an original pass.

### What remains before a defensible thesis conclusion

| Priority | Proposed next work | Decision sought |
| --- | --- | --- |
| 1 | Integrate A3/A4 with the full-framework and ablation evidence; reconcile terminology and claim boundaries | Agree on a narrower, conditional contribution |
| 2 | Explain S failures and separate diagnostic accuracy, feature stability and prediction | Decide whether analysis is sufficient or a prospective revision is needed |
| 3 | Consider deferred WDBC folds or further external validation under a separate plan | Choose the most informative next experiment |

The traditional and native A3 endpoints and all four A4 layers are delivered. Whole-stage A3/A4 synthesis, closure and thesis completion remain pending. STOP markers remain set; this report does not initiate new experiments.

Sources: [E6], [E8-E11], including cross_layer_pairing and current state/resume records.

## Evidence register and reading notes

This is a progress report based on the delivered experiment records, not a new literature review. Values were checked against current aggregates and dated reports. The identifiers below map each claim to its owning record; full-precision extracts and source hashes accompany the public update.

| ID | Owning record / source | Used for |
| --- | --- | --- |
| E1 | Research_Progress_20260930_v3.pptx; ../build/content.json and bilingual script | Previous reporting boundary |
| E2 | HTRU2_Results_Discussion_20261012.md, Tables 1-4 | Full-framework F1, compression, stability and contamination |
| E3 | HTRU2_Ablation_Report_20261012.md; HTRU2_Methods_Evidence_Consistency_20261013.md; qa/ablation_20261011/full_snapshot | Fixed-budget ablation and definitions |
| E4 | A2_Development_Progress_20261017.md; a2_state.json; RESUME_A2.md | WDBC and Adult status |
| E5 | thesis_method_validation_20261018/qa/a3_20261024/report_inputs.json | Traditional five-fold metrics and paired differences |
| E6 | qa/a3_3_formal_20261029/aggregate.json; A3_3_Formal_Comparison_20261029.md | Native five-fold comparison |
| E7 | qa/a4_20261020_v2/report_inputs.json and its bound summary | Signal/noise LR and C diagnostics |
| E8 | qa/a4_20261025/report_only_summary.json | Correlated-group LR; exact direction reconciliation |
| E9 | qa/a4_20261026/report_inputs.json and its bound summary | XOR RF |
| E10 | qa/a4_20261027/report_inputs.json and its bound summary; cross_layer_pairing.json | XOR LR and paired design |
| E11 | CURRENT_EXPERIMENT_PLAN.json; current A3/A4 state and resume files | Completion boundaries and pending work |

Unless otherwise stated, qa paths E6-E10 are relative to thesis_method_validation_20261018. E2-E3 belong to thesis_htru2_20261001; E4 belongs to thesis_crossdataset_20261014. Original records and frozen failures remain unchanged.

### How to read the metrics

**F1** balances precision and recall at the frozen threshold. **ROC AUC** describes ranking across thresholds; **AP** is average precision. They measure different aspects of prediction and are not interchangeable.

**Nogueira** and **Jaccard** summarize feature-set stability across resamples. A stable feature set can still omit useful signals. **MSE** in Section 6 measures error in a specific variance diagnostic, not prediction error.

Differences are method minus comparator unless explicitly labeled as loss. A difference of 0.00701 F1 is 0.701 percentage points on the 0-100 scale; it is not a 0.701% relative improvement. No unreported confidence intervals or significance claims have been added.

### Public research materials

[github.com/luolaoban338-sketch/asss-htru2-thesis](https://github.com/luolaoban338-sketch/asss-htru2-thesis)

The dated update contains research sources, result extracts and these reports. Large run artifacts remain in the local evidence archive.
