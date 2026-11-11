> Dated public research copy, prepared 11 November 2026. The source date and scientific findings are retained. Current progress is stated in the update README: Adult confirmation timing passed on 10 November; its main experiment has not run. Local source links are shown as source references because this is a curated inspection snapshot.

# Advisor evidence and exhibit index

6 November 2026 | T5 review materials | Current status overlay

The accepted HTRU2 and synthetic evidence is now accompanied by all five WDBC development folds. This index connects the review text to its source records. Source keys below resolve to checked local files. Earlier packages retain their dated coverage statements: T1 and P4 describe WDBC at three folds, and the T4 preparation review describes four. The current five-fold status comes from A2 and WREPORT; these historical snapshots must not be read as current status.

## Research questions

| Review question | Evidence available for discussion | Source and interpretation boundary |
|---|---|---|
| RQ1 What predictive loss accompanies a smaller training matrix | Five HTRU2 and five WDBC outer folds, with full, sample-only, feature-only and joint rules and their actual retained rows and columns. | HOUTER, HTABLE, WNUM and WREPORT. Average within-fold cell fractions; do not substitute the product of separate means. Preserve every tolerance violation. Adult main results are absent. |
| RQ2 What changes when training rows change | HTRU2 sampling-only performance, class coverage, sample-conditioned contribution summaries and complete-workflow resampling stability; WDBC restricted joint policies. | HSAMPLING, HCONTRIB, HSTABILITY and WREPORT. The WDBC restricted policies use full rows in 3 of 15 arm-fold records; their labels do not establish matched row budgets or close R08. |
| RQ3 How do joint and sequential decisions differ | Five paired HTRU2 policy comparisons and five paired WDBC policy comparisons under each frozen shared candidate scheme. | HCOUPLE, WNUM, WREPORT and WPROTOCOL. Report actual candidate reuse and realized compression. Shared execution totals do not establish standalone method timings. |
| RQ4 Where do selection rules fail | HTRU2 training contamination and fixed-budget ablations; signal-plus-noise LR, correlated-group LR and paired XOR RF/LR diagnostics. | HSTRESS, HABLATION, A4 and A4DETAIL. Eight independent generated datasets per setting; resamples are nested and XOR learner roles change together. |

## Twelve accepted P4 claims

Placement labels identify logical sections in the review manuscript, not final thesis page numbers. The original claim register and T1 index preserve earlier paragraph-level placements.

| Claim | Suggested review section | Supported statement | Source fields | Boundary to retain |
|---|---|---|---|---|
| A1 | Feature baselines on HTRU2 | ASSS111 and mean Shapley selected the same four profile variables and matched F1, AUC and AP in all five folds. | A3: rows, summary; P4CLAIMS | Single development dataset and common logistic predictor; no universal equivalence. |
| A2 | Feature baselines on HTRU2 | At four variables, L1 ranking and forward selection exceeded ASSS111 F1 in each fold; RF impurity and random-four comparisons included reversals. | A3: paired, paired_summary | Fixed-column comparison; fold means do not erase reversals. |
| A3 | Feature baselines on HTRU2 | The all-feature reference and native PowerSHAP had higher mean F1 than ASSS111; native PowerSHAP retained 4, 6, 6, 6 and 4 variables. | A3: summary, rows.actual_k | Native selector and cardinality vary together; not an equal-column-budget comparison. |
| A4 | Feature baselines on HTRU2 | Methods share training pools, held-out rows and final-model seeds within each fold; ASSS111 and mean Shapley share upstream contribution work. | A3: identity and downstream; A3COST | Folds overlap and some costs are shared; no isolated runtime advantage. |
| B1 | Synthetic module diagnostics | Each setting averages eight generated datasets with six nested resamples per dataset; the XOR RF/LR workflows share datasets and schedules. | A4: independent_unit, xor_pairing | Generated dataset is the independent unit; 48 resamples are not 48 independent datasets. |
| B2 | Synthetic module diagnostics | Raw and zero-clipped subtraction of estimated permutation noise reduced variance-estimation MSE at M=16 and M=64 in all four settings. | A4: layers.<setting>.C_diagnostics | Target is exact Shapley sample variance across the same four games; not population calibration or predictive benefit. |
| B3 | Synthetic module diagnostics | At k=2 in signal-plus-noise LR, S increased mean weak-signal omission; Nogueira stability fell on four datasets and tied on four. | A4: layers.signal_noise_LR.S_111_minus_101_k2 | Paired 111 versus 101 with C and R fixed; adverse result must remain visible. |
| B4 | Synthetic module diagnostics | At k=2 in the correlated-group design, R increased signal-group coverage on six datasets and tied on two; noise counts were unchanged. | A4: layers.correlated_groups_LR.R_111_minus_110_k2 | Paired 111 versus 110; group recovery does not identify a unique correct variable. Retain original audit failure and reconciliation. |
| B5 | Synthetic module diagnostics | The all-enabled XOR RF workflow recovered both signals more often and had higher mean F1 than the paired LR workflow. | A4: xor_RF_minus_LR_k2.111; layer arms | Both utility and prediction learners change; neither role is isolated. |
| D1 | Discussion of fixed-budget agreement | The four-slot limit alone cannot explain ASSS111/mean-Shapley agreement because equal-size alternatives achieved different F1. | A3 A1-A2; HABLATION | Interpretation, not identification of a unique cause of agreement. |
| D2 | Discussion of evidence boundaries | C, S and R have distinct observed diagnostic effects; these cannot be pooled or transferred into a predictive-gain claim on HTRU2. | A4 B2-B4; A3; HABLATION | Preserve different targets, settings and adverse outcomes. |
| D3 | Discussion of unresolved comparisons | Separate utility/prediction learner roles on XOR and repeat matched-budget selector comparisons on another real dataset. | A4; A3; R09 | Prospective questions. WDBC adaptive-policy completion does not execute the R09 selector comparison. |

## Exhibits and figure readiness

The identifiers below are editorial catalogue identifiers, not assertions that every exhibit has been inserted into the new Word file. Local source table numbers remain local to their original report.

| ID | Object and status | Content and source | Remaining work |
|---|---|---|---|
| E01 | HTRU2 four-cell table available | HTABLE and HOUTER: 20 arm-by-fold records, F1, AUC, actual rows, k, cells and violations. | Use the same fold denominators and label its development scope. |
| E02 | WDBC eight-arm means available | WREPORT Table 1 and WNUM: all five folds; four core policies plus sequential and three restricted policies. | Keep mean retained cells calculated from per-fold products. |
| E03 | WDBC paired differences available | WREPORT Tables 3-4: seven compressed-vs-full contrasts and joint-vs-sequential differences. | Preserve signs and unrounded tolerance flags. |
| E04 | WDBC full fold appendix available | WREPORT Tables 5-6: 40 arm-by-fold records. | Retain duplicated model outcomes as separate decision arms. |
| E05 | HTRU2 method comparison available | A3: eight methods across five folds, selected identities and paired contrasts. | Separate fixed k=4 from native PowerSHAP cardinality. |
| E06 | Synthetic diagnostics available | A4 and A4DETAIL: dataset-level C/S/R contrasts and paired XOR workflows. | Keep each target and independent-unit denominator explicit. |
| E07 | Literature comparison available | LIT, LITMATRIX and LITREG: nine references, eight full method sections and one partial primary-text precedent. | A targeted review; complete SI(FS)² implementation was not audited. |
| F01 | Framework and validation diagram planned | HMETHOD, HCOMPANION and WPROTOCOL describe the train/utility/inner/outer boundaries. | Author a diagram and pseudocode; an index alone does not complete R19. |
| F02 | Cross-dataset four-cell figure planned | HOUTER and WNUM cover two datasets. | Adult is missing; never draw a three-dataset result with a fabricated Adult panel. |
| F03 | Compression frontier planned | HTRU2 candidate evidence is indexed in T1INDEX; WDBC frozen candidates are governed by WPROTOCOL. | Define aggregation and show held-out policy results separately from inner candidate scores. |
| F04 | Signal and group recovery figure planned | A4 and A4DETAIL. | Label datasets as independent units and resamples as nested. |
| F05 | Approximation and cost figure planned | HCHAPTER, A4 and A3COST. | Separate fixed-game approximation error, actual attempt counts and unavailable isolated wall times. |

## Current literature boundary

The T4 source review is reused as a dated review, not represented as a new November 6 full-text audit. It establishes precedents for simultaneous row/column selection and retained-row times retained-feature fractions. Neither is claimed as a first invention here. T4's local a2_state hash refers to its preparation-time four-fold snapshot; the present source receipt separately hashes the current five-fold state. Original paper URLs and page/section locators remain in LITREG. No new cited algorithm was run for this package.

## Source key

- **PLAN**: 硕士论文下一阶段工作计划_20261102.md (source: `outputs/thesis_T5_20261106/outputs/researchwrite/thesis_next_work_20261102/exports/硕士论文下一阶段工作计划_20261102.md>`)
- **CURRENT**: CURRENT_PROJECT.md (source: `outputs/thesis_T5_20261106/CURRENT_PROJECT.md>`)
- **REQ**: requirements_matrix.json (source: `outputs/thesis_T5_20261106/outputs/researchwrite/thesis-gap-audit-20261013/requirements_matrix.json>`)
- **T1**: English_thesis_working_draft.md (source: `outputs/thesis_T5_20261106/outputs/thesis_T1_20261103/English_thesis_working_draft.md>`)
- **T1INDEX**: Evidence_index.md (source: `outputs/thesis_T5_20261106/outputs/thesis_T1_20261103/Evidence_index.md>`)
- **HTABLE**: HTRU2_four_cell.md (source: `outputs/thesis_T5_20261106/outputs/thesis_T1_20261103/HTRU2_four_cell.md>`)
- **HOUTER**: outer_results.csv (source: `outputs/thesis_T5_20261106/thesis_htru2_20261001/runs/joint-htru2-pilot-20261001-v1/outer_results.csv>`)
- **HMETHOD**: 10_htru2_methods_en.md (source: `outputs/thesis_T5_20261106/thesis_htru2_20261001/docs/10_htru2_methods_en.md>`)
- **HCOMPANION**: 23_methods_evidence_consistency_20261013.md (source: `outputs/thesis_T5_20261106/thesis_htru2_20261001/docs/23_methods_evidence_consistency_20261013.md>`)
- **HCHAPTER**: 21_htru2_results_integrated_en_20261012.md (source: `outputs/thesis_T5_20261106/thesis_htru2_20261001/docs/21_htru2_results_integrated_en_20261012.md>`)
- **HABLATION**: 20_ablation_report_en_20261012.md (source: `outputs/thesis_T5_20261106/thesis_htru2_20261001/docs/20_ablation_report_en_20261012.md>`)
- **HSTABILITY**: stability_summary.csv (source: `outputs/thesis_T5_20261106/thesis_htru2_20261001/runs/stability-htru2-pilot-20261001-v1/stability_summary.csv>`)
- **HSAMPLING**: fold_results.csv (source: `outputs/thesis_T5_20261106/thesis_htru2_20261001/runs/baseline-htru2-pilot-20261001-v1/fold_results.csv>`)
- **HCONTRIB**: feature_summary.csv (source: `outputs/thesis_T5_20261106/thesis_htru2_20261001/runs/feature-htru2-pilot-20261001-v1/feature_summary.csv>`)
- **HSTRESS**: joint_contamination_summary.csv (source: `outputs/thesis_T5_20261106/thesis_htru2_20261001/qa/joint_contamination_summary.csv>`)
- **HCOUPLE**: joint_sequential.csv (source: `outputs/thesis_T5_20261106/thesis_htru2_20261001/runs/joint-htru2-pilot-20261001-v1/joint_sequential.csv>`)
- **P4CLAIMS**: Claim_evidence.md (source: `outputs/thesis_T5_20261106/thesis_method_validation_20261018/synthesis_20261030/P4_20261103_delivered_20261101/Claim_evidence.md>`)
- **P4GAPS**: Thesis_gap_register.md (source: `outputs/thesis_T5_20261106/thesis_method_validation_20261018/synthesis_20261030/P4_20261103_delivered_20261101/Thesis_gap_register.md>`)
- **A3**: a3_results.json (source: `outputs/thesis_T5_20261106/thesis_method_validation_20261018/synthesis_20261030/20261031/a3_results.json>`)
- **A3COST**: costs_and_boundaries.json (source: `outputs/thesis_T5_20261106/thesis_method_validation_20261018/synthesis_20261030/20261031/costs/costs_and_boundaries.json>`)
- **A4**: four_layers.json (source: `outputs/thesis_T5_20261106/thesis_method_validation_20261018/synthesis_20261030/P2_20261101_delivered_20261031/four_layers.json>`)
- **A4DETAIL**: A4_逐数据集与方案附表.md (source: `outputs/thesis_T5_20261106/thesis_method_validation_20261018/synthesis_20261030/P2_20261101_delivered_20261031/A4_逐数据集与方案附表.md>`)
- **T2**: state.json (source: `outputs/thesis_T5_20261106/outputs/researchwrite/r09_t2_20261103/state.json>`)
- **R09**: R09_无拟合比较方案.md (source: `outputs/thesis_T5_20261106/outputs/researchwrite/r09_t2_20261103/exports/R09_无拟合比较方案.md>`)
- **HISTORY**: 历史Shapley规则核对表.md (source: `outputs/thesis_T5_20261106/outputs/researchwrite/r09_t2_20261103/exports/历史Shapley规则核对表.md>`)
- **LIT**: related_work_en.md (source: `outputs/thesis_T5_20261106/outputs/researchwrite/t4_20261105/literature/related_work_en.md>`)
- **LITMATRIX**: literature_matrix.md (source: `outputs/thesis_T5_20261106/outputs/researchwrite/t4_20261105/literature/literature_matrix.md>`)
- **LITREG**: source_registry.json (source: `outputs/thesis_T5_20261106/outputs/researchwrite/t4_20261105/literature/source_registry.json>`)
- **A1**: state.json (source: `outputs/thesis_T5_20261106/thesis_crossdataset_20261014/state.json>`)
- **A2**: a2_state.json (source: `outputs/thesis_T5_20261106/thesis_crossdataset_20261014/a2_state.json>`)
- **A2RESUME**: RESUME_A2.md (source: `outputs/thesis_T5_20261106/thesis_crossdataset_20261014/RESUME_A2.md>`)
- **WREPORT**: A2_WDBC_Fivefold_20261105.md (source: `outputs/thesis_T5_20261106/thesis_crossdataset_20261014/exports/A2_WDBC_Fivefold_20261105.md>`)
- **WNUM**: daily_summary.json (source: `outputs/thesis_T5_20261106/thesis_crossdataset_20261014/qa/a2_20261105/daily_summary.json>`)
- **WPROTOCOL**: protocol_en.md (source: `outputs/thesis_T5_20261106/thesis_crossdataset_20261014/docs/protocol_en.md>`)
- **WMANIFEST**: delivery_manifest.json (source: `outputs/thesis_T5_20261106/thesis_crossdataset_20261014/qa/a2_20261105/delivery_manifest.json>`)
