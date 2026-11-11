# 11 November 2026 research update

This update covers progress after the 30 October report, using accepted project evidence through 10 November. It adds scientific source snapshots, the latest English stage manuscript, numerical summaries and an unexecuted comparison proposal. Advisor-specific reporting files remain local.

## What changed

| Workstream | Status at 30 October | New accepted evidence | Current boundary |
| --- | --- | --- | --- |
| HTRU2 A3 and synthetic A4 | Method and layer packages delivered; synthesis pending | Synthesis, claim mapping and scoped closure completed 1 November | HTRU2 development comparisons and four synthetic diagnostic settings only |
| WDBC | Outer folds 0-2 complete | Folds 3 and 4 completed; all five folds independently audited by 5 November | Descriptive cross-dataset evidence; outer loss violations remain |
| Manuscript | Separate chapters | Integrated English stage manuscript; REV01-REV05 applied on 7 November | Author adoption, advisor decisions and thesis completion remain pending |
| Adult | Earlier runtime feasibility failed | Reduced design, versioned resource implementation, 75-task confirmation and independent audit; new gate passed on 10 November | No Adult main, replay, direct-audit fit package or outer/official test prediction |
| R09 | Remaining comparison gap | HTRU2 shared-half-sample, fixed-k proposal and historical-rule review | Proposed comparison has not run; historical threshold and executable remain unresolved |

## Read the evidence

| Material | Link |
| --- | --- |
| Latest English Methods, Results and Discussion stage manuscript | [7 November revision](manuscript/English_Thesis_Revision_20261107.md) |
| A3/A4 synthesis manuscript and claim-to-source map | [Results and Discussion](manuscript/A3_A4_Results_Discussion_20261101.md), [claim evidence](manuscript/A3_A4_Claim_Evidence_20261101.md) |
| WDBC full five-fold report and exact-precision results | [Report](reports/WDBC_Fivefold_20261105.md), [JSON](results/wdbc_five_fold.json), [CSV summary](results/wdbc_five_fold_summary.csv) |
| A3 40 method-fold records and A4 four-layer synthesis | [A3](results/a3_htru2_synthesis.json), [A4](results/a4_four_layer_synthesis.json) |
| Adult exact contract and scientific configuration | [Resource revision](protocols/Adult_Exact_Resource_Revision_20261110.md), [contract](protocols/adult_exact_resource_contract.json), [config](protocols/adult_exact_scientific_config.json) |
| Adult 10 November confirmation gate and independent audit | [Gate](results/adult_confirmation_runtime_gate.json), [status](results/adult_confirmation_status.json), [audit](results/adult_confirmation_independent_audit.json) |
| Adult source and no-fit tests | [Code scope](code/README.md), [source files](code/adult_exact/) |
| Unexecuted R09 comparison and historical provenance limits | [Proposal](protocols/R09_Proposed_Comparison_20261103.md), [historical-rule review](protocols/Historical_Shapley_Rule_Review_20261103.md) |
| Original source hashes and public-copy transformations | [Publication manifest](PUBLICATION_MANIFEST.json) |

## Results and interpretation

Across five WDBC development folds, joint selection has mean F1 **0.958229** versus **0.964939** for full data, while retaining **14.6908%** of training cells. Its paired mean F1 difference is **-0.006710**, with losses above 0.01 in folds **0 and 2**. Joint-robust additionally violates the line in fold 4. Retention is calculated within each fold and then averaged; it is not the product of separate mean row and feature fractions. These findings support an observed compression/performance tradeoff, not outer noninferiority or universal robustness. Some restricted joint arms use all rows, so their names do not establish matched sampling budgets.

The A3/A4 synthesis does not establish overall ASSS superiority. ASSS111 matches mean Shapley on HTRU2 and is below L1 ranking and forward selection in all five folds. The four synthetic layers support specific variance-estimation and redundancy findings, with mixed or adverse variability-penalty results. Each synthetic layer has eight generated datasets and 48 nested selectors; the XOR RF/LR analyses reuse the same datasets and schedules. Nested selectors are not extra independent replications.

For Adult, the 8 November reduced-design timing gate failed at **79.136235 hours**. The 9 November **44.171263-hour** value was only a historical-rate sensitivity calculation. The isolated exact-resource v3 contract then retained B=4, M=64, 8,000-row development pools and five folds while reconstructing **301,350** main-plus-replay utility cycles and preserving typed retry reserves. The new 10 November confirmation completed **75/75** training-only tasks; its independent audit passed **165,440 checks**. The new gate projects **41.457911 hours**, below the frozen 48-hour threshold. This is an engineering allowance, not observed main-experiment duration or a statistical/absolute guarantee. The original failed gate and earlier failed/interrupted records are retained. Main work remains unexecuted and requires a separately bounded execution decision.

## Dates, completeness and provenance

Dated manuscripts and frozen contracts retain the status that was true when they were written. In particular, the 7 November manuscript and the 10 November no-fit resource document precede the new confirmation gate. The current overview and `adult_confirmation_*` results provide the later state. The inherited theoretical counts inside the Adult config are retained as provenance; the exact contract and `projection.py` govern the v3 resource estimate.

`byte_preserving_copy` entries match source bytes. `derived` entries document omitted account-use metadata, mapped local paths, selected scientific fields and converted local source links. Their public hashes must not be presented as hashes of the original frozen files. All source files were read without fitting, prediction, ledger updates or changes to STOP markers. This publication does not provide full raw run artifacts, model files, source datasets, personal usage logs, advisor reports or archive ZIPs. It adds no code license and retains the repository's existing dataset attribution.

Adult main, R07 sufficiency, R08, complete R09, A2 overall and the thesis remain open. No automatic continuation follows this update.

## 中文概览

本次新增的核心进展是：A3/A4 在限定范围内完成综合与收口；WDBC 从三折推进到五折；英文阶段稿完成五项修订；Adult 新版确认计时于 11 月 10 日通过 48 小时工程门槛，投影约 41.46 小时。Adult 主实验尚未运行，R08/R09 等缺口和论文整体仍未完成。导师汇报专用 PDF 与解读材料保留在本地，不纳入公开目录。
