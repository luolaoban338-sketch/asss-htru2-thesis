> Public research copy prepared on 2026-10-30. This is a dated report, not a current stage-status statement. Operational account-use material was omitted where present; scientific results and audit limitations are retained. See the publication manifest for the original source hash and removed source-line ranges.

# A2 development comparison progress

Progress report  17 October 2026  Europe Moscow

The WDBC LR outer development fold 2 package is complete and independently audited. Fold 2 exceeds the pre specified 0.01 outer F1 loss tolerance for 6 arms: feature only, joint, sequential, joint random, joint ordinary, joint robust. In fold 2, joint, sequential and joint robust are not identical across all recorded F1, AUC and retention metrics. WDBC coverage is 3 of 5 planned outer folds and remains incomplete development evidence. The previously completed fold 0 cold replay is retained; no additional cold replay is claimed today. The classifier attempt interrupted on 16 October remains counted as failed with unknown completion; its earlier recovery evidence is retained. Adult main remains deferred under the recorded runtime gate status failed runtime feasibility. [S1, A1]

The audited WDBC coverage is 3 of 5 planned outer development folds. Completed IDs are 0, 1, 2; newly completed today are 2; partial IDs are None. Adult main remains subject to its separate runtime gate. [A1, A0, P1]


## Newly completed outer fold results

Table 1 WDBC outer fold 2

| Arm | F1 | AUC | k | Cells | Violation |
| --- | --- | --- | --- | --- | --- |
| full | 0.9880 | 1.0000 | 30 | 1.0000 | No |
| sample only | 0.9880 | 1.0000 | 30 | 0.5011 | No |
| feature only | 0.9524 | 0.9967 | 7 | 0.2333 | Yes |
| joint | 0.9756 | 0.9977 | 10 | 0.1670 | Yes |
| sequential | 0.9762 | 1.0000 | 25 | 0.4176 | Yes |
| joint random | 0.9756 | 0.9977 | 10 | 0.1670 | Yes |
| joint ordinary | 0.9425 | 0.9977 | 10 | 0.1670 | Yes |
| joint robust | 0.9524 | 0.9967 | 7 | 0.2333 | Yes |

Cells is the actual retained training row fraction times the retained original variable fraction. Violation marks outer F1 loss above 0.01 plus the 1e-12 numerical tolerance, using the matched full arm before rounding. A lower cell fraction does not establish lower runtime. [A1, P1]

## Frozen scientific settings

Logistic regression uses B=4, M=64, 3 utility folds and 3 inner folds. Selection first requires inner mean F1 at least full minus 0.01, then minimizes retained cells, with frozen ties resolved by fewer variables, higher F1 and candidate ID. Outer outcomes do not alter this rule. [P1]

<!--page-->

## Cumulative results and interpretation

Table 2 Descriptive means across completed folds

| Arm | Folds | Mean F1 | Mean AUC | Mean loss | Violations |
| --- | --- | --- | --- | --- | --- |
| full | 3 | 0.9672 | 0.9984 | 0.000000 | 0/3 |
| sample only | 3 | 0.9731 | 0.9974 | -0.005870 | 1/3 |
| feature only | 3 | 0.9581 | 0.9967 | 0.009158 | 2/3 |
| joint | 3 | 0.9560 | 0.9953 | 0.011184 | 2/3 |
| sequential | 3 | 0.9562 | 0.9961 | 0.010990 | 2/3 |
| joint random | 3 | 0.9563 | 0.9942 | 0.010911 | 2/3 |
| joint ordinary | 3 | 0.9418 | 0.9950 | 0.025440 | 2/3 |
| joint robust | 3 | 0.9575 | 0.9966 | 0.009691 | 2/3 |

Mean loss is the arithmetic mean of matched full F1 minus arm F1. The denominator in Violations is the number of observed folds. These incomplete development summaries do not establish significance or noninferiority. [A1]

Fold 0 has 7 outer tolerance violations: sample only, feature only, joint, sequential, joint random, joint ordinary, joint robust. [A1]

In fold 0, joint, sequential and joint robust have zero observed differences in F1, AUC, retained rows, variable count and retained cells. Equality of these recorded outcomes does not establish equivalence of the methods. [A1]

Fold 1 has 0 outer tolerance violations: none. [A1]

Fold 2 has 6 outer tolerance violations: feature only, joint, sequential, joint random, joint ordinary, joint robust. [A1]

## Scope of the evidence

3 of 5 WDBC development folds are complete. Shared searches and caches prevent independent per arm timing comparisons. M64 is a prospective cost control setting; its approximation adequacy for 12 or 30 variables remains unvalidated. Exact enumeration above eight variables is skipped under the frozen guard. [A1, P1]

The results do not establish robustness, superiority, noninferiority, scalability, interaction recovery or deployment suitability. Traditional feature selection baselines, synthetic studies, isolated complete process timing and the full English thesis remain separate work. [P1]

<!--page-->

## Resource accounting and Adult decision

Table 3 Cumulative WDBC operations in the shared ledger

| Operation | Status | Attempts | Seconds |
| --- | --- | --- | --- |
| encoder | success | 1,369,953 | 1,979.93 |
| kmeans | success | 832 | 9.48 |
| logistic | success | 1,369,470 | 2,194.98 |
| logistic | failed | 1 | Unknown |

Counts include recorded main runs, the previous fold 0 cold replay and direct audit refits. Attempts are recorded before fitting; failed or unfinished attempts remain visible. Operation seconds are call durations, not isolated end to end method timings. The frozen model and wall time limits remain unchanged. [A1, P1]

The worker interruption on 16 October preserved 33 complete fold 1 games. One classifier attempt has unknown completion and remains counted as failed; its duration is unknown. Recovery changed only that unfinished status and the interrupted session, retained all attempt counters, and passed independent review before resuming missing work. The session consumes 1930.43 seconds as a conservative elapsed time bound, not a measured invocation duration. [R1, R2]


## Adult remains deferred

Recorded Adult runtime gate status: failed runtime feasibility. [G1]

The retained bounded probe projects 243.82 hours against the frozen 48 hour warning. Its independent audit status is passed; runtime feasibility demonstrated is no. Passing integrity checks does not satisfy a failed runtime gate. [G2]

This empirical projection includes fixed safety margins and is not a guaranteed runtime bound. Any revised Adult scope requires a separate prospective freeze. The official Adult test set remains outside this development task. [G1, G2, P1]

<!--page-->

## Verification and remaining work

Table 5 Recorded acceptance checks

| Check | Status |
| --- | --- |
| Independent main snapshot audit | passed complete folds |
| Direct classifier refits | passed all committed games and available arms |
| Retained complete fold 0 cold replay | passed |
| Retained independent replay audit | passed complete folds |
| Unchanged resume | passed |
| Previous evidence preservation | passed |
| Frozen inputs verified | 36 files |

The main audit verifies 600 direct classifier refits across the completed folds: 200 new refits today and 400 retained refits. Each is counted once in the shared ledger. The complete fold 0 cold replay is retained from 15 October; no new replay is claimed today. [A1, A0, Q3, Q4]

Document tables are compared cell by cell with saved evidence contracts, and the tolerance flags and retained cell fractions are independently recomputed. Rendering and page inspection are recorded separately. Archive creation, extraction and hash verification are governed by the separate delivery receipt; this report does not predeclare those later checks passed. [S1]

## Remaining work


2. Retain all observed outer tolerance violations and zero contrasts; do not retune the frozen protocol from outer outcomes.

3. Keep Adult main deferred under its runtime gate; any revised scope needs a separately frozen prospective protocol.

4. Aggregate final development findings only after planned coverage and acceptance checks are complete. Baselines, synthetic validation and isolated cost evidence remain separate work.

## Source index

P1  docs/protocol_en.md and configs/comparison.json with protocols/freeze_manifest.json

S1  qa/a2_20261017/daily_summary.json

A1  qa/a2_20261017/audit_wdbc_main_direct.json

A0  qa/a2_20261016/audit_wdbc_main_direct.json

B1  qa/a2_20261017/budget_revision.json

G1  qa/adult_runtime_gate.json

G2  qa/a2_20261015/adult_runtime_audit.json

Q1  qa/a2_20261017/unchanged_resume.json

Q2  qa/a2_20261017/preservation.json

Q3  qa/a2_20261015/replay_wdbc_fold_0.json

Q4  qa/a2_20261015/audit_wdbc_replay_direct.json

R1  qa/a2_20261016/interruption_recovery.json

R2  qa/a2_20261016/interruption_independent_review.json
