# Method map

| Research component | Implementation | Main checks |
| --- | --- | --- |
| Training partitions and disjointness | `src/contracts.py` | `tests/test_contracts.py` |
| Data identity and feature schema | `src/data.py` | `tests/test_htru2_migration.py` |
| Training-fitted preprocessing | `src/preprocessing.py` | `tests/test_baseline.py` |
| Random / ordinary / robust row selection | `src/samplers.py` | `tests/test_baseline.py` |
| Nested training-only feature utility | `experiments/feature_utility.py` | `tests/test_feature_utility.py` |
| Exact / permutation Shapley and variance correction | `experiments/shapley.py` | `tests/test_feature_utility.py`, `tests/test_ablation.py` |
| Redundancy penalty and feature path | `experiments/feature_path.py` | `tests/test_feature_path.py` |
| Joint sample–feature selection | `experiments/joint_selection.py`, `scripts/run_joint_pilot.py` | `tests/test_joint_selection.py` |
| Complete-selection stability | `experiments/stability.py`, `scripts/run_stability_pilot.py` | `tests/test_stability.py` |
| Training contamination | `experiments/contamination.py`, `scripts/run_joint_contamination.py` | `tests/test_contamination.py`, `tests/test_joint_contamination.py` |
| Fixed-budget module ablation | `experiments/ablation.py`, `scripts/run_ablation.py` | `tests/test_ablation.py` |

The frozen protocols describe why these components are used and how the experiments are bounded. `docs/10_htru2_methods_en.md` describes the earlier development methods; `docs/18_ablation_protocol_v1.md` describes the later fixed-budget study. Historical protocol wording reflects its freeze date and should not be interpreted as the latest completion status.
