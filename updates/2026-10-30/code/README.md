# Source scope and portability

The files here are byte-preserving scientific source snapshots. They document the implemented methods and selected regression contracts. They are not a complete standalone reproduction environment.

- `method_validation/a3`: traditional-method selection, training-only data contracts, contribution estimation and correction.
- `method_validation/a4`: synthetic data generators, exact Shapley diagnostics and module-comparison definitions.
- `method_validation/execution_a3_3_formal_20261029`: native PowerSHAP selection and downstream LR evaluation logic. Their original orchestration dependencies are intentionally absent; do not run these files as a public quickstart.
- `crossdataset`: preprocessing, learners, sample/feature selection, fit-operation accounting, configurations and source tests for the WDBC/Adult validation framework. Actual data and frozen run fixtures are absent.

Source files can refer to original run paths, modules and frozen manifests. The public directory does not replace those historical bindings. Scientific fit-budget accounting is included; private account-usage controls, execution authorizations and operational logs are excluded. No experiment was rerun to prepare this update.

Use the repository-root quickstart for the earlier HTRU2 demonstration. Do not interpret that demonstration or passing syntax checks as validation of this entire update. The original archive remains the basis for full historical replay.
