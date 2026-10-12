# HTRU2 data provenance

Source: Lyon, R. (2015). HTRU2 [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5DK6R

Dataset page: https://archive.ics.uci.edu/dataset/372/htru2

License: Creative Commons Attribution 4.0 International, https://creativecommons.org/licenses/by/4.0/ . The data license does not license the thesis source code.

`raw/` preserves the official ZIP, CSV, ARFF and Readme retrieved on 1 October 2026, with hashes in `raw/manifest.json`. These are public astronomical candidate features, not the previously used judicial dataset.

`audit/htru2_table.csv` is the derived table used in the experiments. It adds stable row IDs and predictor-vector hash groups, names the eight predictors in original column order and places the unchanged binary label in `target`. Numerical values are retained, with CSV serialization precision recorded by the checksum. `audit/audit_report.json` and `audit/feature_dictionary.csv` document the schema and descriptive checks.

There are 17,898 rows and 1,639 positives. The Readme's skewness/kurtosis description order differs from the ARFF names; original column positions and ARFF names are retained. Predictor-vector hash groups are not astronomical source identities. Source-level independence cannot be established from this table.

Data and processed-file integrity are checked by `python scripts/verify_share.py`. Raw/processed numerical equality is also checked by the test suite.
