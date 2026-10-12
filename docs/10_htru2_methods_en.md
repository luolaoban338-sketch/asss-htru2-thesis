# HTRU2 development validation methods

## Draft

### Task and dataset

This study evaluates joint selection of training observations and original variables for binary classification. A procedure returns a subset of labelled training rows, a subset of the eight original variables, and a classifier fitted on the resulting matrix. Validation and test partitions remain complete. The primary compression measure is the retained training-row fraction multiplied by the retained-variable fraction. This measure describes matrix size; total computation must also include the work required to identify the retained rows and variables.

HTRU2 contains 17,898 pulsar candidates with eight numerical predictors and a binary label [1]. The positive class comprises 1,639 candidates. All rows and original variables are retained at data ingestion. Feature-vector fingerprints define duplicate groups, without using labels. No exact duplicate vectors were found. These groups do not identify astronomical sources: the public table lacks source identities and observation metadata, so source-level independence cannot be established. Predictor order and ARFF names are preserved because the accompanying Readme reverses the textual order of skewness and kurtosis. The conflict restricts physical interpretation of individual columns.

### Preprocessing and sample selection

Four conditions use all training rows, a class-stratified random half, an ordinary geometry-based half, or a robust geometry-based half. The retained count is the ceiling of the requested fraction times the current training size. Class quotas use the Hamilton largest-remainder rule. Geometry-based selectors cluster each class with at most ten clusters and allocate retained rows across clusters. Within each cluster, deterministic distance ranks select existing observations. The ordinary geometry uses mean and standard-deviation scaling. The robust geometry uses median and interquartile-range scaling with clipping at the first and ninety-ninth percentiles. Geometry preprocessing uses only the current training partition. It does not transform held-out data for selection.

The classifier is L2-regularized logistic regression with C=1, an intercept, the lbfgs solver, at most 5,000 iterations, tolerance 10^-6 and no class weights. A predicted probability of at least 0.5 defines a positive prediction. Classifier imputation and standardization are fitted only on the selected rows and current variables. The same settings are retained across all comparisons. The implementation rejects convergence warnings rather than treating affected fits as valid results.

### Sample-conditioned feature contributions

Each feature-subset utility is mean positive-class F1 across three utility-validation folds. In each fold, the selected training rows, validation rows and model seed remain fixed across all feature subsets. A new classifier is fitted for each nonempty subset. The empty subset uses a constant probability equal to the positive-class proportion in its selected training rows. Preprocessing is refitted on the relevant selected rows and columns for every nonempty subset.

For each selector, B=4 class-stratified subsamples retain 80% of its training partition. Independent uniform feature permutations estimate marginal contributions within each subsample. The standalone implementation pilot uses M=16 permutations; the joint and stability workflows use the inherited development budget M=256. All 256 subsets provide exact Shapley references for diagnostics. Exact references do not replace permutation estimates in budget selection. A separately recorded sweep of M=16, 64, 256 and 1,024 reuses the saved utility tables and does not refit classifiers or revise selection budgets.

The estimated contribution variance across subsamples is reduced by the mean estimated Monte Carlo variance, with negative differences truncated to zero. A feature score equals its mean estimated contribution minus lambda times the square root of this corrected variance. Lambda is either 0 or 0.5. The correction is a development estimator; truncation does not establish population stability or a confidence bound.

### Feature paths and joint budgets

Redundancy is measured from selected training rows using absolute Spearman association between original variables. The path objective sums feature scores and subtracts eta/8 times the sum of pairwise redundancy over included unordered pairs. Eta is either 0 or 0.1. Each greedy step adds the variable with the largest objective increment, resolving ties by original variable order. All path prefixes from one to eight variables are retained, including steps with negative increments.

Five fixed outer folds assess predictive performance. Within each outer training partition, three inner folds rebuild the sample selection, contribution estimates and feature paths from their own training rows. Four sampling conditions, two lambda values, two eta values and eight feature counts define 128 candidates per inner partition. Candidate subsets shared within a partition and method reuse fitted models. A candidate is feasible if its equal-fold mean inner F1 is at least the paired full-data mean minus 0.01. Among feasible candidates, selection minimizes mean retained matrix fraction, then feature count, then maximizes F1, followed by fixed candidate order. The full-row, full-variable candidate provides a feasible fallback.

Inner decisions are saved before outer evaluation. The chosen settings then rebuild selection on the full outer training partition, fit the final classifier and predict every outer test row. Comparisons include no selection, row selection only, feature selection only, joint selection and sequential selection, plus restricted joint candidate sets. The sequential rule first chooses the row-selection condition using all variables, then chooses a feature budget within that condition under the same original F1 threshold. All rules use a shared candidate table. Shared computation cannot establish separate per-rule runtime advantages. Inner feasibility does not guarantee that the outer F1 loss remains within 0.01.

### Training contamination and complete-selection stability

The sample-selection sensitivity experiment perturbs 0%, 5% or 10% of training rows. Each selected row receives a random signed offset of ten clean-training interquartile ranges in one eligible numerical column. Three fixed draws are used at each nonzero rate, with the 5% row set nested within the 10% set for each draw. Labels and test data remain unchanged. The resulting 140 fits test the specified numerical stress; they do not model astronomical instrument noise and do not include joint feature selection.

Stability uses ten complete selection workflows on 80% class-stratified resamples of the first outer fold's training pool. Each workflow reruns inner budget selection and final feature construction. The outer test pool is physically excluded from the stability engine. This stage produces selected sets without fitting a final classifier. Nogueira stability [2], pairwise Jaccard similarity, feature frequencies, set sizes and chosen budgets are reported. All-full and all-empty cases have undefined Nogueira values. The overlapping resamples and 45 pairwise similarities do not constitute independent experimental datasets. Data resampling and algorithmic seeds vary together, so their individual effects are not separated.

### Metrics and reproducibility

Positive-class F1 is the primary metric. Secondary measures include precision, recall, ROC-AUC, average precision, confusion counts and actual compression. Average precision uses a stepwise precision-recall summary and is not labelled as trapezoidal PR-AUC. Fold means and pooled out-of-fold scores are distinguished. Fold standard deviations are descriptive because training partitions overlap. No significance, noninferiority or population-stability claims are made from these development runs.

Separate immutable run identifiers preserve raw and processed data checksums, code and configuration snapshots, environment versions, split identities, selected rows and variables, model coefficients, predictions and failure states. Verification reconstructs saved predictions, checks training-only boundaries, independently evaluates contributions and stability, replays prespecified partitions and checks unchanged resume hashes. Each numerical process uses one thread. Up to four independent folds or replicates may run concurrently; their elapsed times overlap and are reported accordingly. The study reports selection overhead as well as final model fitting, without equating matrix compression with end-to-end speedup.

## References

1. Lyon, R. HTRU2. UCI Machine Learning Repository (2015). https://doi.org/10.24432/C5DK6R . Dataset documentation: https://archive.ics.uci.edu/dataset/372/htru2 . The source study is Lyon, R. J., Stappers, B. W., Cooper, S., Brooke, J. M. and Knowles, J. D. Fifty years of pulsar candidate selection: from simple filters to a new principled real-time classification approach. *Monthly Notices of the Royal Astronomical Society* **459**(1), 1104–1123 (2016). https://doi.org/10.1093/mnras/stw656 .
2. Nogueira, S., Sechidis, K. and Brown, G. On the Stability of Feature Selection Algorithms. *Journal of Machine Learning Research* **18**(174), 1–54 (2018). https://jmlr.org/papers/v18/17-514.html .

## Section outline and author review notes

- Task and data scope define what compression means and identify schema and grouping limitations.
- Preprocessing and utility specify the training boundaries needed to reproduce selection.
- Feature paths and nested budgets define the candidate search and fallback.
- Contamination and stability describe distinct development experiments.
- Metrics and records define verification and interpretation limits.

Terminology is fixed as follows: HTRU2 is the dataset; positive-class F1 is the primary utility; average precision is abbreviated AP; B denotes contribution subsamples; M denotes sampled permutations; rho denotes row fraction; k denotes retained variable count; epsilon is the absolute F1 tolerance. The project's proposed Adaptive Stable Shapley Selection (ASSS) module remains a development method, with no novelty or superiority asserted here.

Claim-evidence map: procedural claims map to configs, src, experiments and immutable run snapshots; dataset statements map to the official source and local source audit; completed verification is established only by the final delivery manifest. This methods draft contains no claims about measured performance. Remaining author inputs are final thesis formatting and approval of the wording; the source-level independence and column-description conflict remain scientific limitations. Specific paragraphs can be revised without changing the frozen experiments.
