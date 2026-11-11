> Dated public research copy, prepared 11 November 2026. The source date and scientific findings are retained. Current progress is stated in the update README: Adult confirmation timing passed on 10 November; its main experiment has not run. Local source links are shown as source references because this is a curated inspection snapshot.

# Predictive performance and signal recovery under fixed feature budgets

*English thesis manuscript · Results and Discussion · 1 November 2026*

## Results

On HTRU2, did the four features that ASSS selected in every fold preserve the F1 obtained with all eight? We compared feature selectors on the same held-out pulsar candidates. Controlled synthetic data provided a second comparison in which the signal features were known. The full Adaptive Stable Shapley Selection configuration (ASSS111) enables the variance correction, the penalty on variability in feature contributions and the redundancy penalty. The three digits record these switches in that order. [A1] [A3] [B1]

### Four-feature selection on HTRU2

On HTRU2, ASSS111 and mean Shapley selected the same four pulse-profile features in every outer fold (mean, standard deviation, skewness and kurtosis). With the same L2-regularized logistic regression downstream, the two selectors had identical F1, ROC AUC and average precision in each fold. Their five-fold means were 0.861801, 0.973623 and 0.915579, respectively. Each fold carried equal weight. F1 used a positive-class probability threshold of 0.5. [A1]

Two other selectors retained four features and achieved higher F1 than ASSS111 in every fold. Ranking features by L1 coefficient magnitude gave a mean F1 of 0.872498, and forward selection by cross-validation gave 0.867402. Random-forest impurity ranking also had a higher mean (0.863574), although its advantage occurred in three folds. ASSS111 exceeded random four-feature selection by 0.002699 on average, but its F1 was higher in only two folds. A difference of +0.030196 in the final fold outweighed the losses elsewhere. [A2]

Retaining all eight features gave a mean F1 of 0.874328 and improved on ASSS111 in every fold. Native PowerSHAP retained 4, 6, 6, 6 and 4 features across the five folds. Its mean F1 was 0.868810, higher than ASSS111 by 0.007009, with higher F1 in four folds. The downstream classifier was common to both methods, but the selection rule and the number of features varied together in this comparison. [A3]

The eight methods used matched training pools, test observations and final-model seeds within each fold. The five training pools overlapped, and the development results had been examined previously, so the paired differences describe performance on this dataset. ASSS111 and mean Shapley also shared contribution estimates. Their recorded timings cannot establish how long each method would take to run independently from start to finish. [A4]

### Correcting for permutation noise improved estimates of contribution variance

The synthetic experiments separated two questions. How accurately did the method estimate variability in a feature’s contribution, and which features did it ultimately select? Each of four settings contained eight generated datasets. Within each dataset, we summarized selection over six nested resamples and then gave the eight datasets equal weight. The random forest and logistic regression comparisons on XOR shared the same eight datasets and all 48 resampling schedules. [B1]

The variance correction (C) subtracts the estimated contribution of Monte Carlo sampling noise from the observed variance across four games. We evaluated the result against the sample variance of the exact Shapley contributions in those same games. At both 16 and 64 permutations, the raw correction and its nonnegative version reduced mean squared error on every dataset in each setting. For example, at 64 permutations in the signal-and-noise setting with logistic regression, mean squared error fell from 3.212 × 10⁻⁷ to 2.021 × 10⁻⁷ with the raw correction and to 1.740 × 10⁻⁷ after clipping negative estimates to zero. This comparison measures estimation error for the four realized games. [B2]

### Selection depended on signal structure and the learner

With two features selected, penalizing variability in feature contributions (S) made the weak signal easier to miss in the signal-and-noise setting. With C and the redundancy penalty (R) enabled in both configurations, adding S raised the mean weak-signal omission rate by 0.104167. Nogueira stability fell on four datasets and tied on four, with a mean change of −0.118750. [B3]

In the correlated-group setting, the same two selection slots could cover both signal groups or go to two members of one group. With C and S enabled, adding R increased the fraction of signal groups represented by 0.145833 on average. Coverage rose on six datasets and tied on two. The number of noise features selected was unchanged on all eight, so the additional coverage came from fewer selections within an already represented group. [B4]

In the paired XOR comparison, we enabled all three modules and selected two features. The random forest recovered both interacting signals in every resample, whereas logistic regression recovered both in 0.083333 of resamples on average. Mean test F1 was 0.943483 with the random forest and 0.494482 with logistic regression. Because each learner supplied both the feature-selection utilities and the final predictions, this contrast measures the difference between the two complete workflows under this XOR design. [B5]

## Discussion

The HTRU2 results qualify the interpretation of the earlier fixed-budget ablation. In that experiment, removing individual modules left the final four-feature set and F1 unchanged, although some selection paths changed. The present comparison shows that the same four-feature budget allowed other selectors to reach sets with higher F1. In this dataset, the agreement between ASSS and mean Shapley therefore reflects a property of the two selection rules under the tested conditions. A fixed number of slots alone does not explain it. [D1]

The synthetic experiments help distinguish three steps that a predictive score alone cannot resolve. The first is estimating how much a feature’s contribution varies. Removing estimated permutation noise improved that estimate against a known, finite target. The second is deciding how much to penalize that variation. In the weak-signal setting, the penalty increased omissions and reduced stability. The third is deciding whether another member of a correlated group deserves a remaining slot. Penalizing redundancy increased the number of groups represented with the same number of noise features selected. These findings support separate judgments about estimation accuracy, signal retention and group coverage. None of the three establishes a predictive advantage for ASSS on HTRU2. [D2]

Two questions remain open. First, the XOR contrast combines the learner used to value feature subsets with the learner used to predict from them. Holding one learner fixed while varying the other would help distinguish selection effects from predictive effects. Second, the HTRU2 folds come from one previously examined dataset. A prospectively fixed comparison on another dataset, with the same training rows, feature budget and downstream model for competing selectors, would test whether the observed differences persist. [D3]

---

*Evidence notes [A1]–[D3] are indexed in the claim–evidence table (source: `thesis_method_validation_20261018/synthesis_20261030/P4_20261103_delivered_20261101/Claim_evidence.md`). Outstanding thesis requirements are recorded in the gap register (source: `thesis_method_validation_20261018/synthesis_20261030/P4_20261103_delivered_20261101/Thesis_gap_register.md`).*
