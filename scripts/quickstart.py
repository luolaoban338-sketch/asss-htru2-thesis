"""A deterministic illustration of the modules, not an empirical thesis result."""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from src.data import load_dataset
from experiments.shapley import exact_shapley, permutation_shapley, corrected_scores
from experiments.feature_path import forward_path


def main():
    data = load_dataset("htru2")
    features = ("toy_a", "toy_b", "toy_c", "toy_d")
    weights = np.array([0.4, 0.3, 0.2, 0.1])
    lookup = dict(zip(features, weights))
    def utility(subset):
        return sum(lookup[name] for name in subset)
    exact = exact_shapley(utility, features)
    estimated = permutation_shapley(utility, features, M=64, seed=42)
    np.testing.assert_allclose(exact, weights, atol=1e-12, rtol=0)
    np.testing.assert_allclose(estimated["phi"], exact, atol=1e-12, rtol=0)
    # Four prescribed synthetic contribution estimates illustrate variability correction.
    offsets = np.array([-0.03, -0.01, 0.01, 0.03])[:, None]
    estimates = weights[None, :] + offsets * np.array([1.0, 2.0, 1.0, 0.5])
    scores = corrected_scores(estimates, np.full((4, 4), 0.0001), penalty=0.5)
    redundancy = np.zeros((4, 4))
    redundancy[0, 1] = redundancy[1, 0] = 0.8
    path = forward_path(features, scores["score"], redundancy, eta=0.1)
    print(json.dumps({"purpose": "Synthetic module demonstration; not a thesis result",
                      "htru2_rows": len(data.y), "htru2_features": len(data.X.columns),
                      "htru2_positive": int(data.y.sum()), "toy_exact_shapley": exact.tolist(),
                      "toy_corrected_scores": scores["score"].tolist(),
                      "toy_two_feature_subset": path[1]["features"], "model_fits": 0}, indent=2))


if __name__ == "__main__":
    main()
