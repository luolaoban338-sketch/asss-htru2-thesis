"""Deterministic synthetic design and no-fit CSR/variance diagnostics.

Selection consumes only M=64 permutation estimates. Exact Shapley values,
generator truth, test data, and diagnostic streams are separate interfaces.
An instrumented model-fitting runner is deliberately not supplied by A3.0.
"""
from dataclasses import dataclass
import hashlib
import itertools
import json
import math

import numpy as np


PROTOCOL_ID = "synthetic-csr-20261018-v1"
FEATURES = tuple(f"X{i}" for i in range(1, 7))
SCENARIOS = ("signal_noise", "correlated_groups", "xor")
LAYERS = ("signal_noise_LR", "correlated_groups_LR", "xor_RF", "xor_LR")
DATA_SEEDS = tuple(range(2026102401, 2026102409))
CSR = ("111", "011", "101", "110", "001", "100", "010", "000")
ALIASES = {"001": "101", "000": "100"}


def canonical_json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def coordinate_seed(*coordinate):
    """31-bit sklearn-compatible SHA256 seed; collisions are fatal at freeze."""
    if not coordinate or any(type(v) not in (str, int) for v in coordinate):
        raise ValueError("Seed coordinates must be nonempty typed strings/integers")
    key = [PROTOCOL_ID, *coordinate]
    raw = canonical_json(key).encode("utf-8")
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big") % 2147483647


def layer_scenario(layer):
    if layer not in LAYERS:
        raise ValueError("Unknown scenario/learner layer")
    return layer.rsplit("_", 1)[0]


def _validate_scenario_seed(scenario, data_seed):
    if scenario not in SCENARIOS or type(data_seed) is not int:
        raise ValueError("Unknown generator or invalid dataset seed")


def seed_records():
    """Full declared schedule including intentional shared coordinates once.

    Main permutations, utility-model seeds and final-model seeds omit q so the
    six selection resamples hold algorithm randomness fixed. XOR rows and
    nested partitions omit learner and therefore agree exactly in RF and LR.
    Different b games have independent permutation substreams. Diagnostic
    streams are independent of main and include q, stream, b coordinates.
    """
    keys = set()
    for scenario, root in itertools.product(SCENARIOS, DATA_SEEDS):
        for split in ("train", "test"):
            keys.add((scenario, root, "data", split))
        keys.add((scenario, root, "random_features"))
        for q in range(6):
            keys.add((scenario, root, "selection_rows", q))
            for b in range(4):
                keys.add((scenario, root, "contribution_rows", q, b))
                keys.add((scenario, root, "utility_partition", q, b))
                for stream in range(20):
                    keys.add((scenario, root, "diagnostic_permutation", q, stream, b))
        for layer in LAYERS:
            if layer_scenario(layer) != scenario:
                continue
            keys.add((layer, root, "final_model"))
            for b in range(4):
                keys.add((layer, root, "main_permutation", b))
                for fold in range(2):
                    keys.add((layer, root, "utility_model", b, fold))
    for learner in ("LR", "RF"):
        for probe in range(16):
            keys.add(("timing_probe", learner, probe))
    records = [{"coordinate": list(k), "seed": coordinate_seed(*k)}
               for k in sorted(keys, key=lambda k: canonical_json(k))]
    seen = {}
    for record in records:
        seed = record["seed"]
        if seed in seen:
            raise ValueError(f"Distinct seed-coordinate collision: {seen[seed]} and {record['coordinate']}")
        seen[seed] = record["coordinate"]
    return records


def seed_manifest():
    records = seed_records()
    return {"algorithm": "SHA256 canonical [protocol_id,*typed_coordinates], first8 big-endian modulo 2147483647",
            "unique_coordinates": len(records), "collision_check": "passed",
            "records_sha256": hashlib.sha256(canonical_json(records).encode("utf-8")).hexdigest(),
            "intentional_sharing": ["all CSR labels and k values share a selector's rows/utility tables",
                                    "XOR LR/RF share train/test data and nested partitions",
                                    "main/model seeds fixed across six selection resamples",
                                    "M16 is prefix of corresponding M64 sequence",
                                    "random-feature path fixed across resamples; conditional stability only"]}


@dataclass(frozen=True)
class GeneratedDataset:
    scenario: str
    data_seed: int
    train_X: np.ndarray
    train_y: np.ndarray
    test_X: np.ndarray
    test_y: np.ndarray


def _draw(scenario, n, seed):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 6))
    if scenario == "signal_noise":
        logit = 1.5 * X[:, 0] + 0.5 * X[:, 1]
        y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(np.int64)
    elif scenario == "correlated_groups":
        # This exact transform produces joint Gaussians with unit marginal
        # variances and correlation .9; columns 5/6 remain independent noise.
        X[:, 1] = .9 * X[:, 0] + np.sqrt(1 - .9 ** 2) * X[:, 1]
        X[:, 3] = .9 * X[:, 2] + np.sqrt(1 - .9 ** 2) * X[:, 3]
        z1 = (X[:, 0] + X[:, 1]) / np.sqrt(3.8)
        z2 = (X[:, 2] + X[:, 3]) / np.sqrt(3.8)
        logit = 1.5 * (z1 + z2) / np.sqrt(2)
        y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(np.int64)
    elif scenario == "xor":
        latent = np.logical_xor(X[:, 0] > 0, X[:, 1] > 0)
        y = np.logical_xor(latent, rng.random(n) < .05).astype(np.int64)
    else:
        raise ValueError("Unknown scenario")
    return X, y


def generate(scenario, data_seed):
    """Frozen 600 training + independent 2000 test rows; does not write files."""
    _validate_scenario_seed(scenario, data_seed)
    train_X, train_y = _draw(scenario, 600, coordinate_seed(scenario, data_seed, "data", "train"))
    test_X, test_y = _draw(scenario, 2000, coordinate_seed(scenario, data_seed, "data", "test"))
    for array in (train_X, train_y, test_X, test_y):
        array.setflags(write=False)
    return GeneratedDataset(scenario, data_seed, train_X, train_y, test_X, test_y)


def generate_layer(layer, data_seed):
    return generate(layer_scenario(layer), data_seed)


def stratified_subset(labels, fraction, seed):
    """Without replacement; Hamilton allocation, class-id tie break."""
    y = np.asarray(labels)
    if y.ndim != 1 or set(y.tolist()) != {0, 1} or not 0 < fraction <= 1:
        raise ValueError("Binary labels and fraction in (0,1] required")
    counts = np.bincount(y.astype(int), minlength=2)
    n = int(math.floor(len(y) * fraction + .5))
    raw = counts * (n / len(y))
    quotas = np.floor(raw).astype(int)
    order = sorted(range(2), key=lambda c: (-(raw[c] - quotas[c]), c))
    for c in order[:n - int(quotas.sum())]:
        quotas[c] += 1
    if np.min(quotas) < 2:
        raise ValueError("Stratified resample must retain at least two rows of each class")
    rng = np.random.default_rng(seed)
    chosen = np.concatenate([rng.permutation(np.flatnonzero(y == c))[:quotas[c]] for c in (0, 1)])
    return np.sort(chosen)


def stratified_twofold(labels, seed):
    y = np.asarray(labels)
    if y.ndim != 1 or set(y.tolist()) != {0, 1} or min(np.bincount(y.astype(int))) < 2:
        raise ValueError("Two binary utility folds need two members per class")
    rng = np.random.default_rng(seed)
    pieces = [np.array_split(rng.permutation(np.flatnonzero(y == c)), 2) for c in (0, 1)]
    folds = []
    for f in range(2):
        validation = np.sort(np.concatenate([pieces[c][f] for c in (0, 1)]))
        training = np.setdiff1d(np.arange(len(y)), validation)
        folds.append({"train": training, "validation": validation})
    return folds


def nested_training_schedule(train_y, scenario, data_seed):
    """Only train_y is accepted; all returned row indices refer to its 600 rows."""
    _validate_scenario_seed(scenario, data_seed)
    y = np.asarray(train_y)
    if y.shape != (600,):
        raise ValueError("A4 nested schedules require exactly 600 training labels")
    selectors = []
    for q in range(6):
        rows = stratified_subset(y, .8, coordinate_seed(scenario, data_seed, "selection_rows", q))
        games = []
        for b in range(4):
            relative = stratified_subset(y[rows], .8, coordinate_seed(scenario, data_seed, "contribution_rows", q, b))
            game_rows = rows[relative]
            folds = stratified_twofold(y[game_rows], coordinate_seed(scenario, data_seed, "utility_partition", q, b))
            games.append({"b": b, "rows": game_rows.tolist(), "folds": [
                {"train": game_rows[f["train"]].tolist(), "validation": game_rows[f["validation"]].tolist()}
                for f in folds]})
        selectors.append({"q": q, "rows": rows.tolist(), "games": games})
    return selectors


def _utility_table(table, d):
    if type(d) is not int or not 1 <= d <= 8:
        raise ValueError("Exact-table diagnostics support 1 to 8 original variables")
    values = np.asarray(table, dtype=float)
    if values.shape != (1 << d,) or not np.isfinite(values).all():
        raise ValueError("A finite utility for every bitmask, including zero, is required")
    return values


def exact_shapley(table, d=6):
    """Diagnostic-only exact reference. The d<=8 guard is never relaxed."""
    values = _utility_table(table, d)
    phi = np.zeros(d)
    for j in range(d):
        for mask in range(1 << d):
            if mask & (1 << j):
                continue
            size = mask.bit_count()
            weight = math.factorial(size) * math.factorial(d - size - 1) / math.factorial(d)
            phi[j] += weight * (values[mask | (1 << j)] - values[mask])
    np.testing.assert_allclose(phi.sum(), values[-1] - values[0], atol=1e-12, rtol=0)
    return phi


def permutation_estimate(table, M, seed, d=6):
    values = _utility_table(table, d)
    if type(M) is not int or M < 2:
        raise ValueError("At least two independent permutation draws required")
    rng = np.random.default_rng(seed)
    permutations = np.array([rng.permutation(d) for _ in range(M)])
    marginals = np.zeros((M, d))
    for m, order in enumerate(permutations):
        mask = 0
        for j in order:
            updated = mask | (1 << int(j))
            marginals[m, j] = values[updated] - values[mask]
            mask = updated
    np.testing.assert_allclose(marginals.sum(axis=1), values[-1] - values[0], atol=1e-12, rtol=0)
    return {"phi": marginals.mean(axis=0), "mc_variance_of_mean": marginals.var(axis=0, ddof=1) / M,
            "permutations": permutations, "marginals": marginals}


@dataclass(frozen=True)
class MainPermutationEstimates:
    estimates: np.ndarray
    mc_variances: np.ndarray
    M: int = 64
    source: str = "main_permutation"


def main_estimates(tables, layer, data_seed):
    if layer not in LAYERS:
        raise ValueError("Unknown model layer")
    tables = np.asarray(tables, dtype=float)
    if tables.shape != (4, 64):
        raise ValueError("B=4 six-variable utility tables required")
    draws = [permutation_estimate(tables[b], 64, coordinate_seed(layer, data_seed, "main_permutation", b)) for b in range(4)]
    phi = np.array([draw["phi"] for draw in draws])
    mc = np.array([draw["mc_variance_of_mean"] for draw in draws])
    phi.setflags(write=False)
    mc.setflags(write=False)
    return MainPermutationEstimates(phi, mc)


def variance_components(estimates, mc_variances):
    values = np.asarray(estimates, dtype=float)
    mc = np.asarray(mc_variances, dtype=float)
    if values.ndim != 2 or values.shape[0] < 2 or mc.shape != values.shape:
        raise ValueError("At least two aligned game estimates required")
    if not np.isfinite(values).all() or not np.isfinite(mc).all() or (mc < 0).any():
        raise ValueError("Finite contributions and nonnegative MC variances required")
    observed = values.var(axis=0, ddof=1)
    raw = observed - mc.mean(axis=0)
    return {"mean": values.mean(axis=0), "uncorrected": observed,
            "raw_corrected": raw, "clipped_corrected": np.maximum(0., raw), "clipped": raw < 0}


def _average_ranks(column):
    order = np.argsort(column, kind="stable")
    ranks = np.empty(len(column), dtype=float)
    start = 0
    while start < len(column):
        end = start + 1
        while end < len(column) and column[order[end]] == column[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2 + 1
        start = end
    return ranks


def redundancy_matrix(training_X):
    """Absolute Spearman on numeric columns; no encoder or fitting operation.

    Synthetic data have finite numeric values, no missing values or one-hot
    blocks; ranks directly equal the historical encoded-rank association.
    """
    X = np.asarray(training_X, dtype=float)
    if X.ndim != 2 or len(X) < 2 or X.shape[1] != 6 or not np.isfinite(X).all():
        raise ValueError("Finite six-column selection-training frame required")
    ranks = np.column_stack([_average_ranks(X[:, j]) for j in range(6)])
    ranks -= ranks.mean(axis=0)
    norms = np.sqrt((ranks ** 2).sum(axis=0))
    denominator = np.outer(norms, norms)
    R = np.divide(ranks.T @ ranks, denominator, out=np.zeros_like(denominator), where=denominator > 0)
    R = np.clip(np.abs(R), 0, 1)
    np.fill_diagonal(R, 0.)
    return R


def forward_path(scores, redundancy, eta):
    scores = np.asarray(scores, dtype=float)
    R = np.asarray(redundancy, dtype=float)
    d = len(scores)
    if scores.ndim != 1 or d < 1 or R.shape != (d, d) or not np.isfinite(scores).all() or not np.isfinite(R).all():
        raise ValueError("Finite scores and matching redundancy required")
    if not np.isfinite(eta) or eta < 0 or np.any((R < 0) | (R > 1)) or not np.allclose(R, R.T, atol=1e-12, rtol=0) or np.any(np.diag(R) != 0):
        raise ValueError("Symmetric [0,1] redundancy with zero diagonal and nonnegative eta required")
    selected = []
    remaining = list(range(d))
    while remaining:
        pick = min(remaining, key=lambda j: (-(scores[j] - eta / d * R[j, selected].sum()), j))
        selected.append(pick)
        remaining.remove(pick)
    return tuple(selected)


def select_variants(main, redundancy):
    """All eight labels, fixed lambda=.5/eta=.1; diagnostic inputs rejected."""
    if not isinstance(main, MainPermutationEstimates) or main.source != "main_permutation" or main.M != 64:
        raise TypeError("Only main M64 permutation estimates can drive selection")
    if main.estimates.shape != (4, 6):
        raise ValueError("Main selection requires B=4 and d=6")
    stats = variance_components(main.estimates, main.mc_variances)
    variants = {}
    for label in CSR:
        c, s, r = map(int, label)
        variance = stats["clipped_corrected"] if c else stats["uncorrected"]
        score = stats["mean"] - (.5 * np.sqrt(variance) if s else 0)
        path = forward_path(score, redundancy, .1 if r else 0.)
        variants[label] = {"path": list(path), "sets": {str(k): sorted(path[:k]) for k in (1, 2, 3)},
                           "score": score.tolist(), "structural_alias_of": ALIASES.get(label)}
    for alias, canonical in ALIASES.items():
        if variants[alias]["path"] != variants[canonical]["path"]:
            raise AssertionError("CSR structural alias broken")
    return variants


def random_control(data_seed, scenario):
    _validate_scenario_seed(scenario, data_seed)
    path = np.random.default_rng(coordinate_seed(scenario, data_seed, "random_features")).permutation(6).tolist()
    return {"path": path, "sets": {str(k): sorted(path[:k]) for k in (1, 2, 3)}, "all_features": list(range(6))}


def variance_diagnostics(tables, scenario, data_seed, q):
    """20 iid streams; paired M16/M64 prefixes; no fits or selector output.

    Target is sample variance across these B=4 *exact* game contributions.
    Error arrays retain stream and feature axes as nested diagnostics, never
    independent experimental sample counts.
    """
    _validate_scenario_seed(scenario, data_seed)
    tables = np.asarray(tables, dtype=float)
    if tables.shape != (4, 64) or q not in range(6):
        raise ValueError("B=4 six-variable tables and declared selector required")
    exact = np.array([exact_shapley(table) for table in tables])
    target = exact.var(axis=0, ddof=1)
    estimates = {16: [], 64: []}
    for stream in range(20):
        draws = [permutation_estimate(tables[b], 64, coordinate_seed(scenario, data_seed, "diagnostic_permutation", q, stream, b)) for b in range(4)]
        for M in (16, 64):
            marginals = [draw["marginals"][:M] for draw in draws]
            stats = variance_components(np.array([a.mean(axis=0) for a in marginals]),
                                        np.array([a.var(axis=0, ddof=1) / M for a in marginals]))
            estimates[M].append(stats)
    output = {"target": target.tolist(), "exact_phi": exact.tolist(), "target_kind": "finite_B_exact_sample_variance",
              "independent_unit": "generated_dataset", "n_streams_nested": 20, "selection_eligible": False, "M": {}}
    for M, draws in estimates.items():
        entry = {}
        for name in ("uncorrected", "raw_corrected", "clipped_corrected"):
            values = np.array([s[name] for s in draws])
            errors = values - target
            entry[name] = {"values": values.tolist(), "bias_by_feature": errors.mean(axis=0).tolist(),
                           "mse_by_feature": (errors ** 2).mean(axis=0).tolist(),
                           "bias": float(errors.mean()), "mse": float((errors ** 2).mean())}
        entry["clipping_rate_by_feature"] = np.array([s["clipped"] for s in draws]).mean(axis=0).tolist()
        entry["clipping_rate"] = float(np.array([s["clipped"] for s in draws]).mean())
        output["M"][str(M)] = entry
    return output


def recovery_metrics(selected, scenario):
    """Diagnostic truth only: no object returned here enters select_variants."""
    selected = set(selected)
    if scenario not in SCENARIOS or not selected or not selected <= set(range(6)):
        raise ValueError("Known scenario and nonempty feature subset required")
    if scenario == "correlated_groups":
        groups = ({0, 1}, {2, 3})
        return {"group_coverage": sum(bool(selected & g) for g in groups) / 2,
                "same_group_excess": sum(max(0, len(selected & g) - 1) for g in groups),
                "noise_count": len(selected & {4, 5})}
    return {"signal_recall": len(selected & {0, 1}) / 2,
            "joint_signal_recovery": int({0, 1} <= selected),
            "weak_signal_missed": int(1 not in selected) if scenario == "signal_noise" else None,
            "noise_count": len(selected - {0, 1})}


def stability_metrics(selections, d=6):
    """Nogueira unbiased indicator variance and nested descriptive Jaccards."""
    if len(selections) != 6:
        raise ValueError("A4 stability uses six complete selection resamples")
    indicators = np.zeros((len(selections), d))
    for i, selection in enumerate(selections):
        if len(set(selection)) != len(selection) or not set(selection) <= set(range(d)):
            raise ValueError("Invalid selection indices")
        indicators[i, list(selection)] = 1
    mean_k = indicators.sum(axis=1).mean()
    denominator = (mean_k / d) * (1 - mean_k / d)
    value = None if denominator == 0 else float(1 - indicators.var(axis=0, ddof=1).mean() / denominator)
    jaccards = []
    for a, b in itertools.combinations(selections, 2):
        union = set(a) | set(b)
        jaccards.append(len(set(a) & set(b)) / len(union) if union else None)
    valid = [j for j in jaccards if j is not None]
    return {"nogueira": value, "nogueira_status": "undefined_all_empty_or_full" if denominator == 0 else "ok",
            "pairwise_jaccard": jaccards, "mean_jaccard": float(np.mean(valid)) if valid else None,
            "feature_frequency": indicators.mean(axis=0).tolist(), "independent_unit": "generated_dataset"}


def paired_dataset_summary(differences):
    """One prespecified layer/endpoint at a time; exactly eight dataset values."""
    values = np.asarray(differences, dtype=float)
    if values.shape != (8,) or not np.isfinite(values).all():
        raise ValueError("Exactly eight finite dataset-level paired differences required")
    return {"values": values.tolist(), "mean": float(values.mean()), "median": float(np.median(values)),
            "minimum": float(values.min()), "maximum": float(values.max()),
            "positive": int((values > 0).sum()), "zero": int((values == 0).sum()), "negative": int((values < 0).sum())}


def resource_inventory():
    selectors = 4 * 8 * 6
    items = {"main_utility": selectors * 4 * 63 * 2, "main_final_model_bound": selectors * 28,
             "direct_audit": selectors * 7, "cold_replay": 4 * (4 * 63 * 2 + 28), "training_timing_probes": 32}
    total = sum(items.values())
    return {"selectors": selectors, "items": items, "planned_classifier_attempt_bound": total,
            "classifier_attempt_cap": 112000, "planned_encoder_attempt_bound": total, "encoder_attempt_cap": 112000,
            "uncommitted_attempt_reserve": 112000 - total, "kmeans_attempt_cap": 0,
            "LR_classifier_attempt_bound": 3 * (24192 + 1344 + 336 + 532) + 16,
            "RF_classifier_attempt_bound": 24192 + 1344 + 336 + 532 + 16,
            "redundancy_encoder_attempts": 0}


def audit_targets():
    """Seven direct refits for every layer/root/q, fixed before outcomes."""
    return [{"kind": "utility", "b": j % 4, "fold": j % 2, "mask": (1 << (j + 1)) - 1}
            for j in range(6)] + [{"kind": "final", "csr": "111", "k": 2}]


def timing_projection(lr_attempt_seconds, rf_attempt_seconds):
    """16 probes/learner; conservative max observed duration, no fits here.

    Inputs are full instrumented preprocessing+classifier durations. Maximum
    duration within each learner is applied to that learner's whole inventory;
    this is a feasibility warning, never a wall-clock guarantee.
    """
    lr, rf = np.asarray(lr_attempt_seconds, dtype=float), np.asarray(rf_attempt_seconds, dtype=float)
    if lr.shape != (16,) or rf.shape != (16,) or not np.isfinite(lr).all() or not np.isfinite(rf).all() or (lr <= 0).any() or (rf <= 0).any():
        raise ValueError("Sixteen positive finite durations per learner required")
    inv = resource_inventory()
    rf_hours = float(rf.max() * inv["RF_classifier_attempt_bound"] / 3600)
    total_hours = float(lr.max() * inv["LR_classifier_attempt_bound"] / 3600 + rf_hours)
    return {"total_projected_hours": total_hours, "RF_projected_hours": rf_hours,
            "total_warning_hours": 12, "RF_warning_hours": 6, "invocation_cap_hours": 1,
            "status": "pass" if total_hours <= 12 and rf_hours <= 6 else "stop_defer",
            "projection_only": True, "authorizes_execution": False}
