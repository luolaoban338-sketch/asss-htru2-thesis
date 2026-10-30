"""One fixed, sample-conditioned retraining game over original variables."""
import hashlib
import json
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.exceptions import ConvergenceWarning
from src.contracts import TrainingPartition, assert_disjoint
from src.data import seed_for
from src.preprocessing import TabularEncoder
from src.samplers import select_rows
from src.evaluation import classification_metrics
from src.models import make_classifier, fit_count_metadata

ROOT = Path(__file__).resolve().parents[1]


def check_stop():
    if (ROOT / 'STOP_REQUESTED').exists():
        raise SystemExit(75)


class FeatureUtility:
    def __init__(self, X, partition, numeric, categorical, method, baseline, game_seed, n_splits=3):
        check_stop()
        partition.validate()
        if tuple(X.columns) != partition.feature_names or len(X) != len(partition.row_ids):
            raise ValueError('Training frame and partition mismatch')
        self.X = X.copy(deep=True).reset_index(drop=True)
        self.features = tuple(X.columns)
        self.y = np.asarray(partition.labels, dtype=int)
        self.ids = np.asarray(partition.row_ids, dtype=str)
        self.groups = np.asarray(partition.group_ids, dtype=str)
        self.numeric = tuple(numeric); self.categorical = tuple(categorical)
        if set(numeric) & set(categorical) or set(numeric) | set(categorical) != set(X.columns):
            raise ValueError('Explicit numeric/categorical schema required')
        # Copy settings so later mutation of caller dictionaries cannot change a cached game.
        self.baseline = json.loads(json.dumps(baseline))
        self.method = json.loads(json.dumps(method))
        # Validate the named learner before constructing any game state.
        make_classifier(self.baseline['model'], seed_for(game_seed, 'model_validation'))
        self.folds = []; self.cache = {}; self.fit_count = 0; self.constant_count = 0; self.cache_hits = 0
        self.utility_requests = 0; self.no_fit_requests = 0
        split_seed = seed_for(game_seed, 'utility_split')
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=split_seed)
        for fold, (tr, va) in enumerate(splitter.split(self.X, self.y, self.groups)):
            check_stop()
            part = TrainingPartition(tuple(self.ids[tr]), tuple(map(int, self.y[tr])), self.features,
                tuple(self.groups[tr]), 'train', f'{partition.partition_id}:utility{fold}')
            assert_disjoint(part, self.ids[va], self.groups[va])
            if set(self.y[va]) != {0, 1}: raise ValueError('One-class utility validation fold')
            sample_seed = seed_for(game_seed, fold, 'utility_sampler')
            selected, result, diagnostic = select_rows(self.X.iloc[tr], part, numeric, categorical,
                method['rho'], sample_seed, method['strategy'], baseline['sampler'])
            self.folds.append({'train': tr, 'validation': va, 'selected': tr[selected],
                'sampler_seed': sample_seed, 'model_seed': seed_for(game_seed, fold, 'utility_model'),
                'diagnostics': diagnostic})
        context = {'partition': partition.partition_id, 'features': self.features, 'numeric': self.numeric,
            'categorical': self.categorical, 'rows': self.ids.tolist(), 'groups': self.groups.tolist(),
            'labels': self.y.tolist(), 'frame_hash': hashlib.sha256(pd.util.hash_pandas_object(self.X, index=False).values.tobytes()).hexdigest(),
            'method': self.method, 'baseline': self.baseline, 'game_seed': game_seed, 'split_seed': split_seed,
            'n_splits': n_splits, 'folds': self.fold_metadata()}
        self.context = context
        self.context_hash = hashlib.sha256(json.dumps(context, sort_keys=True).encode()).hexdigest()

    def fold_metadata(self):
        return [{'training_ids': self.ids[f['train']].tolist(), 'validation_ids': self.ids[f['validation']].tolist(),
            'selected_ids': self.ids[f['selected']].tolist(), 'sampler_seed': f['sampler_seed'],
            'model_seed': f['model_seed'], 'actual_rho': len(f['selected']) / len(f['train'])} for f in self.folds]

    def utility(self, subset):
        subset = tuple(subset)
        if len(subset) != len(set(subset)) or not set(subset) <= set(self.features):
            raise ValueError('Use unique original variables only')
        selected_features = tuple(c for c in self.features if c in subset)
        self.utility_requests += 1
        if not selected_features:
            self.no_fit_requests += 1
        mask = sum(1 << self.features.index(c) for c in selected_features)
        key = (self.context_hash, mask)
        if key in self.cache:
            self.cache_hits += 1
            return self.cache[key]['utility']
        check_stop()
        scores = []
        for f in self.folds:
            check_stop()
            chosen, va = f['selected'], f['validation']
            if not selected_features:
                # A prior-probability constant fitted only to the selected training labels.
                probs = np.full(len(va), float(self.y[chosen].mean()))
                self.constant_count += 1
            else:
                numeric = tuple(c for c in selected_features if c in self.numeric)
                categorical = tuple(c for c in selected_features if c in self.categorical)
                enc = TabularEncoder(numeric, categorical)
                Z = enc.fit_transform(self.X.iloc[chosen][list(selected_features)])
                model = make_classifier(self.baseline['model'], f['model_seed'])
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter('always')
                    check_stop()
                    model.fit(Z, self.y[chosen])
                if any(issubclass(w.category, ConvergenceWarning) for w in caught):
                    raise RuntimeError('Utility model convergence warning')
                probs = model.predict_proba(enc.transform(self.X.iloc[va][list(selected_features)]))[:, list(model.classes_).index(1)]
                self.fit_count += 1
            scores.append(classification_metrics(self.y[va], probs, self.baseline['threshold'])['f1'])
        value = float(np.mean(scores))
        self.cache[key] = {'features': selected_features, 'fold_f1': scores, 'utility': value}
        return value

    def cache_records(self):
        return [{'context_hash': context, 'mask': mask, **record} for (context, mask), record in sorted(self.cache.items())]

    def fit_metadata(self):
        return fit_count_metadata(self.baseline['model'], self.fit_count)

    def accounting_metadata(self):
        nonempty = sum(bool(record['features']) for record in self.cache.values())
        empty = len(self.cache) - nonempty
        expected = nonempty * len(self.folds)
        return {**self.fit_metadata(), 'utility_requests': self.utility_requests,
            'cache_hits': self.cache_hits, 'unique_coalitions': len(self.cache),
            'nonempty_unique_coalitions': nonempty, 'no_fit_utility_requests': self.no_fit_requests,
            'no_fit_unique_coalitions': empty, 'constant_evaluations': self.constant_count,
            'utility_folds': len(self.folds), 'expected_classifier_fits': expected,
            'fit_reconciliation_passed': self.fit_count == expected,
            'request_reconciliation_passed': self.utility_requests == self.cache_hits + len(self.cache),
            'constant_reconciliation_passed': self.constant_count == empty * len(self.folds)}
