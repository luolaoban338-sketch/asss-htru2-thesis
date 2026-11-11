"""Stage-private, session-gated scientific operations.

No split construction occurs here: all raw memberships and seeds come from the
new frozen partition artifact. Test factories must be explicitly supplied.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol
import hashlib
import json
import math
import time
import warnings
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning
from .preprocessing import TabularEncoder
from .contracts import TrainingPartition, assert_disjoint, class_quotas
from .evaluation import classification_metrics, group_metrics
from .shapley import permutation_shapley, corrected_scores


class SessionProtocol(Protocol):
    role: str
    task: str
    def require_execution(self): ...
    def operation(self, kind: str, metadata: dict): ...
    def checkpoint(self): ...
    def artifact_path(self, relative: str) -> Path: ...


def serial(value):
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, Path): return str(value)
    if isinstance(value, dict): return {str(k): serial(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [serial(v) for v in value]
    return value


def fingerprint(value):
    return hashlib.sha256(json.dumps(serial(value), sort_keys=True, separators=(',', ':'),
                                      ensure_ascii=True, allow_nan=False).encode()).hexdigest()


def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path): return json.loads(Path(path).read_text('utf-8-sig'))


def commit(path, value):
    """Immutable result plus digest; no overwrite of completed or partial bytes."""
    path = Path(path)
    payload = json.dumps(serial(value), ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    if path.exists():
        if path.read_text('utf-8') != payload: raise RuntimeError('Immutable artifact differs: ' + str(path))
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + '.partial')
        if temporary.exists(): raise RuntimeError('Preserve interrupted artifact for recovery review: ' + str(temporary))
        temporary.write_text(payload, encoding='utf-8')
        temporary.replace(path)
    marker = path.with_suffix(path.suffix + '.sha256')
    wanted = digest(path) + '\n'
    if marker.exists() and marker.read_text('ascii') != wanted: raise RuntimeError('Artifact marker changed')
    if not marker.exists(): marker.write_text(wanted, encoding='ascii')
    return serial(value)


def cached(path):
    path = Path(path)
    marker = path.with_suffix(path.suffix + '.sha256')
    if not path.exists(): return None
    if not marker.exists(): raise RuntimeError('Incomplete artifact requires recovery review: ' + str(path))
    if marker.read_text('ascii').strip() != digest(path): raise RuntimeError('Artifact checksum mismatch')
    return read(path)


def positions(segment):
    return np.asarray(segment['positions'] if isinstance(segment, dict) else segment, dtype=int)


@dataclass
class Data:
    X: pd.DataFrame
    y: np.ndarray
    row_ids: np.ndarray
    group_ids: np.ndarray
    audit: pd.DataFrame
    numeric: tuple
    categorical: tuple
    source_hash: str

    def partition(self, rows, name):
        rows = positions(rows)
        return TrainingPartition(tuple(self.row_ids[rows]), tuple(map(int, self.y[rows])),
                                 tuple(self.X.columns), tuple(self.group_ids[rows]), 'train', name)


def load_data(config, paths):
    source = config['data_source']
    table = Path(source['table'])
    dictionary = Path(source['dictionary'])
    if digest(table) != source['table_sha256'] or digest(dictionary) != source['dictionary_sha256']:
        raise RuntimeError('Adult source checksum mismatch')
    schema = read(dictionary)
    fields = [r for r in schema if r['role'] == 'predictor']
    features = [r['field'] for r in fields]
    numeric = tuple(r['field'] for r in fields if r['type'] == 'numeric')
    categorical = tuple(r['field'] for r in fields if r['type'] != 'numeric')
    if len(features) != 12 or len(numeric) != 6 or len(categorical) != 6:
        raise RuntimeError('Adult original-variable schema changed')
    if set(features) & {'race', 'sex', 'target', 'group_id', 'row_id'}:
        raise RuntimeError('Audit attribute entered predictor schema')
    frame = pd.read_csv(table)
    if len(frame) != 32561 or not frame.row_id.is_unique or set(frame.target) != {0, 1}:
        raise RuntimeError('Adult row identity or target changed')
    return Data(frame[features].copy(), frame.target.to_numpy(int), frame.row_id.to_numpy(str),
                frame.group_id.to_numpy(str), frame.filter(regex=r'^audit_').copy(), numeric,
                categorical, source['table_sha256'])


def checked_fit(session, kind, estimator, X, y=None, *, metadata):
    session.require_execution()
    session.checkpoint()
    started = time.perf_counter()
    with session.operation(kind, metadata), warnings.catch_warnings():
        warnings.simplefilter('error', ConvergenceWarning)
        result = estimator.fit(X) if y is None else estimator.fit(X, y)
    return result, time.perf_counter() - started


def checked_transform(session, encoder, X):
    session.require_execution()
    return encoder.transform(X)


def checked_predict(session, model, X):
    session.require_execution()
    return model.predict_proba(X)


def encoder_for(data, features, session, mode='model'):
    return TabularEncoder(tuple(f for f in features if f in data.numeric),
                          tuple(f for f in features if f in data.categorical), mode,
                          session=session)


def make_model(config, seed):
    params = dict(config['learner_parameters'])
    return LogisticRegression(**params, random_state=int(seed))


def assert_pair(data, training, evaluation):
    assert_disjoint(data.partition(positions(training), 'checked_training'),
                    data.row_ids[positions(evaluation)], data.group_ids[positions(evaluation)])


def sample_rows(data, raw_rows, method, seed, config, session, metadata):
    """Frozen robust sampler, with all encoder/KMeans work counted."""
    session.require_execution()
    rows = positions(raw_rows)
    y, ids = data.y[rows], data.row_ids[rows]
    quotas = class_quotas(tuple(map(int, y)), 1.0 if method == 'full' else 0.5)
    if method == 'full': return rows.copy(), {'actual_rho': 1., 'class_quotas': quotas, 'encoder_seconds': [], 'kmeans_seconds': []}
    if method != 'robust_half': raise ValueError('Only full and robust_half are frozen')
    enc = encoder_for(data, tuple(data.X.columns), session, 'robust_geometry')
    _, enc_seconds = checked_fit(session, 'encoder', enc, data.X.iloc[rows], metadata={**metadata, 'component': 'geometry'})
    Z = checked_transform(session, enc, data.X.iloc[rows])
    selected, clusters, km_seconds = [], [], []
    settings = config['sampler']
    for c in (0, 1):
        loc = np.flatnonzero(y == c)
        loc = loc[np.argsort(ids[loc], kind='stable')]
        k = min(settings['max_clusters'], math.floor(math.sqrt(len(loc))), quotas[c], len(np.unique(Z[loc], axis=0)))
        km = KMeans(n_clusters=k, init='k-means++', n_init=settings['n_init'],
                    max_iter=settings['max_iter'], algorithm=settings['algorithm'],
                    random_state=seed_for(seed, 'kmeans', c))
        _, elapsed = checked_fit(session, 'kmeans', km, Z[loc], metadata={**metadata, 'class': c})
        km_seconds.append(elapsed)
        labels = sorted(np.unique(km.labels_), key=lambda label: min(ids[loc[km.labels_ == label]]))
        sizes = np.asarray([np.sum(km.labels_ == label) for label in labels], int)
        q = np.ones(len(sizes), int)
        while int(q.sum()) < quotas[c]:
            active = np.flatnonzero(q < sizes)
            remaining = quotas[c] - int(q.sum()); weight = int(sizes[active].sum())
            q[active] += np.minimum((remaining * sizes[active]) // weight, sizes[active] - q[active])
            remainder = quotas[c] - int(q.sum())
            eligible = active[q[active] < sizes[active]]
            order = sorted(eligible, key=lambda j: (-int(remaining * sizes[j] % weight), int(j)))
            for j in order[:remainder]: q[j] += 1
        for label, capacity, quota in zip(labels, sizes, q):
            member = np.flatnonzero(km.labels_ == label)
            distance = np.linalg.norm(Z[loc[member]] - km.cluster_centers_[label], axis=1)
            order = np.lexsort((ids[loc[member]], distance))
            ranks = np.asarray([(len(member) - 1) // 2] if quota == 1 else
                               [j * (len(member) - 1) // (int(quota) - 1) for j in range(int(quota))])
            selected.extend(rows[loc[member[order[ranks]]]].tolist())
            clusters.append({'class': c, 'size': int(capacity), 'quota': int(quota), 'n_iter': int(km.n_iter_)})
    selected = np.asarray(sorted(selected), int)
    if {c: int(np.sum(data.y[selected] == c)) for c in (0, 1)} != quotas: raise RuntimeError('Sampler class quota mismatch')
    return selected, {'actual_rho': len(selected) / len(rows), 'class_quotas': quotas,
                      'encoder': enc.diagnostics(data.X.iloc[rows]), 'clusters': clusters,
                      'encoder_seconds': [enc_seconds], 'kmeans_seconds': km_seconds}


def seed_for(root, *parts):
    return int.from_bytes(hashlib.sha256('|'.join(map(str, (root,) + parts)).encode()).digest()[:4], 'big')


def redundancy(data, selected, session, metadata):
    enc = encoder_for(data, tuple(data.X.columns), session)
    _, elapsed = checked_fit(session, 'encoder', enc, data.X.iloc[selected], metadata={**metadata, 'component': 'redundancy'})
    Z = checked_transform(session, enc, data.X.iloc[selected])
    ranks = pd.DataFrame(Z).rank(method='average').to_numpy(copy=True)
    ranks -= ranks.mean(axis=0)
    norms = np.sqrt((ranks ** 2).sum(axis=0)); denominator = np.outer(norms, norms)
    corr = np.abs(np.divide(ranks.T @ ranks, denominator, out=np.zeros_like(denominator), where=denominator > 0))
    corr = np.clip(corr, 0, 1)
    features = tuple(data.X.columns); R = np.zeros((len(features), len(features)))
    for i, f in enumerate(features):
        for j in range(i + 1, len(features)):
            R[i, j] = R[j, i] = float(corr[np.ix_(enc.blocks_[f], enc.blocks_[features[j]])].max())
    return R, {'encoder_seconds': elapsed, 'encoder': enc.diagnostics(data.X.iloc[selected])}


def model_cycle(data, selected, evaluation, features, seed, config, session, metadata, *,
                allow_timing_overlap=False, keep_predictions=True):
    session.require_execution()
    selected, evaluation = positions(selected), positions(evaluation)
    features = tuple(features)
    if len(features) != len(set(features)) or not set(features) <= set(data.X.columns):
        raise ValueError('Unknown or duplicated original-variable feature')
    features = tuple(f for f in data.X.columns if f in features)
    if not features or len(features) != len(set(features)): raise ValueError('Nonempty original variable subset required')
    if allow_timing_overlap:
        if session.role != 'probe' or metadata.get('phase') != 'reference_timing':
            raise PermissionError('Only declared training-side reference timing may overlap')
    else: assert_pair(data, selected, evaluation)
    started = time.perf_counter(); cpu = time.process_time()
    metadata = session.bind_classifier(selected, evaluation, features, metadata)
    enc = encoder_for(data, features, session)
    _, encoder_seconds = checked_fit(session, 'encoder', enc, data.X.iloc[selected][list(features)], metadata=metadata)
    Z = checked_transform(session, enc, data.X.iloc[selected][list(features)])
    model = make_model(config, seed)
    _, classifier_seconds = checked_fit(session, 'classifier', model, Z, data.y[selected], metadata=metadata)
    probabilities = checked_predict(session, model, checked_transform(session, enc, data.X.iloc[evaluation][list(features)]))[:, list(model.classes_).index(1)]
    metrics = classification_metrics(data.y[evaluation], probabilities, config['threshold'])
    result = {'features': list(features), 'selected_positions': selected.tolist(), 'evaluation_positions': evaluation.tolist(),
              'seed': int(seed), 'encoder': enc.diagnostics(data.X.iloc[evaluation][list(features)]),
              'iterations': np.asarray(model.n_iter_).tolist(), 'encoder_seconds': encoder_seconds,
              'classifier_seconds': classifier_seconds, 'cycle_seconds': time.perf_counter() - started,
              'cpu_seconds': time.process_time() - cpu, 'timing_only': not keep_predictions,
              'prediction_overlap_allowed_for_timing': bool(allow_timing_overlap)}
    if keep_predictions:
        result.update(probabilities=probabilities.tolist(), metrics=metrics,
                      group_metrics=group_metrics(data.audit.iloc[evaluation], data.y[evaluation], probabilities, config['threshold']))
    rate = 't_reference' if metadata['classifier_bucket'] == 'reference' else ('t_utility' if metadata['phase'] == 'utility' else 't_full_candidate')
    session.record_event('resource_rate', {'rate': rate}, result['cycle_seconds'])
    return result, metrics['f1']


class UtilityGame:
    def __init__(self, data, game_record, method, config, session, metadata):
        session.require_execution()
        self.data, self.record, self.method, self.config, self.session, self.metadata = data, game_record, method, config, session, metadata
        self.features = tuple(data.X.columns); self.cache = {}; self.cycles = []; self.folds = []
        self.geometry = []; begin = time.perf_counter()
        for fold in game_record['utility_folds']:
            train, valid = positions(fold['train_positions']), positions(fold['validation_positions'])
            assert_pair(data, train, valid)
            chosen, diagnostic = sample_rows(data, train, method, fold['sampler_seed'], config, session,
                                             {**metadata, 'component': 'utility_geometry', 'utility_fold': fold['fold']})
            self.folds.append({**fold, 'selected_positions': chosen.tolist()})
            self.geometry.append(diagnostic)
        self.initialization_seconds = time.perf_counter() - begin
        session.record_event('resource_rate', {'rate': 't_game_setup'}, self.initialization_seconds)

    def utility(self, features):
        self.session.require_execution(); self.session.checkpoint()
        features = tuple(features)
        if len(features) != len(set(features)) or not set(features) <= set(self.features):
            raise ValueError('Unknown or duplicated utility feature')
        selected_features = tuple(f for f in self.features if f in features)
        mask = sum(1 << self.features.index(f) for f in selected_features)
        if mask in self.cache: return self.cache[mask]['utility']
        values = []
        for fold in self.folds:
            selected, valid = positions(fold['selected_positions']), positions(fold['validation_positions'])
            if not selected_features:
                value = classification_metrics(self.data.y[valid], np.full(len(valid), self.data.y[selected].mean()), self.config['threshold'])['f1']
            else:
                cycle, value = model_cycle(self.data, selected, valid, selected_features, fold['model_seed'],
                    self.config, self.session, {**self.metadata, 'phase': self.metadata.get('phase', 'utility'),
                    'utility_fold': fold['fold'], 'mask': mask}, keep_predictions=False)
                self.cycles.append({k: cycle[k] for k in ('cycle_seconds', 'cpu_seconds', 'encoder_seconds', 'classifier_seconds', 'iterations')})
            values.append(value)
        self.cache[mask] = {'mask': mask, 'features': list(selected_features), 'fold_f1': values, 'utility': float(np.mean(values))}
        return self.cache[mask]['utility']


def feature_path(features, scores, R, eta):
    remaining = list(range(len(features))); chosen = []; path = []
    while remaining:
        increments = {j: float(scores[j] - eta / len(features) * R[j, chosen].sum()) for j in remaining}
        pick = min(remaining, key=lambda j: (-increments[j], j))
        chosen.append(pick); remaining.remove(pick)
        path.append([f for f in features if features.index(f) in chosen])
    return path


def build_selector(data, context, method, config, session, destination, *, cache_allowed=True):
    session.require_execution()
    dest = Path(destination); selector_file = dest / 'selector.json'
    bound = fingerprint({'config': config, 'context': context, 'method': method, 'data': data.source_hash})
    old = cached(selector_file) if cache_allowed else None
    if old is not None:
        if old['input_fingerprint'] != bound: raise RuntimeError('Selector input changed')
        for rel, sha in old['game_hashes'].items():
            if digest(dest / rel) != sha: raise RuntimeError('Selector game changed')
        return old
    train = positions(context['train']); meta = {'fold': context['fold'], 'phase': 'selector', 'context': context['partition_id'], 'method': method}
    begin = time.perf_counter()
    selected, sampler = sample_rows(data, train, method, context['selector_sampler_seed'], config, session, meta)
    R, redundancy_record = redundancy(data, selected, session, meta)
    selector_setup = time.perf_counter() - begin
    session.record_event('resource_rate', {'rate': 't_selector_setup'}, selector_setup)
    games = []
    for record in context['games']:
        session.checkpoint(); path = dest / ('game-%s.json' % record['resample_index'])
        game_bound = fingerprint({'selector': bound, 'game': record})
        saved = cached(path) if cache_allowed else None
        if saved is not None:
            if saved['input_fingerprint'] != game_bound: raise RuntimeError('Game input changed')
        else:
            generator = np.random.default_rng(record['permutation_seeds'][method])
            generated = [generator.permutation(len(data.X.columns)).tolist() for _ in range(config['M'])]
            if generated != record['permutations'][method]: raise RuntimeError('Frozen permutation stream differs')
            game = UtilityGame(data, record, method, config, session,
                               {**meta, 'phase': 'utility', 'resample': record['resample_index']})
            estimate = permutation_shapley(game.utility, game.features, config['M'], record['permutation_seeds'][method])
            if np.asarray(estimate['permutations']).tolist() != record['permutations'][method]:
                raise RuntimeError('Executed permutation stream differs from frozen identity')
            if set(game.cache) - {0} != set(record['nonempty_coalition_masks'][method]):
                raise RuntimeError('Executed coalition cache differs from frozen permutation coverage')
            if len(game.cycles) != len(record['nonempty_coalition_masks'][method]) * config['utility_folds']:
                raise RuntimeError('Utility classifier count does not reconcile with nonempty coalitions')
            saved = commit(path, {'input_fingerprint': game_bound, 'record': record, 'method': method,
                'permutation': estimate, 'folds': game.folds, 'coalitions': list(game.cache.values()),
                'initialization_seconds': game.initialization_seconds, 'geometry': game.geometry,
                'maximum_cycle_seconds': max((x['cycle_seconds'] for x in game.cycles), default=0.),
                'classifier_count': len(game.cycles), 'exact_diagnostic': {'status': 'skipped', 'reason': 'dimension_exceeds_limit', 'limit': 8}})
        games.append(saved)
    score = corrected_scores(np.array([g['permutation']['phi'] for g in games]),
                             np.array([g['permutation']['mc_variance_of_mean'] for g in games]), config['lambda_grid'][0])
    paths = feature_path(tuple(data.X.columns), score['score'], R, config['eta_grid'][0])
    result = {'input_fingerprint': bound, 'context': context['partition_id'], 'method': method,
              'training_positions': train.tolist(), 'selected_positions': selected.tolist(),
              'actual_rho': len(selected) / len(train), 'sampler': sampler, 'redundancy': R,
              'redundancy_encoder': redundancy_record, 'scores': score, 'paths': paths,
              'selector_setup_seconds': selector_setup,
              'game_hashes': {f'game-{r["resample_index"]}.json': digest(dest / f'game-{r["resample_index"]}.json') for r in context['games']}}
    return commit(selector_file, result)
