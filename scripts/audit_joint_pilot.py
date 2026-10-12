"""Reconstruct nested boundaries, games, paths, decisions and every saved prediction."""
from pathlib import Path
import argparse
from math import comb, ceil
import sys
import json
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score
from threadpoolctl import threadpool_limits
from src.data import load_dataset, digest
from src.preprocessing import TabularEncoder
from src.samplers import select_rows
from experiments.feature_path import redundancy_matrix
from experiments.feature_utility import FeatureUtility
from scripts.run_joint_pilot import read, cached, save


def assert_close(a, b, atol=1e-12):
    np.testing.assert_allclose(a, b, atol=atol, rtol=0)


def audit_game(game, data, train, excluded, method, config):
    ids = set(data.row_ids[train]); excluded_ids = set(data.row_ids[excluded])
    pool = game['context']['rows']; pool_positions = np.array(game['subsample_positions'])
    assert pool == data.row_ids[pool_positions].tolist()
    assert set(pool) <= ids and not set(pool) & excluded_ids
    assert len(pool) == ceil(len(train) * config['subsample_rho'])
    assert game['context']['labels'] == data.y[pool_positions].tolist()
    assert game['context']['groups'] == data.group_ids[pool_positions].tolist()
    import hashlib
    assert hashlib.sha256(json.dumps(game['context'], sort_keys=True).encode()).hexdigest() == game['context_hash']
    assert game['context']['frame_hash'] == hashlib.sha256(pd.util.hash_pandas_object(data.X.iloc[pool_positions].reset_index(drop=True), index=False).values.tobytes()).hexdigest()
    id_to_pos = {rid: i for i, rid in enumerate(data.row_ids)}
    validation_ids = []
    for f in game['context']['folds']:
        tr = f['training_ids']; va = f['validation_ids']; chosen = f['selected_ids']
        assert len(tr) == len(set(tr)) and len(va) == len(set(va)) and len(chosen) == len(set(chosen))
        assert set(tr) | set(va) == set(pool) and not set(tr) & set(va)
        assert set(chosen) <= set(tr)
        trpos = np.array([id_to_pos[i] for i in tr]); vapos = np.array([id_to_pos[i] for i in va])
        assert not set(data.group_ids[trpos]) & set(data.group_ids[vapos])
        assert not set(data.group_ids[trpos]) & set(data.group_ids[excluded])
        assert not set(data.group_ids[vapos]) & set(data.group_ids[excluded])
        quota_check(data.y[trpos], data.y[[id_to_pos[i] for i in chosen]], method['rho'])
        validation_ids.extend(va)
    assert sorted(validation_ids) == sorted(pool)
    d = len(game['features']); values = np.empty(2**d)
    assert sorted(c['mask'] for c in game['coalitions']) == list(range(2**d))
    for c in game['coalitions']:
        assert c['context_hash'] == game['context_hash']
        assert c['features'] == [f for j, f in enumerate(game['features']) if c['mask'] & 1 << j]
        assert_close(c['utility'], np.mean(c['fold_f1']))
        assert len(c['fold_f1']) == config['utility_folds']
        values[c['mask']] = c['utility']
    # Independent combinatorial weights, without calling exact_shapley.
    exact = [sum((values[m | 1 << j] - values[m]) / (d * comb(d-1, m.bit_count()))
                 for m in range(2**d) if not m & 1 << j) for j in range(d)]
    assert_close(exact, game['exact_phi'])
    permutations = game['permutation']['permutations']
    rng = np.random.default_rng(game['permutation_seed'])
    np.testing.assert_array_equal(permutations, [rng.permutation(d) for _ in range(config['M'])])
    marginal = np.zeros((len(permutations), d))
    for row, order in enumerate(permutations):
        assert sorted(order) == list(range(d))
        mask = 0
        for j in order:
            nxt = mask | 1 << j; marginal[row, j] = values[nxt] - values[mask]; mask = nxt
    assert_close(marginal, game['permutation']['marginals'])
    assert_close(marginal.mean(axis=0), game['permutation']['phi'])
    assert_close(marginal.var(axis=0, ddof=1) / len(permutations), game['permutation']['mc_variance_of_mean'])
    assert_close(marginal.sum(axis=1), values[-1] - values[0])
    assert game['logistic_fits'] == (2**d - 1) * config['utility_folds']
    assert game['constant_evaluations'] == config['utility_folds']


def quota_check(before, after, rho):
    target = ceil(len(before) * rho)
    counts = [sum(before == c) for c in (0,1)]
    quotas = [target * int(n) // len(before) for n in counts]
    order = sorted((0,1), key=lambda c: (-(target * int(counts[c]) % len(before)), c))
    for c in order[:target - sum(quotas)]: quotas[c] += 1
    assert len(after) == target
    assert [sum(after == c) for c in (0,1)] == quotas


def audit_selector(dest, data, train, excluded, config, base, replay):
    selector = cached(dest / 'selector.json'); assert selector is not None
    assert selector['training_positions'] == train.tolist()
    assert selector['training_ids'] == data.row_ids[train].tolist()
    assert not set(data.group_ids[train]) & set(data.group_ids[excluded])
    method = selector['method']; chosen = np.array(selector['selected_positions'])
    quota_check(data.y[train], data.y[chosen], method['rho'])
    assert selector['selected_ids'] == data.row_ids[chosen].tolist()
    assert len(chosen) == len(set(chosen)) and set(chosen) <= set(train)
    local, _, _ = select_rows(data.X.iloc[train], data.partition(train, selector['partition_id']), data.numeric,
        data.categorical, method['rho'], selector['sample_seed'], method['strategy'], base['sampler'])
    np.testing.assert_array_equal(chosen, train[local])
    R = redundancy_matrix(data.X.iloc[chosen], data.numeric, data.categorical)
    assert_close(R, selector['redundancy'])
    games = []
    refits = 0
    for b in range(config['B']):
        path = dest / f'game-{b}.json'
        assert digest(path) == selector['game_sha256'][path.name]
        game = cached(path); assert game is not None
        audit_game(game, data, train, excluded, method, config); games.append(game)
        if replay and b == 0:
            pool = np.array(game['subsample_positions'])
            actual = FeatureUtility(data.X.iloc[pool], data.partition(pool, game['context']['partition']),
                data.numeric, data.categorical, method, base, game['game_seed'], config['utility_folds'])
            assert actual.context_hash == game['context_hash']
            for mask in [0, 1, 3, 2**len(data.X.columns)-1]:
                fs = [f for j, f in enumerate(data.X.columns) if mask & 1 << j]
                assert_close(actual.utility(fs), game['coalitions'][mask]['utility'])
            refits += actual.fit_count
    phi = np.array([g['permutation']['phi'] for g in games])
    mc = np.array([g['permutation']['mc_variance_of_mean'] for g in games])
    corrected = np.sqrt(np.maximum(0, phi.var(axis=0, ddof=1) - mc.mean(axis=0)))
    fs = list(data.X.columns); d = len(fs)
    assert len(selector['paths']) == len(config['lambda_grid']) * len(config['eta_grid'])
    for path in selector['paths']:
        scores = phi.mean(axis=0) - path['lam'] * corrected
        assert_close(scores, path['scores']['score'])
        assert_close(corrected, path['scores']['corrected_sd'])
        selected = []
        for k, step in enumerate(path['path'], 1):
            remaining = [j for j in range(d) if j not in selected]
            gains = {j: scores[j] - path['eta'] / d * sum(R[j,i] for i in selected) for j in remaining}
            winner = sorted(remaining, key=lambda j: (-gains[j], j))[0]
            selected.append(winner)
            assert step['added'] == fs[winner] and step['features'] == [fs[j] for j in selected]
            assert step['k'] == k
            assert_close(step['increment'], gains[winner])
            assert_close(step['objective'], sum(scores[j] for j in selected) - path['eta']/d * sum(R[j,i] for j in selected for i in selected if i < j))
        assert len(selected) == d and len(set(selected)) == d
    return refits


def audit_model(path, data, train, evaluation, selector, base):
    model = cached(path); assert model is not None
    chosen = np.array(model['selected_positions']); ev = np.array(model['evaluation_positions'])
    assert model['context']['training_ids'] == data.row_ids[train].tolist()
    assert model['context']['selected_ids'] == selector['selected_ids']
    assert model['context']['evaluation_ids'] == data.row_ids[evaluation].tolist()
    np.testing.assert_array_equal(chosen, selector['selected_positions'])
    np.testing.assert_array_equal(ev, evaluation)
    fs = model['context']['features']; assert fs == [f for f in data.X.columns if f in fs]
    assert model['targets'] == data.y[ev].tolist()
    enc = TabularEncoder([f for f in fs if f in data.numeric], [f for f in fs if f in data.categorical])
    enc.fit(data.X.iloc[chosen][fs])
    assert enc.diagnostics(data.X.iloc[ev][fs]) == model['encoder']
    probabilities = expit(enc.transform(data.X.iloc[ev][fs]) @ np.array(model['coef'])[0] + model['intercept'][0])
    assert_close(probabilities, model['probabilities'])
    pred = probabilities >= base['threshold']; y = data.y[ev]
    metric = model['metrics']
    assert_close(f1_score(y, pred, zero_division=0), metric['f1'])
    assert_close(roc_auc_score(y, probabilities), metric['roc_auc'])
    assert_close(average_precision_score(y, probabilities), metric['average_precision'])
    assert [int(np.sum((y == yy) & (pred == pp))) for yy, pp in [(0,0),(0,1),(1,0),(1,1)]] == [metric[k] for k in ['tn','fp','fn','tp']]
    return model


def independent_choice(rows, threshold, tol):
    eligible = [r for r in rows if r['mean_f1'] + tol >= threshold]
    assert eligible
    smallest = min(r['cell_fraction'] for r in eligible)
    eligible = [r for r in eligible if r['cell_fraction'] - smallest <= tol]
    min_k = min(r['k'] for r in eligible); eligible = [r for r in eligible if r['k'] == min_k]
    max_f1 = max(r['mean_f1'] for r in eligible)
    return min(r['candidate_id'] for r in eligible if max_f1 - r['mean_f1'] <= tol)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--run-id', required=True); ap.add_argument('--fold', type=int)
    args = ap.parse_args(); run = ROOT / 'runs' / args.run_id
    config = read(run/'config.json'); base = read(run/'baseline_config.json'); manifest = read(run/'manifest.json')
    assert digest(run/'config.json') == manifest['fingerprint']['config']
    assert digest(run/'baseline_config.json') == manifest['fingerprint']['baseline']
    for rel, sha in manifest['fingerprint']['code'].items(): assert digest(run/'code'/rel) == sha
    data = load_dataset(config['dataset']); outer = read(run/'outer_splits.json')
    assert data.sha256 == manifest['fingerprint']['data']
    assert digest(run/'outer_splits.json') == manifest['fingerprint']['splits']
    counts = dict(folds=0, selectors=0, games=0, saved_models=0, outer_comparisons=0, direct_utility_refits=0, baseline_matches=0)
    with threadpool_limits(limits=1):
        for fold in config['outer_folds']:
            if args.fold is not None and args.fold != fold: continue
            folder = run / f'fold-{fold}'
            train = np.array(outer['folds'][fold]['train_positions']); test = np.array(outer['folds'][fold]['test_positions'])
            inner = cached(folder/'inner_splits.json'); validation = []; all_rows = []
            for i, part in enumerate(inner['folds']):
                tr = np.array(part['training_positions']); va = np.array(part['validation_positions'])
                assert set(tr) | set(va) == set(train) and not set(tr) & set(va)
                assert not set(data.group_ids[tr]) & set(data.group_ids[va])
                validation.extend(va.tolist()); dest = folder/f'inner-{i}'
                for method in config['methods']:
                    counts['direct_utility_refits'] += audit_selector(dest/method, data, tr, np.concatenate([va,test]), config, base, fold == 0 and i == 0)
                    counts['selectors'] += 1; counts['games'] += config['B']
                    selector = cached(dest/method/'selector.json')
                    for path in (dest/method/'models').glob('mask-*.json'):
                        if path.name.endswith('.complete.json'): continue
                        audit_model(path, data, tr, va, selector, base); counts['saved_models'] += 1
                rows = cached(dest/'candidates.json'); assert len(rows) == len(config['methods'])*len(config['lambda_grid'])*len(config['eta_grid'])*len(data.X.columns)
                assert sorted(r['candidate_id'] for r in rows) == list(range(len(rows)))
                for r in rows:
                    model = cached(dest/r['model_path']); selector = cached(dest/r['method']/'selector.json')
                    assert digest(dest/r['model_path']) == r['model_sha256']
                    path = next(p for p in selector['paths'] if p['lam'] == r['lam'] and p['eta'] == r['eta'])
                    expected = [f for f in data.X.columns if f in path['path'][r['k']-1]['features']]
                    assert r['features'] == expected == model['context']['features']
                    assert_close(r['f1'], model['metrics']['f1']); assert_close(r['actual_rho'], selector['actual_rho'])
                all_rows.append(rows)
            assert sorted(validation) == sorted(train)
            decisions = cached(folder/'decisions.json'); summaries = decisions['candidate_scores']
            for row in summaries:
                matches = [next(r for r in rows if r['candidate_id'] == row['candidate_id']) for rows in all_rows]
                assert_close(row['fold_f1'], [r['f1'] for r in matches])
                assert_close(row['mean_f1'], np.mean([r['f1'] for r in matches]))
                assert_close(row['cell_fraction'], np.mean([r['actual_rho'] for r in matches])*row['k']/len(data.X.columns))
            full = [r for r in summaries if r['method'] == 'full' and r['k'] == len(data.X.columns)]
            assert_close([r['mean_f1'] for r in full], decisions['baseline_inner_f1'])
            threshold = decisions['baseline_inner_f1'] - config['epsilon']; tol = config['numeric_tolerance']
            assert_close(threshold, decisions['threshold'])
            assert decisions['feasible_candidate_ids'] == [r['candidate_id'] for r in summaries if r['mean_f1'] >= threshold-tol]
            chosen = decisions['selections']
            restrictions = {'full': full, 'sample_only': [r for r in summaries if r['k'] == len(data.X.columns)],
                'feature_only': [r for r in summaries if r['method'] == 'full'], 'joint': summaries,
                'sequential': [r for r in summaries if r['method'] == chosen['sample_only']['method']]}
            for name, method in [('random','random_half'), ('ordinary','ordinary_half'), ('robust','robust_half')]:
                restrictions['joint_'+name] = [r for r in summaries if r['method'] in ('full',method)]
            for arm, allowed in restrictions.items():
                assert chosen[arm]['candidate_id'] == independent_choice(allowed, threshold, tol)
            for method in config['methods']:
                dest = folder/'refit'/method
                audit_selector(dest, data, train, test, config, base, False); counts['selectors'] += 1; counts['games'] += config['B']
                selector = cached(dest/'selector.json')
                for path in (dest/'models').glob('mask-*.json'):
                    if path.name.endswith('.complete.json'): continue
                    audit_model(path, data, train, test, selector, base); counts['saved_models'] += 1
            for arm in config['comparisons']:
                result = cached(folder/'outer'/f'{arm}.json'); assert result['candidate'] == chosen[arm]
                assert digest(folder/result['model_path']) == result['model_sha256']
                assert digest(folder/'decisions.json') == result['decisions_sha256']
                model = cached(folder/result['model_path']); selector = cached(folder/'refit'/chosen[arm]['method']/'selector.json')
                path = next(p for p in selector['paths'] if p['lam'] == chosen[arm]['lam'] and p['eta'] == chosen[arm]['eta'])
                assert result['features'] == [f for f in data.X.columns if f in path['path'][chosen[arm]['k']-1]['features']]
                assert result['features'] == model['context']['features'] and result['metrics'] == model['metrics']
                assert result['selected_n'] == len(model['selected_positions']) and result['test_n'] == len(test)
                assert_close(result['actual_rho'], result['selected_n']/len(train))
                assert_close(result['cell_fraction'], result['actual_rho']*len(result['features'])/len(data.X.columns))
                counts['outer_comparisons'] += 1
                if arm == 'full':
                    baseline = pd.read_csv(ROOT/'runs'/config['parent_run']/data.name/f'fold-{fold}'/'full'/'predictions.csv', float_precision='round_trip')
                    assert baseline.row_id.tolist() == model['context']['evaluation_ids']
                    np.testing.assert_array_equal(baseline.probability.to_numpy(), model['probabilities'])
                    counts['baseline_matches'] += 1
            counts['folds'] += 1
            print(f'Independent audit passed fold={fold}', flush=True)
    save(ROOT/'qa'/f'{args.run_id}_audit{("_fold"+str(args.fold)) if args.fold is not None else ""}.json', {'status':'passed', **counts})
    print(json.dumps(counts), flush=True)


if __name__ == '__main__': main()
