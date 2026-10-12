"""Reconstruct contaminated training cells and held-out predictions independently."""
from pathlib import Path
import argparse
import json
import math
import sys
import numpy as np
import pandas as pd
from scipy.special import expit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data import load_dataset, digest
from src.preprocessing import TabularEncoder
from src.contracts import class_quotas
from src.evaluation import classification_metrics, group_metrics
from scripts.run_contamination import conditions
from scripts.run_baselines import write_json


def reconstruct(clean, ids, numeric, pollution):
    X = clean.copy(deep=True)
    events = pollution['events']
    assert len(events) == pollution['count'] == math.ceil(pollution['rate'] * len(X))
    assert len({e['local_position'] for e in events}) == len(events)
    for e in events:
        pos, col = e['local_position'], e['feature']
        assert 0 <= pos < len(X) and col in numeric
        assert e['row_id'] == ids[pos]
        assert float(clean.iloc[pos][col]) == e['before']
        a = clean[col].dropna().to_numpy(dtype=float)
        scale = float(np.quantile(a, .75, method='linear') - np.quantile(a, .25, method='linear'))
        assert scale > 0 and scale == e['iqr']
        assert e['sign'] in (-1, 1)
        assert e['after'] == e['before'] + e['sign'] * pollution['multiplier'] * scale
        X[col] = X[col].astype(float)
        X.iloc[pos, X.columns.get_loc(col)] = e['after']
    changed = (X.ne(clean) & ~(X.isna() & clean.isna())).to_numpy()
    assert int(changed.sum()) == len(events)
    assert int(changed.any(axis=1).sum()) == len(events)
    return X


def audit(run_id):
    run = ROOT / 'runs' / run_id
    cfg = json.loads((run / 'config.json').read_text('utf-8'))
    base = json.loads((run / 'baseline_config.json').read_text('utf-8'))
    manifest = json.loads((run / 'manifest.json').read_text('utf-8'))
    assert digest(run / 'config.json') == manifest['fingerprint']['config_sha256']
    assert digest(run / 'baseline_config.json') == manifest['fingerprint']['baseline_config_sha256']
    for path, sha in manifest['fingerprint']['code_sha256'].items():
        assert digest(run / 'code' / path) == sha
    ref = ROOT / 'runs' / cfg['baseline_reference_run']
    records = []; pooled = []; grouped = []; clean_matches = 0; max_error = 0.0
    for name in base['datasets']:
        data = load_dataset(name)
        plan = json.loads((run / name / 'splits.json').read_text('utf-8'))
        assert plan == json.loads((ref / name / 'splits.json').read_text('utf-8'))
        assert digest(ref / name / 'splits.json') == manifest['fingerprint']['split_sha256'][name]
        assert plan['data_sha256'] == data.sha256
        predictions = {}
        for f in plan['folds']:
            tr = np.array(f['train_positions']); te = np.array(f['test_positions']); fold = f['fold']
            assert not set(tr) & set(te)
            assert not set(data.group_ids[tr]) & set(data.group_ids[te])
            assert sorted(np.r_[tr, te].tolist()) == list(range(len(data.X)))
            clean = data.X.iloc[tr].copy()
            by_condition = {}
            for rate, draw, condition in conditions(cfg):
                cd = run / name / f'fold-{fold}' / condition
                pollution = json.loads((cd / 'contamination.json').read_text('utf-8'))
                assert pollution['rate'] == rate and pollution['multiplier'] == cfg['iqr_multiplier']
                X = reconstruct(clean, data.row_ids[tr], data.numeric, pollution)
                polluted = {e['row_id'] for e in pollution['events']}
                by_condition[rate, draw] = pollution
                if rate == .1:
                    prev = by_condition[.05, draw]['events']
                    assert pollution['events'][:len(prev)] == prev
                for method in base['methods']:
                    dest = cd / method['name']
                    r = json.loads((dest / 'record.json').read_text('utf-8'))
                    assert r['status'] == 'completed'
                    assert (r['dataset'], r['fold'], r['rate'], r['draw'], r['method']) == (name, fold, rate, draw, method['name'])
                    assert r['contamination_sha256'] == digest(cd / 'contamination.json')
                    for filename, sha in r['artifact_sha256'].items():
                        assert digest(dest / filename) == sha
                    sel = json.loads((dest / 'selection.json').read_text('utf-8'))
                    model = json.loads((dest / 'model.json').read_text('utf-8'))
                    encstate = json.loads((dest / 'encoder.json').read_text('utf-8'))
                    pred = pd.read_csv(dest / 'predictions.csv', float_precision='round_trip')
                    local = np.array(sel['selected_local_positions']); chosen = np.array(sel['selected_positions'])
                    assert len(local) == len(set(local)) and np.all((0 <= local) & (local < len(tr)))
                    np.testing.assert_array_equal(tr[local], chosen)
                    assert sel['selected_row_ids'] == data.row_ids[chosen].tolist()
                    assert sel['original_features'] == list(data.X.columns)
                    assert pred.row_id.tolist() == data.row_ids[te].tolist()
                    assert pred.group_id.tolist() == data.group_ids[te].tolist()
                    np.testing.assert_array_equal(pred.target, data.y[te])
                    assert model['classes'] == [0, 1]
                    assert model['fit_n_rows'] == encstate['fit_n_rows'] == r['selected_n'] == len(local)
                    assert {c: int(np.sum(data.y[chosen] == c)) for c in (0, 1)} == class_quotas(tuple(map(int, data.y[tr])), method['rho'])
                    if method['rho'] == 1:
                        np.testing.assert_array_equal(chosen, tr)
                    encoder = TabularEncoder(data.numeric, data.categorical).fit(X.iloc[local])
                    assert encoder.params_ == encstate['parameters'] and encoder.blocks_ == encstate['blocks']
                    probability = expit(encoder.transform(data.X.iloc[te]) @ np.array(model['coef'])[0] + model['intercept'][0])
                    error = float(np.max(np.abs(probability - pred.probability.to_numpy())))
                    max_error = max(error, max_error)
                    assert error < 1e-12
                    np.testing.assert_array_equal(pred.prediction, (probability >= base['threshold']).astype(int))
                    metrics = classification_metrics(data.y[te], pred.probability.to_numpy(), base['threshold'])
                    assert metrics == r['metrics']
                    assert group_metrics(data.audit.iloc[te], data.y[te], pred.probability.to_numpy(), base['threshold']) == json.loads((dest / 'group_metrics.json').read_text('utf-8'))
                    retained = len(polluted & set(sel['selected_row_ids']))
                    assert retained == r['retained_contaminated_n'] and r['contaminated_n'] == len(polluted)
                    assert r['contamination_retention'] == (retained / len(polluted) if polluted else None)
                    assert r['contamination_fraction_selected'] == retained / len(local)
                    if rate == 0:
                        old = ref / name / f'fold-{fold}' / method['name']
                        assert model == json.loads((old / 'model.json').read_text('utf-8'))
                        assert sel['selected_row_ids'] == json.loads((old / 'selection.json').read_text('utf-8'))['selected_row_ids']
                        pd.testing.assert_frame_equal(pred, pd.read_csv(old / 'predictions.csv', float_precision='round_trip'), check_exact=True)
                        clean_matches += 1
                    elif method['strategy'] == 'stratified_random':
                        # Fixed labels and seed imply identical row membership despite numeric corruption.
                        clean_sel = json.loads((run / name / f'fold-{fold}' / 'rate-00-draw-0' / method['name'] / 'selection.json').read_text('utf-8'))
                        assert sel['selected_positions'] == clean_sel['selected_positions']
                    records.append({k: r[k] for k in ['dataset', 'fold', 'method', 'rate', 'draw', 'train_n', 'test_n',
                        'selected_n', 'actual_rho', 'contaminated_n', 'retained_contaminated_n', 'contamination_retention',
                        'contamination_fraction_selected', 'sampler_seconds', 'fit_seconds', 'predict_seconds', 'total_seconds']} | metrics)
                    predictions.setdefault((rate, draw, method['name']), []).append(pred)
        for (rate, draw, method), parts in predictions.items():
            full = pd.concat(parts).set_index('row_id')
            assert full.index.is_unique and set(full.index) == set(data.row_ids)
            full = full.loc[data.row_ids]
            pooled.append({'dataset': name, 'rate': rate, 'draw': draw, 'method': method,
                **classification_metrics(data.y, full.probability.to_numpy(), base['threshold'])})
            grouped.extend({'dataset': name, 'rate': rate, 'draw': draw, 'method': method, **g}
                for g in group_metrics(data.audit, data.y, full.probability.to_numpy(), base['threshold']))
    assert len(records) == cfg['planned_fits']
    df = pd.DataFrame(records)
    clean = df[df.rate == 0][['dataset', 'fold', 'method', 'f1', 'roc_auc']].rename(columns={'f1': 'clean_f1', 'roc_auc': 'clean_auc'})
    df = df.merge(clean, on=['dataset', 'fold', 'method'], validate='many_to_one')
    df['delta_f1'] = df.f1 - df.clean_f1
    df['delta_auc'] = df.roc_auc - df.clean_auc
    # Collapse draws first: dependent folds and repeated contamination draws are not independent datasets.
    fold = df.groupby(['dataset', 'method', 'rate', 'fold'], as_index=False).agg(
        f1=('f1', 'mean'), auc=('roc_auc', 'mean'), delta_f1=('delta_f1', 'mean'), delta_auc=('delta_auc', 'mean'),
        retention=('contamination_retention', 'mean'), selected_fraction=('contamination_fraction_selected', 'mean'))
    summary = fold.groupby(['dataset', 'method', 'rate'], as_index=False).agg(
        f1_mean=('f1', 'mean'), f1_sd_dependent_folds=('f1', 'std'), auc_mean=('auc', 'mean'),
        delta_f1_mean=('delta_f1', 'mean'), delta_f1_sd_dependent_folds=('delta_f1', 'std'),
        delta_auc_mean=('delta_auc', 'mean'), retention_mean=('retention', 'mean'),
        contamination_fraction_selected=('selected_fraction', 'mean'))
    timing = df.groupby(['dataset', 'method', 'rate'], as_index=False).agg(median_pipeline_seconds=('total_seconds', 'median'))
    summary = summary.merge(timing, validate='one_to_one')
    contrasts = []
    for comparator in ['ordinary_half', 'random_half', 'full']:
        robust = df[(df.method == 'robust_half') & (df.rate > 0)]
        other = df[(df.method == comparator) & (df.rate > 0)]
        paired = robust.merge(other, on=['dataset', 'fold', 'rate', 'draw'], suffixes=('_robust', '_other'), validate='one_to_one')
        for _, r in paired.iterrows():
            contrasts.append({'dataset': r.dataset, 'fold': r.fold, 'rate': r.rate, 'draw': r.draw, 'comparator': comparator,
                'f1_difference_robust_minus_other': r.f1_robust - r.f1_other,
                'delta_difference_robust_minus_other': r.delta_f1_robust - r.delta_f1_other})
    df.to_csv(run / 'fold_results.csv', index=False)
    fold.to_csv(run / 'fold_draw_means.csv', index=False)
    summary.to_csv(run / 'summary.csv', index=False)
    pd.DataFrame(pooled).to_csv(run / 'pooled_results.csv', index=False)
    pd.DataFrame(grouped).to_csv(run / 'pooled_group_metrics.csv', index=False)
    pd.DataFrame(contrasts).to_csv(run / 'paired_contrasts.csv', index=False)
    write_json(run / 'qa/integrity_checks.json', {'status': 'passed', 'completed_fits': len(records),
        'clean_matches_to_previous_run': clean_matches, 'maximum_prediction_reconstruction_error': max_error,
        'official_adult_test_used': False, 'audit_script_sha256': digest(Path(__file__)),
        'checks': ['code/config snapshots', 'source and split identities', 'train/test group disjointness',
            'exact changed cells and IQR offsets', 'nested contamination levels', 'class quotas and actual row identities',
            'selected-row preprocessing reconstruction', 'saved-coefficient predictions', 'metrics and subgroup metrics',
            'contamination retention', 'all clean results identical to prior baseline', 'complete held-out coverage']})
    print(summary.to_string(index=False))
    print(f'Audit passed: {len(records)} fits; {clean_matches} exact clean matches; max reconstruction error {max_error:.3g}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--run-id', required=True)
    audit(parser.parse_args().run_id)
