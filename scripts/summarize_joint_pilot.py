"""Descriptive summaries, with explicit limits on inference and timing."""
from pathlib import Path
import sys
import json
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score
from scripts.run_joint_pilot import read, save, cached
from experiments.feature_path import forward_path

RUN_ID = 'joint-htru2-pilot-20261001-v1'
run = ROOT/'runs'/RUN_ID
config = read(run/'config.json')
audit = read(ROOT/'qa'/f'{RUN_ID}_audit.json')
assert audit['status'] == 'passed' and audit['folds'] == len(config['outer_folds'])
rows = []; by_arm = {}; sources = []; comparisons = []
for fold in config['outer_folds']:
    folder = run/f'fold-{fold}'
    full = cached(folder/'outer/full.json')
    for arm in config['comparisons']:
        r = cached(folder/'outer'/f'{arm}.json'); model = cached(folder/r['model_path'])
        row = {'fold': fold, 'arm': arm, 'method': r['candidate']['method'],
               'k': len(r['features']), 'features': ';'.join(r['features']),
               'lambda': r['candidate']['lam'], 'eta': r['candidate']['eta'],
               'selected_n': r['selected_n'], 'train_n': r['train_n'], 'test_n': r['test_n'],
               'row_fraction': r['actual_rho'], 'cell_fraction': r['cell_fraction'],
               'f1': r['metrics']['f1'], 'auc': r['metrics']['roc_auc'],
               'delta_f1_vs_full': r['metrics']['f1']-full['metrics']['f1'],
               'outer_tolerance_met': r['metrics']['f1'] >= full['metrics']['f1']-config['epsilon']-config['numeric_tolerance'],
               'inner_f1': r['candidate']['mean_f1']}
        rows.append(row); by_arm.setdefault(arm, []).append((row,model))
        sources.append({'fold': fold, 'arm': arm, 'artifact': f'fold-{fold}/outer/{arm}.json', 'model_sha256': r['model_sha256']})
    joint = cached(folder/'outer/joint.json'); seq = cached(folder/'outer/sequential.json')
    comparisons.append({'fold': fold, 'joint_minus_sequential_cell_fraction': joint['cell_fraction']-seq['cell_fraction'],
                        'joint_minus_sequential_f1': joint['metrics']['f1']-seq['metrics']['f1'],
                        'same_candidate': joint['candidate']['candidate_id'] == seq['candidate']['candidate_id']})
summary = []
for arm, values in by_arm.items():
    records = [v[0] for v in values]; models = [v[1] for v in values]
    ids = sum([m['context']['evaluation_ids'] for m in models], [])
    assert len(ids) == len(set(ids)) == 17898
    y = np.concatenate([m['targets'] for m in models]); p = np.concatenate([m['probabilities'] for m in models])
    summary.append({'arm': arm, 'folds': len(records), 'pooled_n': len(y),
        'mean_f1': float(np.mean([r['f1'] for r in records])),
        'sd_f1': float(np.std([r['f1'] for r in records],ddof=1)),
        'mean_auc': float(np.mean([r['auc'] for r in records])),
        'pooled_f1': float(f1_score(y,p>=.5)), 'pooled_auc': float(roc_auc_score(y,p)),
        'mean_row_fraction': float(np.mean([r['row_fraction'] for r in records])),
        'mean_k': float(np.mean([r['k'] for r in records])),
        'mean_cell_fraction': float(np.mean([r['cell_fraction'] for r in records])),
        'mean_delta_f1_vs_full': float(np.mean([r['delta_f1_vs_full'] for r in records])),
        'outer_tolerance_violations': sum(not r['outer_tolerance_met'] for r in records)})
counts = dict(selectors=0, games=0, contribution_fits=0, constant_evaluations=0, scored_models=0,
              inner_parameter_candidates=0, exact_diagnostic_extra_fits=0)
timing = dict(selector_recorded_compute_seconds=0., model_fit_seconds=0., model_predict_seconds=0.)
errors = []; path_checks = []; hashes = set()
for path in sorted(run.glob('fold-*/**/selector.json')):
    s = cached(path); counts['selectors'] += 1; counts['contribution_fits'] += s['logistic_fits']
    counts['constant_evaluations'] += s['constant_evaluations']
    timing['selector_recorded_compute_seconds'] += s['selector_compute_seconds']
    games = [cached(path.parent/f'game-{b}.json') for b in range(config['B'])]
    exact = np.array([g['exact_phi'] for g in games])
    for g in games:
        counts['games'] += 1
        counts['exact_diagnostic_extra_fits'] += g['logistic_fits']-g['permutation_logistic_fits']
        errors.append({'selector': path.parent.relative_to(run).as_posix(), 'resample': g['resample'],
            'rmse': float(np.sqrt(np.mean((np.array(g['permutation']['phi'])-g['exact_phi'])**2))), 'max_abs_error':g['max_abs_error']})
    for p in s['paths']:
        exact_score = exact.mean(axis=0)-p['lam']*exact.std(axis=0,ddof=1)
        exact_path = forward_path(s['features'], exact_score, s['redundancy'], p['eta'])
        path_checks.append({'selector':path.parent.relative_to(run).as_posix(),'lambda':p['lam'],'eta':p['eta'],
            'full_order_agreement': [v['added'] for v in p['path']] == [v['added'] for v in exact_path],
            'matching_prefix_sets': sum(set(a['features']) == set(b['features']) for a,b in zip(p['path'],exact_path)),
            'prefix_sets':len(exact_path),'variance_clipped_features':sum(p['scores']['clipped_to_zero'])})
for path in run.glob('fold-*/**/models/mask-*.json'):
    if path.name.endswith('.complete.json'): continue
    m = cached(path); counts['scored_models'] += 1
    timing['model_fit_seconds'] += m['fit_seconds']; timing['model_predict_seconds'] += m['predict_seconds']
for path in run.glob('fold-*/inner-*/candidates.json'):
    counts['inner_parameter_candidates'] += len(cached(path))
counts['total_logistic_fits'] = counts['contribution_fits'] + counts['scored_models']
diagnostic = {'mean_game_rmse':float(np.mean([r['rmse'] for r in errors])),
    'max_game_abs_error':max(r['max_abs_error'] for r in errors),
    'full_path_order_agreements':sum(r['full_order_agreement'] for r in path_checks),
    'paths_compared':len(path_checks),'prefix_set_agreements':sum(r['matching_prefix_sets'] for r in path_checks),
    'prefix_sets_compared':sum(r['prefix_sets'] for r in path_checks),
    'interpretation':'Exact paths are a post-run diagnostic; no exact-path validation scores or budget reselection were used.'}
for name, records in [('outer_results',rows),('summary',summary),('joint_sequential',comparisons),('mc_diagnostics',errors),('path_diagnostics',path_checks)]:
    pd.DataFrame(records).to_csv(run/f'{name}.csv',index=False,float_format='%.17g')
save(run/'summary.json',{'counts':counts,'timing':timing,'timing_limit':'Recorded component sum, excluding path construction, file I/O and orchestration; not end-to-end wall-clock time.',
    'summary':summary,'diagnostic':diagnostic,'audit':audit,'sources':sources,'joint_sequential':comparisons})
names={'full':'全量','sample_only':'仅选样本','feature_only':'仅选特征','joint':'联合选择','sequential':'顺序选择',
       'joint_random':'随机候选联合','joint_ordinary':'普通几何候选联合','joint_robust':'稳健几何候选联合'}
lines=['# 共同预算与三层验证开发结果','',
    '2026年10月1日。HTRU2，17898行、八原始变量、五个固定外层分组折。结果仅作开发验证；原始折外数据此前已用于其他开发实验。','',
    '|条件|平均F1|折间SD|合并折外F1|平均AUC|平均行率|平均变量数|平均单元比例|外层容差违反折数|',
    '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
for r in summary:
    lines.append(f'|{names[r["arm"]]}|{r["mean_f1"]:.4f}|{r["sd_f1"]:.4f}|{r["pooled_f1"]:.4f}|{r["mean_auc"]:.4f}|{r["mean_row_fraction"]:.4f}|{r["mean_k"]:.1f}|{r["mean_cell_fraction"]:.4f}|{r["outer_tolerance_violations"]}/5|')
lines += ['', '所有决策使用内层平均F1≥全量基线−0.01；表中违反次数仅作外层事后检查，没有用于回选。五折SD描述划分变化，不是独立重复的标准误，也不报告显著性。','',
    '## 实际计算与近似诊断','',f'贡献拟合{counts["contribution_fits"]:,}次；候选与最终模型去重后{counts["scored_models"]:,}次；合计{counts["total_logistic_fits"]:,}次逻辑回归拟合，另有{counts["constant_evaluations"]:,}次常数评价。完整内层参数评价为{counts["inner_parameter_candidates"]:,}条。', '',
    f'M=256的{counts["games"]}个博弈平均RMSE={diagnostic["mean_game_rmse"]:.6f}，最大单变量绝对误差={diagnostic["max_game_abs_error"]:.6f}。近似与精确完整变量顺序一致{diagnostic["full_path_order_agreements"]}/{diagnostic["paths_compared"]}；各k前缀集合一致{diagnostic["prefix_set_agreements"]}/{diagnostic["prefix_sets_compared"]}。精确路径只作诊断，未用于改变预算。', '',
    f'已记录的选择器计算分项合计{timing["selector_recorded_compute_seconds"]:.1f}秒；模型拟合{timing["model_fit_seconds"]:.1f}秒、预测{timing["model_predict_seconds"]:.1f}秒。计时未覆盖全部路径构造、文件读写和调度，不能当作端到端耗时或各比较方法独立运行时间。', '',
    '## 证据文件','', 'outer_results.csv保留40条逐折比较；summary.csv为汇总；joint_sequential.csv为逐折决策比较；mc_diagnostics.csv与path_diagnostics.csv记录近似诊断。每折decisions.json在外层评价前保存。']
(run/'summary.md').write_text('\n'.join(lines)+'\n','utf-8')
print(json.dumps({'counts':counts,'summary':summary,'diagnostic':diagnostic},ensure_ascii=False,indent=2))
