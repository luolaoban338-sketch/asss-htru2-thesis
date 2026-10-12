"""Require complete audited results; export descriptive stability and budget tables."""
from pathlib import Path
from collections import Counter
from fractions import Fraction
from itertools import combinations
import argparse
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from experiments.stability import stability_summary
from scripts.run_joint_pilot import read, cached, save
from src.data import digest


def summarize(run):
    cfg=read(run/'config.json'); joint=read(run/'joint_config.json')
    audit=read(ROOT/'qa'/f'{run.name}_audit.json')
    assert audit['status']=='passed' and audit['replicates']==cfg['replicates']
    assert audit['manifest_sha256']==digest(run/'manifest.json')
    results=[cached(run/f'replicate-{r}'/'result.json') for r in range(cfg['replicates'])]
    for result in results:
        assert result is not None
        folder=run/f'replicate-{result["replicate"]}'
        for rel,sha in result['artifact_sha256'].items(): assert digest(folder/rel)==sha
    features=read(run/'replicate-0/refit/full/selector.json')['features']
    summaries=[]; detailed={}; budgets=[]; pairs=[]; frequencies=[]
    for arm in joint['comparisons']:
        values=[r['selections'][arm] for r in results]
        feature_sets=[v['features'] for v in values]
        s=stability_summary(feature_sets,features)
        # Independent rational arithmetic using integer selection counts.
        n=len(values); d=len(features); counts=[sum(f in fs for fs in feature_sets) for f in features]
        q=Fraction(sum(counts),n*d)
        if q in (0,1): assert s['nogueira'] is None
        else:
            variance=sum(Fraction(c*(n-c),n*(n-1)) for c in counts)/d
            assert abs(s['nogueira']-float(1-variance/(q*(1-q))))<1e-12
        exact_pairs=[]
        for a,b in combinations(feature_sets,2):
            union=set(a)|set(b); exact_pairs.append(Fraction(len(set(a)&set(b)),len(union)) if union else Fraction(1))
        assert abs(s['jaccard_mean']-float(sum(exact_pairs)/len(exact_pairs)))<1e-12
        detailed[arm]=s
        methods=dict(Counter(v['candidate']['method'] for v in values))
        summaries.append({'arm':arm,'n_runs':n,'nogueira':s['nogueira'],'nogueira_status':s['nogueira_status'],
            'jaccard_mean':s['jaccard_mean'],'mean_k':s['mean_k'],'min_k':s['min_k'],'max_k':s['max_k'],
            'mean_actual_rho':float(np.mean([v['actual_rho'] for v in values])),
            'mean_cell_fraction':float(np.mean([v['cell_fraction'] for v in values])),
            'method_counts':methods})
        frequencies.extend({'arm':arm,'feature':f,'selected_count':c,'frequency':c/n} for f,c in zip(features,counts))
        pairs.extend({'arm':arm,**p} for p in s['jaccard_pairs'])
        for r,v in enumerate(values):
            c=v['candidate']
            budgets.append({'arm':arm,'replicate':r,'method':c['method'],'lambda':c['lam'],'eta':c['eta'],
                'k':c['k'],'features':'|'.join(v['features']),'selected_n':v['selected_n'],'resample_n':v['resample_n'],
                'actual_rho':v['actual_rho'],'cell_fraction':v['cell_fraction'],
                'selected_inner_mean_f1':c['mean_f1'],
                'inner_threshold':cached(run/f'replicate-{r}'/'decisions.json')['threshold']})
    for name,rows in [('stability_summary',summaries),('selected_budgets',budgets),('jaccard_pairs',pairs),('feature_frequencies',frequencies)]:
        pd.DataFrame(rows).to_csv(run/f'{name}.csv',index=False,encoding='utf-8-sig')
    report={'run_id':run.name,'summaries':summaries,'details':detailed,'audit':audit,
            'counts':{k:sum(r['counts'][k] for r in results) for k in results[0]['counts']},
            'timing':[r['timing'] for r in results],
            'source_result_sha256':{f'replicate-{r}/result.json':digest(run/f'replicate-{r}/result.json') for r in range(n)},
            'outer_test_evaluated':False,'confirmatory':False,'independent_metric_check':'integer counts and exact rational pair arithmetic passed'}
    save(run/'summary.json',report)
    lines=['# 完整选择流程稳定性开发验证','',f'运行：`{run.name}`。预定10次完整流程全部完成并通过审计。',
      '', '## 范围与口径','',
      '仅使用 HTRU2 第0外层训练池。每次无放回分层抽取80%行，重新完成三折内层预算搜索，并在整份重抽样上重建最终选择器。B=4、M=256及原参数网格保持固定。外层测试没有参与本轮评价；最终集合产生后不再拟合分类器。',
      '', '10次表示完整选择流程重复；每对方法共享同一组重抽样。数据扰动和算法随机性共同变化，不能分离解释各自影响。45个Jaccard对相互依赖，不是45个独立实验。本报告不作显著性检验或总体置信区间推断。',
      '', '## 稳定性与预算','', '| 条件 | Nogueira | 平均Jaccard | 平均k [范围] | 平均单元比例 |',
      '|---|---:|---:|---:|---:|']
    for row in summaries:
        ng='未定义（全选）' if row['nogueira'] is None else f'{row["nogueira"]:.4f}'
        lines.append(f'| {row["arm"]} | {ng} | {row["jaccard_mean"]:.4f} | {row["mean_k"]:.2f} [{row["min_k"]},{row["max_k"]}] | {row["mean_cell_fraction"]:.4f} |')
    joint_row=next(v for v in summaries if v['arm']=='joint')
    sequential=next(v for v in summaries if v['arm']=='sequential')
    lines+=['',f'联合选择的方法次数：{joint_row["method_counts"]}。逐次λ、η、k及原始特征见 selected_budgets.csv。',
       '', '全量和仅采样条件固定保留所有变量；Jaccard=1仅反映其定义。Nogueira分母为零，记为未定义，不能用来证明特征筛选有效或更稳定。单元比例以每次重抽样本身为分母，不把外部80%重抽样当成额外算法压缩。',
       '',f'联合与顺序的描述性Nogueira分别为 {joint_row["nogueira"]} 和 {sequential["nogueira"]}。这些数值受集合大小与当前八变量任务影响，不能据此声称跨数据集优势。',
       '', '## 计算与核验','',
       f'主运行计数：{report["counts"]}。独立直接效用核验另计 {audit["direct_utility_refits"]} 次逻辑回归拟合，重跑不计入主运行。',
       '', '所有保存的内层预测由编码器、系数与截距重建；审计检查行/组隔离、采样配额、贡献与冗余路径、参数候选及最终选择。稳定性指标另用整数计数和分数运算重算。逐重复耗时保存在summary.json，排除导入和run准备；若resumed=true，不得视为首次完整耗时。',
       '', '## 结论边界与后续','',
       '这是10次完整流程的开发稳定性验证，补足了仅重复贡献博弈的不足。它没有提供新的外层预测性能、污染鲁棒性或总体稳定性保证；外层预测结论见本轮联合验证报告。M=256仍是近似预算，本轮结果不证明其充分。',
       '', '下一步仍需先冻结训练侧参数污染协议，再进行联合流程污染重拟合、合成冗余和多数据集验证。不要按当前稳定性结果静默改写原协议。',
       '', '指标依据：[Nogueira, Sechidis and Brown (2018), On the Stability of Feature Selection Algorithms, JMLR 18(174), Eq. 2](https://jmlr.org/papers/volume18/17-514/17-514.pdf)。本项目显式将全选/全空的零分母情况记为未定义。','']
    (run/'summary.md').write_text('\n'.join(lines),'utf-8')
    (ROOT/'exports/完整选择流程稳定性验证_20261001.md').write_text('\n'.join(lines),'utf-8')
    print(pd.DataFrame(summaries).to_string(index=False))
    return report


if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--run-id',required=True); args=ap.parse_args()
    summarize(ROOT/'runs'/args.run_id)
