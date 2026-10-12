"""Recompute results from saved predictions and independently check fitted weights."""
from pathlib import Path
import argparse,json,sys
import numpy as np
import pandas as pd
from scipy.special import expit
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.data import load_dataset,digest
from src.evaluation import classification_metrics,group_metrics
from src.preprocessing import TabularEncoder
from src.contracts import class_quotas

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run-id',required=True);args=ap.parse_args()
    run=ROOT/'runs'/args.run_id
    config=json.loads((run/'config.json').read_text('utf-8'))
    records=[];summaries=[];checks=[];pooled_groups=[]
    for name in config['datasets']:
        data=load_dataset(name)
        plan=json.loads((run/name/'splits.json').read_text('utf-8'))
        assert data.sha256==plan['data_sha256']
        for method in config['methods']:
            predictions=[];method_records=[]
            for f in plan['folds']:
                fold=f['fold'];dest=run/name/f'fold-{fold}'/method['name']
                record=json.loads((dest/'record.json').read_text('utf-8'))
                assert record['status']=='completed'
                for filename,sha in record['artifact_sha256'].items():assert digest(dest/filename)==sha
                selection=json.loads((dest/'selection.json').read_text('utf-8'))
                encstate=json.loads((dest/'encoder.json').read_text('utf-8'))
                model=json.loads((dest/'model.json').read_text('utf-8'))
                predicted=pd.read_csv(dest/'predictions.csv',float_precision='round_trip')
                tr=np.array(f['train_positions']);te=np.array(f['test_positions'])
                chosen=np.array(selection['selected_positions'])
                assert len(chosen)==len(set(chosen))
                assert set(chosen)<=set(tr) and not set(chosen)&set(te)
                assert not set(data.group_ids[tr])&set(data.group_ids[te])
                assert predicted.row_id.tolist()==data.row_ids[te].tolist()
                assert predicted.target.tolist()==data.y[te].tolist()
                assert selection['selected_row_ids']==data.row_ids[chosen].tolist()
                assert model['fit_n_rows']==len(chosen)==encstate['fit_n_rows']==record['selected_n']
                quotas=class_quotas(tuple(map(int,data.y[tr])),method['rho'])
                assert {c:int(np.sum(data.y[chosen]==c)) for c in (0,1)}==quotas
                assert selection['original_features']==list(data.X.columns)
                if method['rho']==1:assert chosen.tolist()==tr.tolist()
                # Refit only preprocessing on the saved selected rows to detect full-train preprocessing leakage.
                encoder=TabularEncoder(data.numeric,data.categorical).fit(data.X.iloc[chosen])
                assert encoder.params_==encstate['parameters'] and encoder.blocks_==encstate['blocks']
                probabilities=expit(encoder.transform(data.X.iloc[te])@np.array(model['coef'])[0]+model['intercept'][0])
                np.testing.assert_allclose(probabilities,predicted.probability.to_numpy(),rtol=0,atol=1e-12)
                np.testing.assert_array_equal(predicted.prediction.to_numpy(),(probabilities>=config['threshold']).astype(int))
                recalculated=classification_metrics(predicted.target.to_numpy(),predicted.probability.to_numpy(),config['threshold'])
                for metric in ['f1','roc_auc','average_precision','accuracy','precision','recall','fpr']:
                    assert abs(recalculated[metric]-record['metrics'][metric])<1e-12
                assert all(record['metrics'][x]==recalculated[x] for x in ['tn','fp','fn','tp','n'])
                if method['rho']<1 and method['strategy']!='stratified_random':
                    clusters=selection['sampler_diagnostics']['clusters']
                    assert all(1<=c['quota']<=c['size'] for c in clusters)
                    assert sum(c['quota'] for c in clusters)==len(chosen)
                checks.append(f'{name}/fold-{fold}/{method["name"]}: checksums, identities, groups, quotas, preprocessing, weights and metrics PASS')
                flat={'dataset':name,'method':method['name'],'fold':fold,**record['metrics'],
                    **{k:record[k] for k in ['train_n','test_n','selected_n','actual_rho','original_d','encoded_d','sampler_seconds','fit_seconds','predict_seconds','total_seconds']}}
                records.append(flat);method_records.append(flat);predictions.append(predicted)
            pooled=pd.concat(predictions,ignore_index=True)
            assert pooled.row_id.is_unique and set(pooled.row_id)==set(data.row_ids)
            metric=classification_metrics(pooled.target.to_numpy(),pooled.probability.to_numpy(),config['threshold'])
            ordered=pooled.set_index('row_id').loc[data.row_ids]
            groups=group_metrics(data.audit,data.y,ordered.probability.to_numpy(),config['threshold'])
            pooled_groups.extend([{'dataset':name,'method':method['name'],**g} for g in groups])
            df=pd.DataFrame(method_records)
            summaries.append({'dataset':name,'method':method['name'],'folds':len(df),
                'f1_mean':float(df.f1.mean()),'f1_sd_across_dependent_folds':float(df.f1.std(ddof=1)),
                'auc_mean':float(df.roc_auc.mean()),'pooled_f1':metric['f1'],'pooled_auc':metric['roc_auc'],
                'mean_actual_rho':float(df.actual_rho.mean()),'median_pipeline_seconds':float(df.total_seconds.median()),
                'median_sampler_seconds':float(df.sampler_seconds.median()),'median_fit_seconds':float(df.fit_seconds.median()),
                'prediction_n':len(pooled)})
    df=pd.DataFrame(records);summary=pd.DataFrame(summaries)
    for i,row in summary.iterrows():
        current=df[(df.dataset==row.dataset)&(df.method==row.method)].set_index('fold')
        baseline=df[(df.dataset==row.dataset)&(df.method=='full')].set_index('fold')
        difference=current.f1-baseline.f1
        summary.loc[i,'mean_paired_f1_difference']=difference.mean()
        summary.loc[i,'folds_drop_gt_0_01']=int((difference < -.01-1e-12).sum())
    qa=run/'qa';qa.mkdir(exist_ok=True)
    df.to_csv(run/'fold_results.csv',index=False)
    summary.to_csv(run/'summary.csv',index=False)
    pd.DataFrame(pooled_groups).to_csv(run/'pooled_group_metrics.csv',index=False)
    (qa/'integrity_checks.json').write_text(json.dumps({'status':'passed','completed_fits':len(records),
        'checks':checks,'official_adult_test_used':False,'maximum_allowed_prediction_reconstruction_error':1e-12},indent=2),'utf-8')
    lines=['# 开发基线与样本选择结果','',
        '20次拟合；HTRU2一个数据集，每个使用共享的五折分组划分和四个预设条件。仅作开发验证，不是最终论文确认性实验。', '',
        '|数据|条件|平均F1|折间SD|合并折外F1|相对全量平均差|实际行率|流水线中位秒|',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for _,r in summary.iterrows():
        lines.append(f'|{r.dataset}|{r.method}|{r.f1_mean:.4f}|{r.f1_sd_across_dependent_folds:.4f}|{r.pooled_f1:.4f}|{r.mean_paired_f1_difference:+.4f}|{r.mean_actual_rho:.4f}|{r.median_pipeline_seconds:.3f}|')
    lines += ['', '## 解释范围', '',
        'F1是脉冲星正类F1。平均F1与合并折外F1口径不同，均保留。折间SD描述这次划分下的波动，不是均值标准误或置信区间。', '',
        '各折训练集重叠，各条件共享测试行。本轮未进行显著性检验、未提供非劣效证明、未选择最佳采样策略或调整参数。0.01仅作后续容差的描述参照；当前没有内层调参，不是三层嵌套流程验收。', '',
        'total_seconds为每次任务从采样前开始到预测完成的墙钟时间，含线程控制初始化、选择、预处理、模型拟合、预测；不含数据载入、划分、结果写盘和总体研发。时间为单次测量，固定条件顺序使第一项可能含预热开销，不能据此宣称普遍加速。', '',
        '每个拟合保存完整测试预测、实际训练行ID、类别配额、模型系数、预处理参数及编码块。审核逐项重算指标，并由保存系数和仅在选中训练行上重新拟合的预处理重建预测，误差上限1e-12。', '',
        '本轮仅使用HTRU2。各原始变量仍全部保留，特征选择和统一预算算法尚未接入。']
    (run/'summary.md').write_text('\n'.join(lines)+'\n','utf-8')
    print(summary[['dataset','method','f1_mean','pooled_f1','mean_paired_f1_difference','median_pipeline_seconds']].to_string(index=False))
    print(f'Integrity checks passed for {len(records)} fits.')

if __name__=='__main__':main()
