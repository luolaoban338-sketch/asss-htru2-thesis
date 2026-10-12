"""Bounded, resumable development validation; one durable artifact per fit."""
from pathlib import Path
import argparse
from datetime import datetime,timezone
import hashlib
import importlib.metadata
import json
import platform
import os
import shutil
import sys
import time
import traceback
import warnings

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
# Explicit one-core job budget also avoids unsupported Windows WMIC detection.
os.environ['LOKY_MAX_CPU_COUNT']='1'
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits,threadpool_info
from src.data import load_dataset,digest,seed_for
from src.contracts import assert_disjoint
from src.samplers import select_rows
from src.preprocessing import TabularEncoder
from src.evaluation import classification_metrics,group_metrics

def write_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.partial')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),'utf-8')
    tmp.replace(path)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--run-id',required=True)
    ap.add_argument('--dataset',choices=['htru2'],required=True)
    ap.add_argument('--max-jobs',type=int,default=20)
    args=ap.parse_args()
    if not args.run_id.replace('-','').replace('_','').isalnum():raise ValueError('Invalid run ID')
    if args.max_jobs<1:raise ValueError('Positive max-jobs required')
    config_path=ROOT/'configs/baseline_pilot_v1.json'
    config=json.loads(config_path.read_text('utf-8'))
    run=ROOT/'runs'/args.run_id
    code_paths=sorted((ROOT/'src').glob('*.py'))+[Path(__file__).resolve()]
    code={str(p.relative_to(ROOT)).replace('\\','/'):digest(p) for p in code_paths}
    if (run/'manifest.json').exists():
        manifest=json.loads((run/'manifest.json').read_text('utf-8'))
        if manifest['code_sha256']!=code or manifest['config_sha256']!=digest(config_path):
            raise RuntimeError('Code/config changed: use a new run ID, never blend versions')
    else:
        run.mkdir(parents=True,exist_ok=False)
        for p in code_paths:
            dest=run/'code'/p.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
        shutil.copy2(config_path,run/'config.json')
        manifest={'created_at_utc':datetime.now(timezone.utc).isoformat(),'purpose':config['purpose'],
            'config_sha256':digest(config_path),'code_sha256':code,'python':sys.version,
            'platform':platform.platform(),'packages':{p:importlib.metadata.version(p) for p in ['numpy','pandas','scipy','scikit-learn','joblib','threadpoolctl']},
            'raw_manifest':json.loads((ROOT/'data/raw/manifest.json').read_text('utf-8')),
            'official_adult_test_used':False}
        write_json(run/'manifest.json',manifest)
    data=load_dataset(args.dataset)
    split_seed=seed_for(config['seed_root'],data.name,'development_outer_split')
    splitter=StratifiedGroupKFold(n_splits=config['split']['n_splits'],shuffle=config['split']['shuffle'],random_state=split_seed)
    splits=list(splitter.split(data.X,data.y,data.group_ids))
    plan={'dataset':data.name,'data_sha256':data.sha256,'split_seed':split_seed,'folds':[]}
    for fold,(itr,ite) in enumerate(splits):
        train=data.partition(itr,f'{data.name}:outer{fold}')
        assert_disjoint(train,data.row_ids[ite],data.group_ids[ite])
        if len(set(data.y[ite]))!=2:raise ValueError('One-class test fold')
        plan['folds'].append({'fold':fold,'train_positions':itr.tolist(),'test_positions':ite.tolist(),
            'train_row_ids':data.row_ids[itr].tolist(),'test_row_ids':data.row_ids[ite].tolist()})
    plan_path=run/data.name/'splits.json'
    if plan_path.exists() and json.loads(plan_path.read_text('utf-8'))!=plan:raise ValueError('Split plan changed')
    write_json(plan_path,plan)
    jobs=0
    for fold,(itr,ite) in enumerate(splits):
        train=data.partition(itr,f'{data.name}:outer{fold}')
        for method in config['methods']:
            dest=run/data.name/f'fold-{fold}'/method['name']
            if (dest/'record.json').exists():
                old=json.loads((dest/'record.json').read_text('utf-8'))
                if old['status']=='completed':
                    for name,sha in old['artifact_sha256'].items():
                        if digest(dest/name)!=sha:raise ValueError('Completed artifact was changed')
                    continue
                raise RuntimeError(f'Previous failure needs review: {dest}')
            if jobs>=args.max_jobs:return
            dest.mkdir(parents=True,exist_ok=True)
            t0=time.perf_counter()
            try:
                # Shared sampler seed for geometry contrast; classifier seed also paired.
                sample_seed=seed_for(config['seed_root'],data.name,fold,'sampler')
                model_seed=seed_for(config['seed_root'],data.name,fold,'classifier')
                with threadpool_limits(limits=config['thread_limit']):
                    t=time.perf_counter()
                    local,res,diagnostics=select_rows(data.X.iloc[itr],train,data.numeric,data.categorical,
                        method['rho'],sample_seed,method['strategy'],config['sampler'])
                    sampler_seconds=time.perf_counter()-t
                    selected=itr[local]
                    if set(data.row_ids[selected])&set(data.row_ids[ite]):raise ValueError('Selected/test overlap')
                    t=time.perf_counter()
                    enc=TabularEncoder(data.numeric,data.categorical)
                    Z=enc.fit_transform(data.X.iloc[selected])
                    params={k:v for k,v in config['model'].items() if k!='class'}
                    model=LogisticRegression(**params,random_state=model_seed)
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter('always')
                        model.fit(Z,data.y[selected])
                    warning_list=[{'type':w.category.__name__,'message':str(w.message)} for w in caught]
                    if any(issubclass(w.category,ConvergenceWarning) for w in caught):
                        raise RuntimeError('Classifier convergence warning; do not accept results')
                    fit_seconds=time.perf_counter()-t
                    t=time.perf_counter()
                    probabilities=model.predict_proba(enc.transform(data.X.iloc[ite]))[:,list(model.classes_).index(1)]
                    predict_seconds=time.perf_counter()-t
                    measured_total=time.perf_counter()-t0
                    pools=threadpool_info()
                metric=classification_metrics(data.y[ite],probabilities,config['threshold'])
                prediction=pd.DataFrame({'row_id':data.row_ids[ite],'group_id':data.group_ids[ite],
                    'target':data.y[ite],'probability':probabilities,'prediction':(probabilities>=config['threshold']).astype(int)})
                prediction.to_csv(dest/'predictions.csv',index=False,float_format='%.17g')
                write_json(dest/'selection.json',{'selected_row_ids':res.row_ids,'selected_positions':selected.tolist(),
                    'original_features':list(data.X.columns),'class_counts':res.class_counts,'requested_rho':method['rho'],
                    'actual_rho':len(selected)/len(itr),'sampler_diagnostics':diagnostics})
                write_json(dest/'encoder.json',enc.diagnostics(data.X.iloc[ite]))
                write_json(dest/'model.json',{'type':'LogisticRegression','parameters':model.get_params(),
                    'coef':model.coef_.tolist(),'intercept':model.intercept_.tolist(),'classes':model.classes_.tolist(),
                    'n_iter':model.n_iter_.tolist(),'fit_n_rows':len(selected)})
                groups=group_metrics(data.audit.iloc[ite],data.y[ite],probabilities)
                write_json(dest/'group_metrics.json',groups)
                counts_before={str(c):int(np.sum(data.y[itr]==c)) for c in (0,1)}
                audit_counts={c:{'before':{str(k):int(v) for k,v in data.audit.iloc[itr][c].value_counts(dropna=False).items()},
                    'after':{str(k):int(v) for k,v in data.audit.iloc[selected][c].value_counts(dropna=False).items()}} for c in data.audit.columns}
                record={'status':'completed','dataset':data.name,'method':method['name'],'fold':fold,'metrics':metric,
                    'train_n':len(itr),'test_n':len(ite),'selected_n':len(selected),'actual_rho':len(selected)/len(itr),
                    'original_d':len(data.X.columns),'selected_d':len(data.X.columns),'encoded_d':Z.shape[1],
                    'class_counts_before':counts_before,'class_counts_after':res.class_counts,'audit_counts':audit_counts,
                    'seeds':{'split':split_seed,'sampler':sample_seed,'classifier':model_seed},
                    'sampler_seconds':sampler_seconds,'fit_seconds':fit_seconds,'predict_seconds':predict_seconds,
                    'total_seconds':measured_total,'wall_including_artifacts_seconds':time.perf_counter()-t0,
                    'model_fit_count':1,'cache_hits':0,'warnings':warning_list,'threadpools':pools,
                    'artifact_sha256':{p.name:digest(p) for p in dest.iterdir() if p.is_file()},
                    'completed_at_utc':datetime.now(timezone.utc).isoformat()}
                write_json(dest/'record.json',record)
                print(f"{data.name} fold={fold} {method['name']} F1={metric['f1']:.5f} rows={len(selected)}/{len(itr)} seconds={measured_total:.3f}",flush=True)
                jobs+=1
            except Exception as exc:
                write_json(dest/'record.json',{'status':'failed','error':str(exc),'traceback':traceback.format_exc()})
                raise

if __name__=='__main__':main()
