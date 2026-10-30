"""Original-environment variable-k LR and separately implemented direct refit."""
from .common import *
import argparse,csv,warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import f1_score,roc_auc_score,average_precision_score
from a3.selection import NumericEncoder,SklearnBackend
from audit_a3_20261024.independent import IndependentData,IndependentBackend,independent_metrics,assert_close,fitted_state,independent_seed
def validate_selection(sel,result,spec,fold,binding):
    cols=sel['selected_indices']
    assert isinstance(cols,list) and all(type(i)is int for i in cols)
    assert cols==sorted(set(cols)) and set(cols)<=set(range(8)),'Invalid native indices'
    assert sel['native_complete'] is True and sel['statistics_valid'] is True
    assert result['native_complete'] is True and result['selected_indices']==cols
    assert sel['outer_fold']==fold and sel['ordered_feature_names']==spec['data']['predictors']
    assert sel['data_sha256']==spec['data']['data_source']['sha256'] and sel['split_sha256']==spec['data']['outer_partitions']['sha256']
    assert sel['train_row_ids_sha256']==row_sha(binding['row_ids'])
    assert sel['execution_manifest_sha256']==sha(HERE/'execution_manifest.json')
    assert sel['plan_spec_sha256']==sha(HERE/'frozen_spec.json') and sel['plan_id']==spec['plan_id']
    assert sel['native_environment_sha256']==sha(spec['environments']['native_selector']['record'])
    assert sel['actual_iterations']==result['iterations'] and 10<=result['iterations']<=40
    return cols
def main_data(spec,fold):
    c=spec['data']
    for k in ['data_source','outer_partitions']:assert sha(c[k]['path'])==c[k]['sha256']
    splits=read(c['outer_partitions']['path']);s=splits['folds'][fold]
    assert s['fold']==fold and splits['data_sha256']==c['data_source']['sha256']
    with open(c['data_source']['path'],encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
    X=np.asarray([[float(row[k]) for k in c['predictors']] for row in rows],dtype=float)
    y=np.asarray([int(r['target']) for r in rows],dtype=int)
    ids=np.asarray([r['row_id'] for r in rows]);groups=[r['group_id'] for r in rows]
    assert X.shape==(17898,8) and np.isfinite(X).all() and y.sum()==1639
    assert len(set(ids))==len(set(groups))==len(rows)
    tr=np.asarray(s['train_positions']);te=np.asarray(s['test_positions'])
    assert not set(tr)&set(te) and sorted(np.r_[tr,te])==list(range(len(rows)))
    assert ids[tr].tolist()==s['train_row_ids'] and ids[te].tolist()==s['test_row_ids']
    assert len(tr)==c['folds'][fold]['train_n'] and len(te)==c['folds'][fold]['test_n']
    return X[tr],y[tr],ids[tr],X[te],y[te],ids[te]
def model_state(enc,model):
    return {'encoder':{'center_':enc.center_.tolist(),'scale_':enc.scale_.tolist()},'model':{**{a:np.asarray(getattr(model,a)).tolist() for a in ['coef_','intercept_','classes_','n_iter_','n_features_in_']},'params':model.get_params(deep=False)}}
def evaluate(fold,replay,do_audit,quota):
    assert fold in range(5) and (not replay or fold==0)
    unit=f'audit_final_f{fold}' if do_audit else f'downstream_{"replay" if replay else "main"}_f{fold}'
    if cached(unit,'downstream') is not None:return
    spec=read(HERE/'frozen_spec.json');native_unit=f'native_{"replay" if replay else "main"}_f{fold}'
    base=RUN/native_unit;nr=read(base/'result.json')
    for rel,h in nr['payload_hashes'].items():assert sha(base/rel)==h
    selection=read(base/'selection.json');binding=read(base/'training_binding.json')
    cols=validate_selection(selection,nr,spec,fold,binding)
    seed=spec['downstream']['final_model_seeds'][fold];assert seed==independent_seed(fold,'final_model')
    with unit_context(unit,quota,'downstream') as ctx:
        # The immutable selection is copied before any held-out data is loaded.
        handoff={**selection,'selector_result_sha256':sha(base/'result.json'),'native_unit':native_unit}
        write(ctx.out/'handoff.json',handoff)
        if do_audit:
            d=IndependentData(ROOT);x,y,ids=d.training(fold);xt,yt,testids=d.heldout(fold)
        else:x,y,ids,xt,yt,testids=main_data(spec,fold)
        assert ids.tolist()==binding['row_ids'] and array_sha(x)==binding['x_sha256'] and array_sha(y)==binding['y_sha256']
        enc=model=None;t=time.perf_counter()
        if cols:
            backend=IndependentBackend() if do_audit else SklearnBackend()
            enc=ctx.fit('encoder',seed,lambda:backend.fit_encoder(x[:,cols]))
            model=ctx.fit('lr',seed,lambda:backend.fit_model(enc.transform(x[:,cols]),y,'lr',seed,C=1.0))
            final_seconds=time.perf_counter()-t;t=time.perf_counter()
            p=model.predict_proba(enc.transform(xt[:,cols]))[:,list(model.classes_).index(1)]
            prediction_seconds=time.perf_counter()-t
            state={'encoder':fitted_state(enc),'model':fitted_state(model)} if do_audit else model_state(enc,model)
            import joblib
            joblib.dump({'encoder':enc,'model':model},ctx.out/'fitted_model.joblib')
        else:
            prevalence=float(np.mean(y));p=np.full(len(yt),prevalence);state={'constant_probability':prevalence}
            final_seconds=0.;prediction_seconds=time.perf_counter()-t
        assert np.isfinite(p).all() and np.all((0<=p)&(p<=1))
        metrics=independent_metrics(yt,p) if do_audit else {'f1':float(f1_score(yt,p>=.5,zero_division=0)),'auc':float(roc_auc_score(yt,p)),'ap':float(average_precision_score(yt,p))}
        prediction=dict(outer_fold=fold,selected_indices=cols,features=[spec['data']['predictors'][i] for i in cols],actual_k=len(cols),train_row_ids_sha256=row_sha(ids.tolist()),test_row_ids=testids.tolist(),labels=yt.tolist(),probabilities=p.tolist(),metrics=metrics,final_seed=seed,fitted_state=state)
        if do_audit:
            reference=read(RUN/f'downstream_main_f{fold}'/'predictions.json')
            assert_close(prediction,reference)
            if fold==0:assert_close(read(RUN/'downstream_replay_f0/predictions.json'),reference)
        write(ctx.out/'predictions.json',prediction)
        result=dict(outer_fold=fold,actual_k=len(cols),selected_indices=cols,metrics=metrics,encoder_fits=int(bool(cols)),classifier_fits=int(bool(cols)),outer_test_predictions=len(yt),final_seconds=final_seconds,prediction_seconds=prediction_seconds,independent_direct_refit=do_audit,semantic_reference_match=do_audit,source_native_result_sha256=sha(base/'result.json'))
        ctx.result(result)
        print(json.dumps({'unit':unit,**result}),flush=True)
def main():
    p=argparse.ArgumentParser();p.add_argument('--fold',type=int,required=True);p.add_argument('--replay',action='store_true');p.add_argument('--audit',action='store_true');p.add_argument('--quota',type=Path,required=True)
    a=p.parse_args();assert not(a.replay and a.audit);evaluate(a.fold,a.replay,a.audit,a.quota)
if __name__=='__main__':main()
