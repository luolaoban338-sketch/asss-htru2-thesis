"""Source tables with explicit coverage; complete-fold aggregation only."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,csv,json,sqlite3,sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from experiments.stability import stability_summary
from scripts.run_joint_pilot import read,save,cached
from scripts.run_ablation import MAIN,package_path,DESIGN,QA
from src.data import digest

def write_csv(path,rows):
    if not rows: return
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def summarize(output):
    output.mkdir(parents=True,exist_ok=False)
    cfg=read(DESIGN/'config.json');run=ROOT/'runs'/MAIN
    predictive=[];stability=[];selection=[];diagnostics=[];audits=[]
    for p in sorted(run.glob('*/unit-*/package.json')):
        audit=cached(p.parent/'audit.json')
        if not audit or audit['status']!='passed': continue
        for rel,h in audit['sha256'].items(): assert digest(p.parent/rel)==h
        package=cached(p);audits.append({'package':p.parent.relative_to(run).as_posix(),'phase':package['phase'],
            'unit':package['unit'],'root':package['root'],'audit_sha256':digest(p.parent/'audit.json'),
            'games':audit['games'],'reconstructed_final_models':audit['reconstructed_final_models'],'direct_utility_refits':audit['direct_utility_refits']})
        for r in package['records']:
            row={'phase':package['phase'],'unit':package['unit'],'root':package['root'],
                'sampler':r['sampler'],'label':r['label'],'features':'|'.join(r['features']),
                'selected_n':r['selected_n'],'train_n':r['train_n'],'actual_rho':r['actual_rho'],'cell_fraction':r['cell_fraction']}
            if package['phase']=='predictive': row.update(r['metrics']);predictive.append(row)
            else: stability.append(row)
        for sampler in cfg['samplers']:
            s=cached(p.parent/sampler/'selector.json')
            for label in cfg['logical_csr']:
                ref=s['variants']['111'];cur=s['variants'][label]
                union=set(ref['features'])|set(cur['features']);inter=set(ref['features'])&set(cur['features'])
                selection.append({'phase':package['phase'],'unit':package['unit'],'root':package['root'],'sampler':sampler,'label':label,
                    'selected_set_changed':set(ref['features'])!=set(cur['features']),
                    'path_equal_111':ref['path'][-1]['features']==cur['path'][-1]['features'],
                    'selected_set_jaccard_111':len(inter)/len(union),'exact_prefix_agreement':s['exact_prefix_agreement'][label],
                    'clipped_variance_features':sum(s['statistics']['clipped_to_zero'])})
            for b in range(cfg['B']):
                g=cached(p.parent/sampler/f'game-{b}.json')
                diagnostics.append({'phase':package['phase'],'unit':package['unit'],'root':package['root'],'sampler':sampler,'b':b,
                    'rmse':g['rmse'],'max_abs_error':g['max_abs_error'],'logistic_fits':g['logistic_fits'],
                    'constant_evaluations':g['constant_evaluations'],'seconds':g['seconds']})
    expected_roots=set(cfg['predictive_root_seeds'])
    complete_folds=[f for f in cfg['outer_folds'] if {r['root'] for r in predictive if r['unit']==f}==expected_roots]
    primary=[];unit_contrasts=[];means=[]
    for sampler in cfg['samplers']:
        for enabled,disabled in cfg['primary_contrasts']+[['111','random_four']]:
            differences=[]
            for unit in sorted({(r['unit'],r['root']) for r in predictive}):
                rows=[r for r in predictive if (r['unit'],r['root'])==unit and r['sampler']==sampler]
                a=next(r for r in rows if r['label']==enabled);b=next(r for r in rows if r['label']==disabled)
                delta=a['f1']-b['f1']
                unit_contrasts.append({'fold':unit[0],'root':unit[1],'sampler':sampler,'contrast':enabled+'-'+disabled,'f1_difference':delta})
            for f in complete_folds:
                vals=[r['f1_difference'] for r in unit_contrasts if r['fold']==f and r['sampler']==sampler and r['contrast']==enabled+'-'+disabled]
                assert len(vals)==3; differences.append(float(np.mean(vals)))
            if len(complete_folds)==5:
                primary.append({'sampler':sampler,'contrast':enabled+'-'+disabled,
                    **{f'fold_{f}':differences[f] for f in range(5)},'mean':float(np.mean(differences)),
                    'descriptive_sd':float(np.std(differences,ddof=1)),'min':min(differences),'max':max(differences),
                    'positive':sum(x>1e-12 for x in differences),'zero':sum(abs(x)<=1e-12 for x in differences),'negative':sum(x < -1e-12 for x in differences)})
    if len(complete_folds)==5:
        for sampler,labels in [(s,cfg['logical_csr']+['random_four','half_all']) for s in cfg['samplers']]+[('full',['full_all'])]:
            for label in labels:
                rows=[r for r in predictive if r['sampler']==sampler and r['label']==label]
                assert len(rows)==15
                item={'sampler':sampler,'label':label}
                for metric in ('f1','roc_auc','average_precision','precision','recall','cell_fraction'):
                    values=[float(np.mean([r[metric] for r in rows if r['unit']==f])) for f in range(5)]
                    item['mean_'+metric]=float(np.mean(values))
                    if metric=='f1': item['sd_f1']=float(np.std(values,ddof=1))
                warning=0
                for r in rows:
                    full=next(x for x in predictive if x['unit']==r['unit'] and x['root']==r['root'] and x['sampler']=='full')
                    warning+=full['f1']-r['f1']>cfg['descriptive_f1_loss_warning']
                item['descriptive_warning_units']=warning;means.append(item)
    stable_summary=[]
    complete_stability=sorted({r['unit'] for r in stability})==list(range(10))
    if complete_stability:
        fs=read(run/'predictive/unit-0-seed-2026100501/random_half/selector.json')['features']
        for sampler in cfg['samplers']:
            for label in cfg['logical_csr']+['random_four']:
                rows=sorted([r for r in stability if r['sampler']==sampler and r['label']==label],key=lambda r:r['unit'])
                assert len(rows)==10
                summary=stability_summary([r['features'].split('|') for r in rows],fs)
                stable_summary.append({'sampler':sampler,'label':label,**summary})
        for r in stable_summary:
            ref=next(x for x in stable_summary if x['sampler']==r['sampler'] and x['label']=='111')
            r['nogueira_difference_from_111']=r['nogueira']-ref['nogueira']
            r['jaccard_difference_from_111']=r['jaccard_mean']-ref['jaccard_mean']
    for name,rows in [('predictive_records',predictive),('stability_selections',stability),('module_selection_diagnostics',selection),
        ('game_diagnostics',diagnostics),('per_seed_contrasts',unit_contrasts),('primary_contrasts',primary),('predictive_summary',means)]: write_csv(output/f'{name}.csv',rows)
    save(output/'stability_summary.json',stable_summary)
    db=sqlite3.connect(f'file:{(run/"fit_budget.sqlite3").as_posix()}?mode=ro',uri=True)
    counts=[{'kind':k,'status':s,'attempts':n,'seconds':t} for k,s,n,t in db.execute('SELECT kind,status,COUNT(*),SUM(seconds) FROM operations GROUP BY kind,status')];db.close()
    result={'created_utc':datetime.now(timezone.utc).isoformat(),'status':'complete_main_schedule' if len(predictive)==465 and len(stability)==270 else 'partial_audited_snapshot',
        'run_id':MAIN,'audits':audits,'predictive_records':len(predictive),'stability_selections':len(stability),
        'complete_predictive_folds':complete_folds,'complete_stability':complete_stability,
        'main_games_audited':len(diagnostics),'operation_counts_all_stage':counts,
        'five_fold_aggregation_emitted':bool(primary),'delivery_final':False,
        'inference':'Descriptive paired results only. Seeds averaged within folds, then five folds equally. No superiority p-values, confidence intervals or noninferiority claims.',
        'source_sha256':{p.name:digest(p) for p in output.iterdir() if p.is_file()}}
    save(output/'summary.json',result);print(json.dumps({k:result[k] for k in ('status','predictive_records','stability_selections','main_games_audited')}))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();summarize(a.output)
