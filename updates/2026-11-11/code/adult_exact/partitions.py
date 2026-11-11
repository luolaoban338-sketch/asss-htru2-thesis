"""No-fit Adult membership preparation and immutable read APIs.

Only identities, labels, group IDs, deterministic random draws and grouped
splitters are used. No estimator, encoder, geometry or prediction is executed.
Large artifacts may only be written to the exact configured F data directory.
"""
from pathlib import Path
from itertools import combinations
from math import ceil
import csv, hashlib, json, platform, sys

sys.dont_write_bytecode = True
STAGE = Path(__file__).resolve().parent
PROJECT = STAGE.parent
DATA_ROOT = Path('F:/ChatGPT/论文大数据/datasets/adult_reduced_20261107')
METHODS = ('full', 'robust_half')

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()

def canonical(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',', ':'),ensure_ascii=True).encode('utf8')).hexdigest()

def seed_for(root,*parts):
    return int.from_bytes(hashlib.sha256('|'.join(map(str,(root,)+parts)).encode('utf8')).digest()[:4],'big')

def checked_paths():
    paths=read(STAGE/'paths.json')
    if Path(paths['data_root']).resolve()!=DATA_ROOT.resolve() or DATA_ROOT.drive.upper()!='F:':
        raise ValueError('Exact declared F data root required; no C fallback')
    for key,name in [('partitions_file','partitions.json'),('games_file','games.json'),('timing_tasks_file','timing_tasks.json')]:
        if Path(paths[key]).resolve()!= (DATA_ROOT/name).resolve():raise ValueError('Unexpected output target '+key)
    return paths

def identity_source():
    import numpy as np
    paths=checked_paths();old=Path(paths['source_root'])
    binding=read(old/'data/source_manifest.json')
    for rel,expected in binding['assets'].items():
        if rel in ['audit/adult_train_table.csv','audit/adult_train_dictionary.json','audit/audit_report.json']:
            if sha(old/'data'/rel)!=expected:raise ValueError('Source hash mismatch '+rel)
    report=read(old/'data/audit/audit_report.json')
    fields=report['datasets']['adult_train']['fields']
    features=[r['field'] for r in fields if r['role']=='predictor']
    numeric=[r['field'] for r in fields if r['role']=='predictor' and r['type']=='numeric']
    categorical=[f for f in features if f not in numeric]
    if len(features)!=12 or len(numeric)!=6 or set(features)&{'race','sex','target'}:raise ValueError('Adult schema mismatch')
    table=old/'data/audit/adult_train_table.csv'
    with table.open(encoding='utf-8-sig',newline='') as f:
        identity=[(r['row_id'],r['group_id'],int(r['target'])) for r in csv.DictReader(f)]
    if len(identity)!=32561 or len({x[0] for x in identity})!=32561:raise ValueError('Adult identity mismatch')
    data={'row_ids':np.array([x[0] for x in identity],dtype=str),
          'group_ids':np.array([x[1] for x in identity],dtype=str),
          'y':np.array([x[2] for x in identity],dtype=int),
          'features':features,'numeric':numeric,'categorical':categorical,
          'source_table_sha256':sha(table),'source_manifest_sha256':sha(old/'data/source_manifest.json')}
    freeze=read(old/'protocols/freeze_manifest.json')
    oldpart=old/'protocols/partitions/adult_train_partitions.json'
    if sha(oldpart)!=freeze['files']['protocols/partitions/adult_train_partitions.json']:
        raise ValueError('Old outer partition changed')
    return data,read(oldpart),old

def segment(data,positions,include_ids=True):
    import numpy as np
    pos=sorted(map(int,positions))
    if len(pos)!=len(set(pos)) or not pos:raise ValueError('Empty/duplicate positions')
    ids=data['row_ids'][pos].tolist();groups=data['group_ids'][pos].tolist();labels=data['y'][pos].tolist()
    result={'positions':pos,'n_rows':len(pos),'n_groups':len(set(groups)),
            'class_counts':{str(c):int(np.sum(data['y'][pos]==c)) for c in (0,1)},
            'positions_sha256':canonical(pos),'row_ids_sha256':canonical(ids),
            'group_ids_sha256':canonical(sorted(set(groups))),
            'ordered_group_ids_sha256':canonical(groups),'labels_sha256':canonical(labels),
            'partition_sha256':canonical({'positions':pos,'row_ids':ids,'group_ids':groups,'labels':labels})}
    if include_ids:result['row_ids']=ids
    if not all(result['class_counts'].values()):raise ValueError('Single-class partition')
    return result

def hamilton(data,positions,target):
    import numpy as np
    counts=np.bincount(data['y'][positions],minlength=2);n=len(positions)
    quotas=target*counts//n
    order=sorted((0,1),key=lambda c:(-int(target*counts[c]%n),c))
    for c in order[:target-int(quotas.sum())]:quotas[c]+=1
    if min(quotas)<2:raise ValueError('Infeasible class quota')
    return quotas

def complete_group_pool(data,available,cap,seed):
    import numpy as np
    available=sorted(map(int,available));quotas=hamilton(data,available,cap)
    groups={}
    for p in available:groups.setdefault(str(data['group_ids'][p]),[]).append(p)
    ranked=sorted(groups,key=lambda g:(canonical([seed,g]),g))
    actual=np.zeros(2,dtype=int);selected=[]
    for group in ranked:
        members=groups[group];delta=np.bincount(data['y'][members],minlength=2)
        if np.all(actual+delta<=quotas):selected.extend(members);actual+=delta
    if not np.array_equal(actual,quotas) or len(selected)!=cap:
        raise ValueError('Whole-group Hamilton quotas unfillable; no seed/cap substitution')
    return sorted(selected),{str(c):int(quotas[c]) for c in (0,1)}

def complete_group_prediction_load(data,available,target,seed):
    groups={}
    for p in sorted(map(int,available)):groups.setdefault(str(data['group_ids'][p]),[]).append(p)
    selected=[]
    for group in sorted(groups,key=lambda g:(canonical([seed,g]),g)):
        selected.extend(groups[group])
        if len(selected)>=target:break
    if len(selected)<target:raise ValueError('Insufficient training-only timing prediction rows')
    return sorted(selected)

def random_resample(data,positions,rho,seed):
    """Same row-level class-stratified random resample as the frozen method.

    Unselected rows are not held-out observations; grouped utility partitions
    below prevent any selected group from spanning training and validation.
    """
    import numpy as np
    positions=np.asarray(positions,dtype=int);quotas=hamilton(data,positions,ceil(rho*len(positions)))
    selected=[]
    for c in (0,1):
        local=positions[data['y'][positions]==c]
        local=local[np.argsort(data['row_ids'][local],kind='stable')]
        rng=np.random.default_rng(seed_for(seed,'class',c))
        selected.extend(rng.choice(local,int(quotas[c]),replace=False).tolist())
    return sorted(selected)

def grouped_splits(data,positions,seed):
    import numpy as np
    from sklearn.model_selection import StratifiedGroupKFold
    positions=np.array(positions,dtype=int)
    splitter=StratifiedGroupKFold(n_splits=3,shuffle=True,random_state=seed)
    result=[]
    for i,(tr,va) in enumerate(splitter.split(np.zeros(len(positions)),data['y'][positions],data['group_ids'][positions])):
        result.append({'fold':i,'train':segment(data,positions[tr]),'validation':segment(data,positions[va])})
    return result

def timing_masks(data,root):
    d=len(data['features']);numerics={data['features'].index(f) for f in data['numeric']}
    cat=set(range(d))-numerics
    masks=[1<<i for i in range(d)]+[sum(1<<i for i in numerics),sum(1<<i for i in cat)]
    for k in (2,4,6,8,10,11):
        candidates=[sum(1<<i for i in comb) for comb in combinations(range(d),k) if set(comb)&numerics and set(comb)&cat]
        chosen=min((m for m in candidates if m not in masks),key=lambda m:(canonical([seed_for(root,'timing_masks',k),k,m]),m))
        masks.append(chosen)
    masks.append((1<<d)-1)
    if len(set(masks))!=21:raise ValueError('Expected exactly 21 masks')
    return masks

def build():
    import numpy as np
    import pandas as pd
    import sklearn
    config=read(STAGE/'config.json');paths=checked_paths();data,oldpart,old=identity_source()
    actual={'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'sklearn':sklearn.__version__}
    if any(config['expected_environment'][k]!=v for k,v in actual.items()):raise ValueError('Scientific split environment mismatch')
    if [config[k] for k in ['B','M','inner_folds','utility_folds','development_pool_cap_per_outer_fold']]!=[4,64,3,3,8000]:
        raise ValueError('This preparation implements only the accepted fixed design')
    root=config['seed_root'];contexts={};games_by_context={};outer=[];tasks=[];masks=timing_masks(data,root)
    for oldfold in oldpart['outer_folds']:
        f=oldfold['fold'];available=oldfold['train']['positions'];test=oldfold['test']['positions']
        pool_seed=seed_for(root,'adult_reduced',f,'development_pool')
        pool,quota=complete_group_pool(data,available,8000,pool_seed)
        excluded=sorted(set(available)-set(pool))
        inner_seed=seed_for(root,'adult_reduced',f,'inner_split')
        inners=grouped_splits(data,pool,inner_seed)
        pids=[f'adult_reduced:outer{f}:inner{i}' for i in range(3)]
        refit=f'adult_reduced:outer{f}:refit'
        imodel=[seed_for(root,'adult_reduced',f,i,'inner_model') for i in range(3)]
        omodel=seed_for(root,'adult_reduced',f,'outer_model')
        rmodel=seed_for(root,'adult_reduced',f,'complete_outer_training_reference_model')
        prediction_seed=seed_for(root,'adult_reduced',f,'refit_timing_prediction')
        prediction=complete_group_prediction_load(data,excluded,6519,prediction_seed)
        reference_prediction_seed=seed_for(root,'adult_reduced',f,'reference_timing_prediction')
        reference_prediction=complete_group_prediction_load(data,available,6519,reference_prediction_seed)
        record={'fold':f,'original_train':segment(data,available),'test':segment(data,test),
                'pool':segment(data,pool),'excluded_pool':segment(data,excluded),
                'pool_seed':pool_seed,'pool_class_quotas':quota,'inner_seed':inner_seed,
                'inner_folds':inners,'inner_context_ids':pids,'refit_context_id':refit,
                'inner_model_seeds':imodel,'outer_model_seed':omodel,'reference_model_seed':rmodel,
                'timing_prediction_seed':prediction_seed,'timing_prediction':segment(data,prediction),
                'reference_timing_prediction_seed':reference_prediction_seed,
                'reference_timing_prediction':segment(data,reference_prediction)}
        outer.append(record)
        for i,pid in enumerate(pids+[refit]):
            train=inners[i]['train'] if i<3 else record['pool']
            context={'partition_id':pid,'fold':f,'kind':'inner' if i<3 else 'refit',
                     'inner_id':i if i<3 else None,'train':train,
                     'selector_sampler_seed':seed_for(root,pid,'selector_sampler'),
                     'model_seed':imodel[i] if i<3 else omodel}
            if i<3:context['validation']=inners[i]['validation']
            context['timing_prediction']=inners[i]['validation'] if i<3 else record['timing_prediction']
            contexts[pid]=context;games=[]
            for b in range(4):
                resample_seed=seed_for(root,pid,b,'feature_resample')
                positions=random_resample(data,train['positions'],0.8,resample_seed)
                game_seed=seed_for(root,pid,b,'utility');split_seed=seed_for(game_seed,'utility_split')
                utility=grouped_splits(data,positions,split_seed)
                for fold in utility:
                    u=fold['fold'];fold['train_positions']=fold['train']['positions'];fold['validation_positions']=fold['validation']['positions']
                    fold['sampler_seed']=seed_for(game_seed,u,'utility_sampler')
                    fold['model_seed']=seed_for(game_seed,u,'utility_model')
                    fold['kmeans_seeds']={str(c):seed_for(fold['sampler_seed'],'kmeans',c) for c in (0,1)}
                perm_seeds={m:seed_for(root,pid,m,b,'permutations') for m in METHODS}
                perms={};coalitions={}
                for method,pseed in perm_seeds.items():
                    rng=np.random.default_rng(pseed)
                    perms[method]=[rng.permutation(12).tolist() for _ in range(64)]
                    seen=set()
                    for order in perms[method]:
                        mask=0
                        for j in order:mask|=1<<j;seen.add(mask)
                    coalitions[method]=sorted(seen)
                games.append({'partition_id':pid,'resample_index':b,'positions':positions,
                              'resample':segment(data,positions),'resample_seed':resample_seed,'game_seed':game_seed,
                              'split_seed':split_seed,'permutation_seeds':perm_seeds,'permutations':perms,
                              'nonempty_coalition_masks':coalitions,'utility_folds':utility})
            games_by_context[pid]={'games':games}
        chosen=max(range(3),key=lambda i:(inners[i]['train']['n_rows'],-i))
        record['timing_maximum_inner_id']=chosen
        for context_id in [pids[chosen],refit]:
            context=contexts[context_id]
            for method in METHODS:
                for repeat,b in enumerate((0,1,3)):
                    tasks.append({'task_id':f"fold{f}-{context['kind']}-{method}-repeat{repeat}",
                                  'kind':'selector_timing','fold':f,'partition_id':context_id,
                                  'context':context['kind'],'inner_id':context['inner_id'],
                                  'method':method,'repeat':repeat,'resample_index':b,'masks':masks,
                                  'train_positions':context['train']['positions'],
                                  'prediction_positions':context['timing_prediction']['positions'],
                                  'prediction_overlap_allowed':False,'candidate_model_seed':context['model_seed'],
                                  'selector_sampler_seed':context['selector_sampler_seed'],
                                  'game_seed':games_by_context[context_id]['games'][b]['game_seed']})
        for repeat in range(3):
            tasks.append({'task_id':f'fold{f}-full-reference-repeat{repeat}',
                          'kind':'full_reference_timing','fold':f,'partition_id':f'adult_reduced:outer{f}:reference',
                          'context':'original_train','method':'full','repeat':repeat,
                          'resample_index':None,'masks':[4095],'train_positions':available,
                          'prediction_positions':reference_prediction,'prediction_overlap_allowed':True,
                          'candidate_model_seed':rmodel})
    common={'schema_version':1,'protocol_id':config['protocol_id'],'seed_root':root,
            'source_table_sha256':data['source_table_sha256'],'official_adult_test_used':False,
            'generation_mode':'membership_only_no_fit_or_prediction'}
    parts={**common,'environment':actual,'features':data['features'],'numeric':data['numeric'],'categorical':data['categorical'],
           'source_manifest_sha256':data['source_manifest_sha256'],
           'old_outer_partitions_sha256':sha(old/'protocols/partitions/adult_train_partitions.json'),
           'seed_derivation':"int.from_bytes(SHA256('|'.join(map(str,(root,)+parts))).digest()[:4], 'big')",
           'outer_folds':outer,'contexts':contexts,'identity':segment(data,range(32561)),
           'resample_rule':'row-level class-stratified sampling within parent train; unused rows never evaluated; utility split grouped',
           'new_inner_membership':True,'outer_test_evaluation_authorized':False}
    games={**common,'contexts':games_by_context,'method_independent_games':80,'selector_method_games':160}
    timing={**common,'tasks':tasks,'task_count':75,'selector_tasks':60,'reference_tasks':15,
            'mask_list':masks,'masks_sha256':canonical(masks),'model_cache_reuse':False,
            'note':'Timing only. Train-row overlap in reference prediction is deliberate, not validation.'}
    if len(contexts)!=20 or len(tasks)!=75:raise ValueError('Unexpected task coverage')
    return parts,games,timing

def immutable_write(path,value):
    content=(json.dumps(value,sort_keys=True,separators=(',', ':'),ensure_ascii=True)+'\n').encode('utf8')
    if path.exists():
        if path.read_bytes()!=content:raise RuntimeError('Refusing to alter existing artifact '+str(path))
        return
    partial=path.with_suffix(path.suffix+'.partial')
    if partial.exists():raise RuntimeError('Preserve previous partial write for inspection '+str(partial))
    with partial.open('xb') as f:f.write(content);f.flush()
    partial.replace(path)

def generate():
    paths=checked_paths()
    if not Path('F:/ChatGPT/论文大数据').is_dir():raise RuntimeError('F store unavailable')
    DATA_ROOT.mkdir(parents=True,exist_ok=True)
    if DATA_ROOT.resolve()!=Path(paths['data_root']).resolve():raise RuntimeError('Resolved storage mismatch')
    probe=DATA_ROOT/'write_access_probe.tmp'
    if probe.exists():raise RuntimeError('Unexpected previous write probe')
    with probe.open('xb') as f:f.write(b'bounded no-fit partition preparation\n')
    probe.unlink()
    parts,games,timing=build()
    entries={}
    for name,value in [('partitions.json',parts),('games.json',games),('timing_tasks.json',timing)]:
        path=DATA_ROOT/name;immutable_write(path,value)
        entries[name]={'path':str(path),'sha256':sha(path),'bytes':path.stat().st_size}
    manifest={'schema_version':1,'protocol_id':parts['protocol_id'],'storage_root':str(DATA_ROOT.resolve()),
              'files':entries,'source_table_sha256':parts['source_table_sha256'],
              'old_outer_partitions_sha256':parts['old_outer_partitions_sha256'],
              'generation_script_sha256':sha(__file__),'environment':parts['environment'],
              'no_fit':True,'new_fits':0,'new_predictions':0,'task_count':75,
              'contexts':20,'method_independent_games':80,'selector_method_games':160,
              'outer_test_rows':[f['test']['n_rows'] for f in parts['outer_folds']],
              'pool_counts':[f['pool']['class_counts'] for f in parts['outer_folds']]}
    (STAGE/'qa').mkdir(exist_ok=True)
    manifest_path=STAGE/'partition_manifest.json'
    # Manifest may be regenerated only when its content is identical.
    immutable_write(manifest_path,manifest)
    print(json.dumps(manifest,ensure_ascii=False))

def _load_file(name,verify=True):
    checked_paths();manifest=read(STAGE/'partition_manifest.json')
    entry=manifest['files'][name];path=Path(entry['path'])
    if path.resolve()!=(DATA_ROOT/name).resolve():raise ValueError('Unapproved partition target')
    if verify and (path.stat().st_size!=entry['bytes'] or sha(path)!=entry['sha256']):raise ValueError('Frozen partition bytes changed')
    return read(path)

def load_partitions(verify=True):
    parts=_load_file('partitions.json',verify);games=_load_file('games.json',verify)
    for pid,record in parts['contexts'].items():record['games']=games['contexts'][pid]['games']
    return parts

def load_games(verify=True):return _load_file('games.json',verify)

def load_timing_tasks(verify=True):return _load_file('timing_tasks.json',verify)['tasks']

def get_context(partitions,fold,context_id):
    if context_id=='refit':context_id=f'adult_reduced:outer{fold}:refit'
    elif context_id.startswith('inner-'):context_id=f"adult_reduced:outer{fold}:inner{int(context_id.split('-')[1])}"
    record=partitions['contexts'][context_id]
    if record['fold']!=fold:raise ValueError('Cross-fold context request')
    return record

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description='No-fit membership preparation only')
    parser.add_argument('--generate-no-fit',action='store_true')
    args=parser.parse_args()
    if args.generate_no_fit:generate()
    else:print('No action. Explicit --generate-no-fit creates only partition artifacts; never models.')
