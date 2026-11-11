"""Independent, read-only timing evidence reconciliation and optional gate publication.

No estimators are imported here. Missing timing evidence can never produce a
passing gate. Publication is a separate operator action after the 75-task probe.
"""
from pathlib import Path
import argparse,hashlib,json,math,sqlite3,sys
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from runtime import read_json,sha256,finite_number,ExecutionDenied,validate_paths,prior_probe_usage
from projection import total_seconds, PROBE_TASKS

def canonical(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False).encode()).hexdigest()
def positive(value,name,allow_zero=False):
    if not finite_number(value) or value<0 or (not allow_zero and value==0):raise ExecutionDenied('Invalid measured '+name)
    return value
def max_verified(values,name):
    if not values:raise ExecutionDenied('Empty timing series '+name)
    return max(positive(v,name,True) for v in values)

def evidence(paths,freeze_sha):
    validate_paths(paths)
    prior=prior_probe_usage()
    tasks=read_json(paths['timing_tasks_file'])['tasks']
    if len(tasks)!=75 or {t['task_id'] for t in tasks}!=PROBE_TASKS:raise ExecutionDenied('Frozen timing schedule is incomplete')
    runroot=Path(paths['run_root']).resolve();records=[];source_hashes={};maxima={}
    mapping={'utility_cycle':'t_utility','full_candidate_cycle':'t_full_candidate','reference_cycle':'t_reference',
             'encoder':'t_encoder','kmeans':'t_kmeans','game_setup':'t_game_setup','selector_setup':'t_selector_setup'}
    for task in tasks:
        path=runroot/'probe'/task['task_id']/'timing.json'
        if not path.resolve().is_relative_to(runroot):raise ExecutionDenied('Timing artifact escapes run root')
        marker=path.with_suffix('.json.sha256')
        if not path.is_file() or not marker.is_file() or marker.read_text('ascii').strip()!=sha256(path):raise ExecutionDenied('Missing or changed timing task: '+task['task_id'])
        record=read_json(path)
        if record.get('task_id')!=task['task_id'] or record.get('task_fingerprint')!=canonical(task) or record.get('status')!='complete':raise ExecutionDenied('Timing task does not match freeze')
        if record.get('cache_reuse') is not False or record.get('outer_test_used') is not False or record.get('official_adult_test_used') is not False:raise ExecutionDenied('Invalid timing data role or cache reuse')
        if record.get('kind')!=task['kind']:raise ExecutionDenied('Timing record role differs from frozen task')
        details=record.get('details',{})
        if details.get('utility_scores_retained') is not False or details.get('predictions_retained') is not False:raise ExecutionDenied('Timing may not retain scores or predictions')
        expected_overlap=task['kind']=='full_reference_timing'
        if task.get('prediction_overlap_allowed') is not expected_overlap or details.get('prediction_overlap_allowed') is not expected_overlap:raise ExecutionDenied('Timing overlap declaration differs from task role')
        expected={'classifier':1,'encoder':1,'kmeans':0} if task['kind']=='full_reference_timing' else {'classifier':64,'encoder':69 if task['method']=='robust_half' else 65,'kmeans':8 if task['method']=='robust_half' else 0}
        if record.get('counts')!=expected:raise ExecutionDenied('Timing count mismatch')
        for k in expected:
            samples=record['operation_seconds'][k]
            if len(samples)!=expected[k]:raise ExecutionDenied('Missing raw operation timing')
            raw=max((positive(v,k,True) for v in samples),default=0.)
            if k in ('encoder','kmeans'):
                if not math.isclose(raw,record['maxima'][k],rel_tol=1e-12,abs_tol=1e-12):raise ExecutionDenied('Raw timing maximum mismatch')
        if task['kind']=='selector_timing':
            cycles=details['utility_fold_cycle_measurements']
            if len(cycles)!=63 or details['masks']!=task['masks']:raise ExecutionDenied('Utility timing mask/fold coverage mismatch')
            raw=max_verified([x['cycle_seconds'] for x in cycles],'utility cycle')
            if raw!=record['maxima']['utility_cycle'] or details['candidate_cycle']['cycle_seconds']!=record['maxima']['full_candidate_cycle']:raise ExecutionDenied('Complete cycle timing mismatch')
        for key,target in mapping.items():maxima[target]=max(maxima.get(target,0),positive(record['maxima'][key],key,True))
        positive(record['actual_wall_seconds'],'task wall')
        source_hashes[path.relative_to(runroot).as_posix()]=sha256(path)
        source_hashes[marker.relative_to(runroot).as_posix()]=sha256(marker)
        records.append(record)
    ledger=Path(paths['ledger_file'])
    if not ledger.is_file():raise ExecutionDenied('Timing ledger is absent')
    with sqlite3.connect('file:'+ledger.as_posix()+'?mode=ro',uri=True) as db:
        if db.execute("SELECT value FROM metadata WHERE key='freeze_sha256'").fetchone()!=(freeze_sha,):raise ExecutionDenied('Timing ledger belongs to another freeze')
        if db.execute("SELECT COUNT(*) FROM sessions WHERE role='probe' AND status!='complete'").fetchone()[0] or db.execute("SELECT COUNT(*) FROM operations WHERE role='probe' AND status!='success'").fetchone()[0]:raise ExecutionDenied('Failed or unfinished new probe session/operation requires review; cannot pass gate')
        completed={x[0] for x in db.execute("SELECT task FROM completed_tasks WHERE role='probe'")}
        if completed!={t['task_id'] for t in tasks}:raise ExecutionDenied('Timing completion ledger incomplete')
        totals={k:n for k,n in db.execute("SELECT kind,COUNT(*) FROM operations WHERE role='probe' GROUP BY kind")}
        clean={'classifier':3855,'encoder':4035,'kmeans':240};caps={'classifier':4200,'encoder':4500,'kmeans':300}
        if any(not clean[k]<=totals.get(k,0)<=caps[k] for k in clean):raise ExecutionDenied('Timing ledger counts outside budget')
        for record in records:
            actual={k:n for k,n in db.execute("SELECT kind,COUNT(*) FROM operations WHERE role='probe' AND task=? GROUP BY kind",(record['task_id'],))}
            if any(actual.get(k,0)<n for k,n in record['counts'].items()):raise ExecutionDenied('Task record lacks ledger operations')
        elapsed=db.execute("SELECT COALESCE(SUM(seconds),0) FROM sessions WHERE role='probe'").fetchone()[0]
        positive(elapsed,'new probe cumulative wall')
        positive(prior['session_seconds'],'preserved old probe wall',True)
        maxima['T_probe_actual']=elapsed+prior['session_seconds']
        if elapsed<sum(r['actual_wall_seconds'] for r in records) or elapsed>7200:raise ExecutionDenied('New probe cumulative wall mismatch or 2h overrun')
        for kind,key in [('encoder','t_encoder'),('kmeans','t_kmeans')]:
            observed=db.execute("SELECT MAX(seconds) FROM operations WHERE role='probe' AND kind=?",(kind,)).fetchone()[0]
            maxima[key]=max(maxima[key],positive(observed,key),prior['operation_seconds_max'][kind])
    for key,value in maxima.items():positive(value,key)
    total=total_seconds(maxima,prior['session_seconds'],elapsed)
    combined={k:totals.get(k,0)+prior['counts'][k] for k in clean}
    return {'status':'passed' if total<=172800 else 'failed_runtime_feasibility','freeze_sha256':freeze_sha,
            'resource_contract_id':'adult-exact-resources-v3-20261110',
            'timing_tasks_complete':75,'independent_checks_passed':True,'measured_parameters':maxima,
            'projected_total_seconds':total,'probe_operation_counts':combined,'source_sha256':source_hashes,
            'prior_probe_operation_counts':prior['counts'],'new_probe_operation_counts':totals,
            'combined_probe_operation_counts':combined,'prior_probe_session_seconds':prior['session_seconds'],
            'new_probe_session_seconds':elapsed,'prior_failed_sessions_preserved':prior['failed_sessions'],
            'prior_operation_seconds_max':prior['operation_seconds_max'],
            'prior_probe_ledger_sha256':prior['old_ledger_sha256'],
            'engineering_projection_not_absolute_or_statistical_bound':True,'automatic_resume':False,'execution_authorized':False}

def verify_published_gate(paths,freeze_sha,gate):
    expected=evidence(paths,freeze_sha)
    for key in ('status','freeze_sha256','resource_contract_id','timing_tasks_complete','independent_checks_passed','measured_parameters','projected_total_seconds','source_sha256',
                'probe_operation_counts','prior_probe_operation_counts','new_probe_operation_counts','combined_probe_operation_counts',
                'prior_probe_session_seconds','new_probe_session_seconds','prior_failed_sessions_preserved','prior_probe_ledger_sha256','prior_operation_seconds_max'):
        if gate.get(key)!=expected[key]:raise ExecutionDenied('Published timing gate mismatch: '+key)
    if expected['status']!='passed':raise ExecutionDenied('Runtime feasibility failed')
    return expected

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--publish-reviewed-gate',action='store_true')
    args=parser.parse_args()
    from freeze import verify_freeze
    frozen=verify_freeze();paths=read_json(ROOT/'paths.json')
    result=evidence(paths,frozen['freeze_sha256'])
    if args.publish_reviewed_gate:
        dest=Path(paths['runtime_gate_file'])
        with dest.open('x',encoding='utf8') as out:json.dump(result,out,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps(result,ensure_ascii=False,allow_nan=False))
if __name__=='__main__':main()
