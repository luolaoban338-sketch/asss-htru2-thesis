"""Separate prespecified replay and read-only completed-package resume checks."""
from pathlib import Path
import argparse,subprocess,sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.data import digest
from scripts.run_joint_pilot import read,save,cached
from scripts.run_ablation import MAIN,REPLAY,QA,package_path

def substantive(obj):
    # Derivative file hashes differ because the source objects include timings.
    omit={'seconds','fit_seconds','predict_seconds','model_sha256','game_sha256','sha256'}
    if isinstance(obj,dict):return {k:substantive(v) for k,v in obj.items() if k not in omit}
    if isinstance(obj,list):return [substantive(x) for x in obj]
    return obj

def paths(folder):
    return {p.relative_to(folder).as_posix():p for p in folder.rglob('*.json')
        if not p.name.endswith('.complete.json') and not p.name.startswith('audit')}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['predictive','stability'],required=True)
    ap.add_argument('--mode',choices=['replay','resume'],required=True);a=ap.parse_args()
    main_folder=package_path(ROOT/'runs'/MAIN,a.phase,2026100501,0)
    assert cached(main_folder/'audit.json')['status']=='passed'
    if a.mode=='replay':
        replay_folder=package_path(ROOT/'runs'/REPLAY,a.phase,2026100501,0)
        assert cached(replay_folder/'audit.json')['status']=='passed'
        assert read(ROOT/'runs'/MAIN/'manifest.json')['fingerprint']==read(ROOT/'runs'/REPLAY/'manifest.json')['fingerprint']
        left,right=paths(main_folder),paths(replay_folder);assert left.keys()==right.keys()
        for rel in left:
            assert substantive(read(left[rel]))==substantive(read(right[rel])),rel
        result={'status':'passed','phase':a.phase,'unit':0,'root':2026100501,'substantive_artifacts':len(left),
            'comparison':'Exact substantive equality; wall times and hashes derived from timed artifacts excluded',
            'main_audit_sha256':digest(main_folder/'audit.json'),'replay_audit_sha256':digest(replay_folder/'audit.json')}
    else:
        # Only invoke for a completed package and only while no other worker runs.
        before={p.relative_to(main_folder).as_posix():digest(p) for p in main_folder.rglob('*.json')}
        import sqlite3
        db=sqlite3.connect(ROOT/'runs'/MAIN/'fit_budget.sqlite3')
        n_before=db.execute('SELECT COUNT(*) FROM operations').fetchone()[0]
        subprocess.run([sys.executable,str(ROOT/'scripts/run_ablation.py'),'--phase',a.phase,'--unit','0','--root','2026100501'],cwd=ROOT,check=True)
        after={p.relative_to(main_folder).as_posix():digest(p) for p in main_folder.rglob('*.json')}
        n_after=db.execute('SELECT COUNT(*) FROM operations').fetchone()[0];db.close()
        assert before==after and n_before==n_after
        result={'status':'passed','phase':a.phase,'unit':0,'unchanged_json_files':len(before),'new_operation_attempts':0}
    save(QA/f'{a.phase}_{a.mode}_checks.json',result);print(result)

if __name__=='__main__':main()
