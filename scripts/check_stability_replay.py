"""Verify full replicate-0 replay or an unchanged completed-run resume."""
from pathlib import Path
import argparse
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.data import digest
from scripts.run_joint_pilot import read,save


def substantive(value):
    omit={'seconds','geometry_seconds','game_seconds','selector_compute_seconds','fit_seconds','predict_seconds',
          'game_sha256','model_sha256','decisions_sha256','artifact_sha256','timing'}
    if isinstance(value,dict): return {k:substantive(v) for k,v in value.items() if k not in omit}
    if isinstance(value,list): return [substantive(v) for v in value]
    return value


def hashes(run):
    return {p.relative_to(run).as_posix():digest(p) for p in sorted(run.rglob('*.json'))}


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--mode',choices=['snapshot','verify-resume','replay'],required=True)
    args=ap.parse_args(); run=ROOT/'runs/stability-htru2-pilot-20261001-v1'; qa=ROOT/'qa'
    if args.mode=='snapshot':
        save(qa/'stability_resume_hashes.json',hashes(run)); print('Snapshot saved'); return
    if args.mode=='verify-resume':
        before=read(qa/'stability_resume_hashes.json'); assert before==hashes(run)
        save(qa/'stability_resume_checks.json',{'status':'passed','unchanged_json_files':len(before)})
        print(f'All {len(before)} JSON hashes unchanged'); return
    replay=ROOT/'runs/stability-htru2-replay-20261001-v1'
    assert read(run/'manifest.json')['fingerprint']==read(replay/'manifest.json')['fingerprint']
    original=run/'replicate-0'; copy=replay/'replicate-0'
    left={p.relative_to(original).as_posix():p for p in original.rglob('*.json') if not p.name.endswith('.complete.json')}
    right={p.relative_to(copy).as_posix():p for p in copy.rglob('*.json') if not p.name.endswith('.complete.json')}
    assert left.keys()==right.keys()
    for rel,p in left.items(): assert substantive(read(p))==substantive(read(right[rel])),rel
    result=read(copy/'result.json')
    report={'status':'passed','replicate':0,'compared_artifacts':len(left),'counts':result['counts'],
            'comparison':'Exact equality of all substantive fields; timings and derivative hashes excluded'}
    save(qa/'stability_replay_checks.json',report); print(report)


if __name__=='__main__': main()
