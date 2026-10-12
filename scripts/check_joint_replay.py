"""Compare a prespecified full outer-fold replay and hash completed artifacts."""
from pathlib import Path
import argparse
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data import digest
from scripts.run_joint_pilot import read, save


def stable(value):
    omit = {'seconds','geometry_seconds','game_seconds','selector_compute_seconds','fit_seconds','predict_seconds',
            'game_sha256','model_sha256','decisions_sha256'}
    if isinstance(value, dict): return {k: stable(v) for k,v in value.items() if k not in omit}
    if isinstance(value, list): return [stable(v) for v in value]
    return value


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--mode', choices=['replay','snapshot','verify-resume'], required=True)
    args = ap.parse_args()
    run = ROOT/'runs/joint-htru2-pilot-20261001-v1'; qa = ROOT/'qa'
    snapshot = qa/'joint_resume_hashes.json'
    if args.mode == 'snapshot':
        files = {p.relative_to(run).as_posix():digest(p) for p in run.rglob('*.json')}
        save(snapshot, files); print(f'Saved {len(files)} artifact hashes'); return
    if args.mode == 'verify-resume':
        before = read(snapshot)
        now = {p.relative_to(run).as_posix():digest(p) for p in run.rglob('*.json')}
        assert before == now
        save(qa/'joint_resume_checks.json', {'status':'passed','unchanged_json_files':len(before)})
        print(f'Resume preserved all {len(before)} JSON hashes'); return
    replay = ROOT/'runs/joint-htru2-replay-20261001-v1'
    assert read(run/'manifest.json')['fingerprint'] == read(replay/'manifest.json')['fingerprint']
    root = run/'fold-0'; other = replay/'fold-0'; counts = {}
    originals = {p.relative_to(root).as_posix():p for p in root.rglob('*.json') if not p.name.endswith('.complete.json')}
    copies = {p.relative_to(other).as_posix():p for p in other.rglob('*.json') if not p.name.endswith('.complete.json')}
    assert originals.keys() == copies.keys()
    for rel, path in originals.items():
        assert stable(read(path)) == stable(read(copies[rel])), rel
        kind = ('game' if path.name.startswith('game-') else 'model' if path.name.startswith('mask-') else path.stem)
        counts[kind] = counts.get(kind,0)+1
    save(qa/'joint_replay_checks.json', {'status':'passed','fold':0,'counts':counts,
        'comparison':'Exact equality of all substantive values; timing and derivative file hashes excluded',
        'contribution_refits':sum(read(p)['logistic_fits'] for p in root.rglob('selector.json')),'scoring_refits':counts['model']})
    print(counts)


if __name__ == '__main__': main()
