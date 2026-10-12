from pathlib import Path
import json
import subprocess
import sys
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.data import digest
from scripts.run_baselines import write_json
run=ROOT/'runs/baseline-htru2-pilot-20261001-v1'
copy=ROOT/'runs/baseline-htru2-replay-20261001-v1'
records=list(copy.glob('htru2/fold-0/*/record.json'))
assert len(records)==4
for p in records:
    original=run/p.relative_to(copy).parent
    for name in ['selection.json','encoder.json','model.json','group_metrics.json']:
        assert json.loads((original/name).read_text('utf-8'))==json.loads((p.parent/name).read_text('utf-8'))
    pd.testing.assert_frame_equal(pd.read_csv(original/'predictions.csv',float_precision='round_trip'),pd.read_csv(p.parent/'predictions.csv',float_precision='round_trip'),check_exact=True)
before={p.relative_to(run).as_posix():digest(p) for p in run.rglob('*') if p.is_file()}
subprocess.run([sys.executable,str(ROOT/'scripts/run_baselines.py'),'--run-id',run.name,'--dataset','htru2','--max-jobs','1'],check=True,cwd=ROOT)
after={p.relative_to(run).as_posix():digest(p) for p in run.rglob('*') if p.is_file()}
assert before==after
write_json(run/'qa/replay_checks.json',{'status':'passed','replayed_fits':4,'resume_unchanged_files':len(before),'predictions_exact':True})
print('Four baseline replays and full-file resume hashes passed')
