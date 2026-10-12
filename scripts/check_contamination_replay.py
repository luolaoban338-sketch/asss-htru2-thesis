from pathlib import Path
import json
import subprocess
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data import digest
from scripts.run_baselines import write_json

run = ROOT / 'runs/contamination-htru2-pilot-20261001-v1'
replay = ROOT / 'runs/contamination-htru2-replay-20261001-v1'
checked = []
for p in sorted(replay.glob('*/fold-0/rate-*/*/record.json')):
    original = run / p.relative_to(replay).parent
    for name in ['selection.json', 'model.json', 'encoder.json', 'group_metrics.json']:
        assert json.loads((original / name).read_text('utf-8')) == json.loads((p.parent / name).read_text('utf-8'))
    assert json.loads((original.parent / 'contamination.json').read_text('utf-8')) == json.loads((p.parent.parent / 'contamination.json').read_text('utf-8'))
    pd.testing.assert_frame_equal(pd.read_csv(original / 'predictions.csv', float_precision='round_trip'),
        pd.read_csv(p.parent / 'predictions.csv', float_precision='round_trip'), check_exact=True)
    checked.append(p.relative_to(replay).as_posix())
assert len(checked) == 28
records = sorted((run / 'htru2').glob('fold-*/rate-*/*/record.json'))
before = {str(p): {'hash': digest(p), 'mtime_ns': p.stat().st_mtime_ns} for p in records}
result = subprocess.run([sys.executable, str(ROOT / 'scripts/run_contamination.py'), '--run-id', run.name,
    '--dataset', 'htru2', '--max-jobs', '1'], cwd=ROOT, capture_output=True, text=True, check=True)
assert 'completed=0, skipped=140' in result.stdout
after = {str(p): {'hash': digest(p), 'mtime_ns': p.stat().st_mtime_ns} for p in records}
assert before == after and len(records) == 140
write_json(run / 'qa/replay_checks.json', {'status': 'passed', 'representative_replays': len(checked),
    'checks': checked, 'selections_models_encoders_predictions_equal': True, 'resume_records_unchanged': len(records),
    'timing_not_compared': True})
print('28 deterministic replays and 140-record no-overwrite resume check passed')
