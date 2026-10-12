from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from .contracts import TrainingPartition

ROOT = Path(__file__).resolve().parents[1]

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def seed_for(root, *parts):
    value = '|'.join(map(str,(root,)+parts)).encode('utf-8')
    return int.from_bytes(hashlib.sha256(value).digest()[:4], 'big')

@dataclass
class Dataset:
    name: str
    X: pd.DataFrame
    y: np.ndarray
    row_ids: np.ndarray
    group_ids: np.ndarray
    audit: pd.DataFrame
    numeric: tuple[str, ...]
    categorical: tuple[str, ...]
    sha256: str

    def partition(self, positions, partition_id):
        return TrainingPartition(tuple(self.row_ids[positions]), tuple(map(int,self.y[positions])),
            tuple(self.X.columns), tuple(self.group_ids[positions]), 'train', partition_id)

def load_dataset(name):
    # The reserved Adult official test is deliberately not accepted by this runner.
    if name not in ('htru2',):
        raise ValueError('Unsupported development dataset; official test is reserved')
    base=ROOT/'data/audit'
    report=json.loads((base/'audit_report.json').read_text('utf-8'))
    path=base/f'{name}_table.csv'
    expected=report['processed_sha256'][path.name]
    if digest(path)!=expected: raise ValueError('Processed data checksum mismatch')
    frame=pd.read_csv(path)
    fields=report['datasets'][name]['fields']
    features=[f['field'] for f in fields if f['role']=='predictor']
    numeric=tuple(f['field'] for f in fields if f['role']=='predictor' and f['type']=='numeric')
    categorical=tuple(f for f in features if f not in numeric)
    y=frame.target.to_numpy(dtype=int)
    if set(y)!={0,1} or not frame.row_id.is_unique: raise ValueError('Invalid dataset identity or target')
    return Dataset(name, frame[features].copy(),y,frame.row_id.to_numpy(dtype=str),
        frame.group_id.to_numpy(dtype=str),frame.filter(regex=r'^audit_').copy(),numeric,categorical,expected)
