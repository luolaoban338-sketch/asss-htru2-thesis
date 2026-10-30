from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from .contracts import TrainingPartition

ROOT = Path(__file__).resolve().parents[1]
ALLOWED_DATASETS = ('htru2', 'adult_train', 'wdbc')
EXPECTED_PREDICTOR_COUNTS = {'htru2': 8, 'adult_train': 12, 'wdbc': 30}
ADULT_NUMERIC = ('age', 'fnlwgt', 'education_num', 'capital_gain', 'capital_loss', 'hours_per_week')
ADULT_CATEGORICAL = ('workclass', 'education', 'marital_status', 'occupation', 'relationship', 'native_country')

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
    # Reject before constructing any path: official test and COMPAS are excluded.
    if name not in ALLOWED_DATASETS:
        raise ValueError('Unsupported development dataset; official test is reserved')
    manifest = json.loads((ROOT/'data/source_manifest.json').read_text('utf-8'))
    binding = manifest['datasets'][name]
    for relative in ('audit/audit_report.json', binding['table'], binding['dictionary']):
        if digest(ROOT/'data'/relative) != manifest['assets'][relative]:
            raise ValueError('Processed data checksum mismatch: ' + relative)
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
    if len(features) != EXPECTED_PREDICTOR_COUNTS[name] or len(set(features)) != len(features):
        raise ValueError('Original-variable predictor schema mismatch')
    if not set(features).issubset(frame.columns) or set(features) & {'target', 'row_id', 'group_id', 'race', 'sex'}:
        raise ValueError('Invalid predictor role or missing column')
    if any(f.startswith('audit_') for f in features):
        raise ValueError('Audit attributes cannot be predictors')
    if name == 'adult_train' and (numeric != ADULT_NUMERIC or categorical != ADULT_CATEGORICAL):
        raise ValueError('Adult must retain its original numerical and categorical variables')
    if name != 'adult_train' and categorical:
        raise ValueError('Expected a numeric-only dataset')
    if any(not pd.api.types.is_numeric_dtype(frame[f]) for f in numeric):
        raise ValueError('Numeric predictor type mismatch')
    if any(not frame[f].dropna().map(lambda v: isinstance(v, str)).all() for f in categorical):
        raise ValueError('Categorical predictor type mismatch')
    if not np.isfinite(frame[list(numeric)].to_numpy(dtype=float)).all():
        raise ValueError('Numeric predictors must be finite')
    expected_rows = report['datasets'][name].get('rows', report['datasets'][name].get('n_rows'))
    if len(frame) != expected_rows or frame[['row_id','group_id','target']].isna().any().any():
        raise ValueError('Invalid row count or missing identity/target')
    y=frame.target.to_numpy(dtype=int)
    if set(frame.target)!={0,1} or set(y)!={0,1} or not frame.row_id.is_unique: raise ValueError('Invalid dataset identity or target')
    return Dataset(name, frame[features].copy(),y,frame.row_id.to_numpy(dtype=str),
        frame.group_id.to_numpy(dtype=str),frame.filter(regex=r'^audit_').copy(),numeric,categorical,expected)


def canonical_sha256(value):
    """Version-stable digest for ordered identity/position lists and objects."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    ensure_ascii=True).encode('utf-8')).hexdigest()


def load_frozen_partitions(name):
    """Read immutable partitions, bound to verified processed data and file hash.

    Membership and leakage checks are performed by validate_frozen_partitions;
    this loader additionally rejects data or persisted partition tampering.
    """
    data = load_dataset(name)
    base = ROOT/'protocols/partitions'
    manifest = json.loads((base/'manifest.json').read_text('utf-8'))
    entry = manifest['datasets'][name]
    path = base/entry['file']
    if digest(path) != entry['sha256']:
        raise ValueError('Frozen partition checksum mismatch')
    partitions = json.loads(path.read_text('utf-8'))
    if partitions['dataset_sha256'] != data.sha256:
        raise ValueError('Frozen partitions refer to a different dataset')
    if manifest['source_manifest_sha256'] != digest(ROOT/'data/source_manifest.json'):
        raise ValueError('Data binding manifest differs from frozen partitions')
    validate_frozen_partitions(data, partitions)
    return partitions


def validate_frozen_partitions(data, partitions):
    """Prove frozen IDs, complete group membership, and every nested exclusion."""
    if partitions['dataset'] != data.name or partitions['dataset_sha256'] != data.sha256:
        raise ValueError('Partition dataset binding mismatch')
    universe = set(range(len(data.y)))

    def check_segment(record):
        positions = record['positions']
        if positions != sorted(set(positions)) or not set(positions) <= universe:
            raise ValueError('Invalid, duplicated or unordered partition positions')
        ids = data.row_ids[positions].tolist()
        groups = sorted(set(data.group_ids[positions].tolist()))
        expected = {'positions_sha256': canonical_sha256(positions),
                    'row_ids_sha256': canonical_sha256(ids),
                    'group_ids_sha256': canonical_sha256(groups),
                    'n_rows': len(positions), 'n_groups': len(groups), 'row_ids': ids,
                    'class_counts': {str(c): int(np.sum(data.y[positions] == c)) for c in (0, 1)}}
        if any(record[key] != value for key, value in expected.items()):
            raise ValueError('Partition identity, count or hash mismatch')
        if not all(expected['class_counts'].values()):
            raise ValueError('Partition must contain both classes')
        return set(positions), set(groups)

    def check_pair(left, right, parent):
        left_pos, left_groups = check_segment(left)
        right_pos, right_groups = check_segment(right)
        if left_pos & right_pos or left_pos | right_pos != parent:
            raise ValueError('Nested partition row leakage or incomplete coverage')
        if left_groups & right_groups:
            raise ValueError('Nested partition group leakage')
        return left_pos, right_pos

    def check_inner(folds, parent):
        if len(folds) != 3 or [f['fold'] for f in folds] != list(range(3)):
            raise ValueError('Expected exactly three ordered inner folds')
        covered = set()
        for fold in folds:
            _, validation = check_pair(fold['train'], fold['validation'], parent)
            if covered & validation:
                raise ValueError('Repeated inner validation membership')
            covered.update(validation)
        if covered != parent:
            raise ValueError('Inner validation folds do not cover training partition')

    identity, _ = check_segment(partitions['identity'])
    if identity != universe:
        raise ValueError('Dataset identity coverage mismatch')
    folds = partitions['outer_folds']
    if len(folds) != 5 or [f['fold'] for f in folds] != list(range(5)):
        raise ValueError('Expected exactly five ordered outer folds')
    covered = set()
    for fold in folds:
        train, test = check_pair(fold['train'], fold['test'], universe)
        if covered & test:
            raise ValueError('Repeated outer test membership')
        covered.update(test)
        check_inner(fold['inner_folds'], train)
    if covered != universe:
        raise ValueError('Outer folds do not cover the dataset')
    probe = partitions['probe']
    if probe['scope'] != 'outer_fold_0_training_only' or probe['outer_fold'] != 0:
        raise ValueError('Invalid probe scope')
    outer_training = set(folds[0]['train']['positions'])
    members, _ = check_pair(probe['members'], probe['excluded_outer_training'], outer_training)
    if len(members) > probe['cap'] or probe['members']['class_counts'] != probe['class_quotas']:
        raise ValueError('Probe exceeds frozen cap or violates prespecified quotas')
    check_inner(probe['inner_folds'], members)
    if partitions['outer_test_evaluation_authorized'] is not False:
        raise ValueError('A1 does not authorize outer-test evaluation')
    if data.name == 'htru2' and partitions['predictive_execution_enabled'] is not False:
        raise ValueError('A1 HTRU2 predictive execution is disabled')
    return True
