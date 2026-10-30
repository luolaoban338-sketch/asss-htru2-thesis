"""Executable interface checks for future row/feature selectors.

This module does not implement ASSS, clustering, training, or nested CV.
Caller-provided partition labels are assertions, not a security boundary.
"""
from dataclasses import dataclass
from math import ceil, isfinite
from typing import Mapping, Sequence

@dataclass(frozen=True)
class TrainingPartition:
    row_ids: tuple[str, ...]
    labels: tuple[int, ...]
    feature_names: tuple[str, ...]
    group_ids: tuple[str, ...]
    role: str
    partition_id: str

    def validate(self):
        if self.role != 'train':
            raise ValueError('Selectors accept training partitions only')
        if not self.partition_id or len(self.row_ids) == 0:
            raise ValueError('Partition ID and rows required')
        if len(set(self.row_ids)) != len(self.row_ids):
            raise ValueError('Duplicate row IDs')
        if len(self.labels) != len(self.row_ids) or len(self.group_ids) != len(self.row_ids):
            raise ValueError('Row, label and group lengths differ')
        if any(type(y) is not int or y not in (0,1) for y in self.labels):
            raise ValueError('Expected integer binary labels')
        if not self.feature_names or len(set(self.feature_names)) != len(self.feature_names):
            raise ValueError('Nonempty, unique original feature names required')
        if set(self.labels) != {0,1}:
            raise ValueError('Both classes required')

def assert_disjoint(train: TrainingPartition, heldout_row_ids: Sequence[str],
                    heldout_group_ids: Sequence[str]):
    train.validate()
    if set(train.row_ids) & set(heldout_row_ids):
        raise ValueError('Train/held-out row leakage')
    if set(train.group_ids) & set(heldout_group_ids):
        raise ValueError('Train/held-out group leakage')

def class_quotas(labels: Sequence[int], rho: float) -> dict[int, int]:
    """Hamilton largest remainders, then enforce minimum two by infeasibility.

    Do not silently modify rho or reallocate proportional quotas to rescue
    a candidate. This is an explicit stage-one resolution of the plan's rule.
    """
    if not isfinite(rho) or not 0 < rho <= 1:
        raise ValueError('rho must be finite and in (0,1]')
    if not labels or any(type(x) is not int or x not in (0,1) for x in labels):
        raise ValueError('Expected integer binary labels')
    n = len(labels); target = ceil(rho * n)
    counts = {c: labels.count(c) for c in (0,1)}
    # Integer arithmetic gives an exact deterministic remainder ordering.
    quota = {c: target * counts[c] // n for c in counts}
    order = sorted(counts, key=lambda c: (-(target*counts[c] % n), c))
    for c in order[:target-sum(quota.values())]:
        quota[c] += 1
    if any(quota[c] < 2 or quota[c] > counts[c] for c in counts):
        raise ValueError('Infeasible proportional class quota (minimum two per class)')
    return quota

@dataclass(frozen=True)
class SampleSelection:
    row_ids: tuple[str, ...]
    requested_rho: float
    class_counts: Mapping[int, int]
    seed: int
    strategy: str
    partition_id: str

    def validate(self, train: TrainingPartition):
        train.validate()
        if self.partition_id != train.partition_id:
            raise ValueError('Selection belongs to a different partition')
        if len(set(self.row_ids)) != len(self.row_ids):
            raise ValueError('Repeated selected rows')
        label_map = dict(zip(train.row_ids, train.labels))
        if not set(self.row_ids) <= set(label_map):
            raise ValueError('Selected rows outside current training partition')
        expected = class_quotas(train.labels, self.requested_rho)
        actual = {c:sum(label_map[i] == c for i in self.row_ids) for c in (0,1)}
        if actual != dict(self.class_counts) or actual != expected:
            raise ValueError('Actual selected rows do not match proportional quotas')
        if self.requested_rho == 1 and set(self.row_ids) != set(train.row_ids):
            raise ValueError('rho=1 must return all rows')
        if type(self.seed) is not int or not self.strategy:
            raise ValueError('Seed and strategy required')
        return len(self.row_ids) / len(train.row_ids)

@dataclass(frozen=True)
class FeatureSelection:
    feature_names: tuple[str, ...]
    partition_id: str

    def validate(self, train: TrainingPartition):
        train.validate()
        if self.partition_id != train.partition_id:
            raise ValueError('Feature selection belongs to a different partition')
        if not self.feature_names or len(set(self.feature_names)) != len(self.feature_names):
            raise ValueError('Final model requires a nonempty unique feature subset')
        if not set(self.feature_names) <= set(train.feature_names):
            raise ValueError('Only original variables may be selected; not one-hot columns')
        return len(self.feature_names) / len(train.feature_names)
