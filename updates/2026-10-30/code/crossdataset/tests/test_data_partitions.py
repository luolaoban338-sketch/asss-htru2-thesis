"""No-fit tests for immutable data bindings and nested group exclusions."""
import copy
import json
import shutil
import tempfile
import unittest
from functools import lru_cache
from pathlib import Path
from unittest.mock import patch

import numpy as np

from src import data as data_module
from src.data import (ALLOWED_DATASETS, load_dataset, load_frozen_partitions,
                      validate_frozen_partitions)
from scripts.freeze_partitions import group_stratified_probe, segment


@lru_cache(maxsize=3)
def frozen(name):
    return load_dataset(name), load_frozen_partitions(name)


def copy_minimal_binding(tmp_path, name='htru2'):
    origin = data_module.ROOT
    manifest = json.loads((origin/'data/source_manifest.json').read_text('utf-8'))
    (tmp_path/'data/audit').mkdir(parents=True)
    shutil.copy2(origin/'data/source_manifest.json', tmp_path/'data/source_manifest.json')
    binding = manifest['datasets'][name]
    for relative in ['audit/audit_report.json', binding['table'], binding['dictionary']]:
        shutil.copy2(origin/'data'/relative, tmp_path/'data'/relative)
    return origin, binding


class DataPartitionTests(unittest.TestCase):
    def test_reserved_and_unknown_datasets_rejected_before_file_access(self):
        with patch.object(data_module, 'ROOT', Path('nonexistent-data-path')):
            for name in ['adult_test', 'compas', 'adult', '../adult_test', '/adult_test']:
                with self.subTest(name=name), self.assertRaisesRegex(ValueError, 'Unsupported development dataset'):
                    load_dataset(name)

    def test_original_variable_schemas_and_adult_categories(self):
        adult, _ = frozen('adult_train')
        self.assertEqual(adult.X.shape, (32561, 12))
        self.assertEqual(adult.numeric, data_module.ADULT_NUMERIC)
        self.assertEqual(adult.categorical, data_module.ADULT_CATEGORICAL)
        self.assertEqual(set(adult.audit.columns), {'audit_race', 'audit_sex'})
        self.assertFalse(set(adult.audit.columns) & set(adult.X.columns))
        self.assertIn('Bachelors', set(adult.X['education']))
        self.assertEqual(adult.X['workclass'].isna().sum(), 1836)
        self.assertEqual(len(adult.partition(np.arange(10), 'schema').feature_names), 12)
        for name, expected in [('wdbc', 30), ('htru2', 8)]:
            dataset, _ = frozen(name)
            self.assertEqual(len(dataset.X.columns), expected)
            self.assertEqual(len(dataset.numeric), expected)
            self.assertEqual(dataset.categorical, ())

    def test_processed_table_hash_tampering_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            _, binding = copy_minimal_binding(tmp_path)
            with (tmp_path/'data'/binding['table']).open('ab') as handle:
                handle.write(b'\n')
            with patch.object(data_module, 'ROOT', tmp_path):
                with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
                    load_dataset('htru2')

    def test_frozen_partition_file_hash_tampering_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            origin, _ = copy_minimal_binding(tmp_path)
            target = tmp_path/'protocols/partitions'
            target.mkdir(parents=True)
            shutil.copy2(origin/'protocols/partitions/manifest.json', target/'manifest.json')
            shutil.copy2(origin/'protocols/partitions/htru2_partitions.json', target/'htru2_partitions.json')
            with (target/'htru2_partitions.json').open('ab') as handle:
                handle.write(b' ')
            with patch.object(data_module, 'ROOT', tmp_path):
                with self.assertRaisesRegex(ValueError, 'Frozen partition checksum mismatch'):
                    load_frozen_partitions('htru2')

    def test_all_nested_rows_groups_and_outer_test_exclusions(self):
        for name in ALLOWED_DATASETS:
            with self.subTest(name=name):
                dataset, partitions = frozen(name)
                self.assertTrue(validate_frozen_partitions(dataset, partitions))
                outer_tests = []
                for outer in partitions['outer_folds']:
                    train, test = set(outer['train']['positions']), set(outer['test']['positions'])
                    self.assertTrue(train.isdisjoint(test))
                    self.assertTrue(set(dataset.group_ids[list(train)]).isdisjoint(dataset.group_ids[list(test)]))
                    outer_tests.extend(test)
                    inner_validation = []
                    for inner in outer['inner_folds']:
                        tr, va = set(inner['train']['positions']), set(inner['validation']['positions'])
                        self.assertEqual(tr | va, train)
                        self.assertTrue(tr.isdisjoint(va))
                        self.assertTrue((tr | va).isdisjoint(test))
                        self.assertTrue(set(dataset.group_ids[list(tr)]).isdisjoint(dataset.group_ids[list(va)]))
                        inner_validation.extend(va)
                    self.assertEqual(sorted(inner_validation), sorted(train))
                self.assertEqual(sorted(outer_tests), list(range(len(dataset.y))))
                probe = partitions['probe']
                members = set(probe['members']['positions'])
                self.assertTrue(members <= set(partitions['outer_folds'][0]['train']['positions']))
                self.assertEqual(len(members), probe['cap'])
                self.assertTrue(members.isdisjoint(partitions['outer_folds'][0]['test']['positions']))
                selected, quotas = group_stratified_probe(dataset, partitions['outer_folds'][0]['train']['positions'],
                                                         probe['cap'], probe['seed'])
                self.assertEqual(selected.tolist(), probe['members']['positions'])
                self.assertEqual(quotas, probe['class_quotas'])
                for inner in probe['inner_folds']:
                    tr, va = set(inner['train']['positions']), set(inner['validation']['positions'])
                    self.assertEqual(tr | va, members)
                    self.assertTrue(set(dataset.group_ids[list(tr)]).isdisjoint(dataset.group_ids[list(va)]))
                self.assertIs(partitions['outer_test_evaluation_authorized'], False)
                if name == 'htru2':
                    self.assertIs(partitions['predictive_execution_enabled'], False)

    def test_identity_tampering_is_rejected_even_with_parseable_json(self):
        dataset, partitions = frozen('wdbc')
        tampered = copy.deepcopy(partitions)
        tampered['probe']['members']['row_ids'][0] = 'other-dataset:row'
        with self.assertRaisesRegex(ValueError, 'identity, count or hash mismatch'):
            validate_frozen_partitions(dataset, tampered)

    def test_duplicate_predictor_group_cannot_be_split_across_outer_boundary(self):
        dataset, partitions = frozen('adult_train')
        tampered = copy.deepcopy(partitions)
        outer = tampered['outer_folds'][0]
        groups = {}
        for p in outer['train']['positions']:
            groups.setdefault(dataset.group_ids[p], []).append(p)
        duplicate = next(v for v in groups.values() if len(v) > 1)
        moved = duplicate[0]
        outer['train'] = segment(dataset, set(outer['train']['positions']) - {moved})
        outer['test'] = segment(dataset, set(outer['test']['positions']) | {moved})
        with self.assertRaisesRegex(ValueError, 'group leakage'):
            validate_frozen_partitions(dataset, tampered)


if __name__ == '__main__':
    unittest.main()
