import unittest
from dataclasses import replace
from src.contracts import TrainingPartition, SampleSelection, FeatureSelection, class_quotas, assert_disjoint

class ContractTests(unittest.TestCase):
    def setUp(self):
        self.train = TrainingPartition(tuple(f'r{i}' for i in range(8)), (0,0,0,0,1,1,1,1),
            ('age','sex'), tuple(f'g{i}' for i in range(8)), 'train','outer0-inner0')
        self.result = SampleSelection(('r0','r1','r4','r5'), .5, {0:2,1:2}, 17,'stratified_random','outer0-inner0')

    def test_valid_half_selection(self):
        self.assertEqual(self.result.validate(self.train), .5)

    def test_full_selection_identity(self):
        r = replace(self.result,row_ids=self.train.row_ids,requested_rho=1,class_counts={0:4,1:4})
        self.assertEqual(r.validate(self.train),1)

    def test_wrong_partition(self):
        with self.assertRaises(ValueError): replace(self.result,partition_id='other').validate(self.train)

    def test_test_input_rejected(self):
        with self.assertRaises(ValueError): self.result.validate(replace(self.train,role='test'))

    def test_foreign_row_rejected(self):
        with self.assertRaises(ValueError): replace(self.result,row_ids=('r0','r1','r4','heldout')).validate(self.train)

    def test_duplicate_selected_row(self):
        with self.assertRaises(ValueError): replace(self.result,row_ids=('r0','r0','r4','r5')).validate(self.train)

    def test_false_class_counts(self):
        with self.assertRaises(ValueError): replace(self.result,class_counts={0:3,1:1}).validate(self.train)

    def test_group_leakage(self):
        with self.assertRaises(ValueError): assert_disjoint(self.train,['new_row'],['g0'])

    def test_row_leakage(self):
        with self.assertRaises(ValueError): assert_disjoint(self.train,['r0'],['new_group'])

    def test_disjoint_pass(self):
        assert_disjoint(self.train,['new_row'],['new_group'])

    def test_onehot_column_rejected(self):
        with self.assertRaises(ValueError): FeatureSelection(('sex_Male',),'outer0-inner0').validate(self.train)

    def test_original_variable_allowed(self):
        self.assertEqual(FeatureSelection(('sex',),'outer0-inner0').validate(self.train),.5)

    def test_quota_ceil_and_tie(self):
        self.assertEqual(class_quotas([0]*5+[1]*5,.5),{0:3,1:2})
        self.assertEqual(class_quotas([0]*5+[1]*4,.5),{0:3,1:2})

    def test_too_small_minority_is_infeasible(self):
        with self.assertRaises(ValueError): class_quotas([0]*18+[1]*2,.5)

    def test_invalid_budget(self):
        for rho in (0,-.1,1.1,float('nan'),float('inf')):
            with self.assertRaises(ValueError): class_quotas([0]*4+[1]*4,rho)

if __name__ == '__main__':
    unittest.main()
