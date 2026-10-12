import unittest
from dataclasses import replace
import numpy as np
import pandas as pd
from src.data import Dataset
from experiments.feature_utility import FeatureUtility
from scripts.run_joint_pilot import read
from scripts.run_joint_contamination import ROOT,perturbed_data,frame_hash,frozen_choices,check_fingerprint
from scripts.audit_joint_contamination import reconstruct

class JointContaminationTests(unittest.TestCase):
    def setUp(self):
        n=120
        X=pd.DataFrame({'x':np.arange(n,dtype=float),'z':np.sin(np.arange(n))*10})
        ids=np.array([f'r{i:03}' for i in range(n)])
        self.data=Dataset('toy',X,np.array([0,1]*60),ids,ids.copy(),pd.DataFrame(index=range(n)),('x','z'),(),'toy')
        self.tr=np.arange(90); self.te=np.arange(90,120)
        self.base=read(ROOT/'configs/baseline_pilot_v1.json')

    def test_test_labels_groups_and_original_unchanged(self):
        before=self.data.X.copy(deep=True)
        changed,e=perturbed_data(self.data,self.tr,self.te,.1,7,10)
        pd.testing.assert_frame_equal(self.data.X,before)
        pd.testing.assert_frame_equal(changed.X.iloc[self.te],before.iloc[self.te])
        np.testing.assert_array_equal(changed.y,self.data.y)
        np.testing.assert_array_equal(changed.group_ids,self.data.group_ids)
        self.assertEqual(int((changed.X!=before).to_numpy().sum()),9)

    def test_nested_shared_events_and_independent_reconstruction(self):
        a,ea=perturbed_data(self.data,self.tr,self.te,.05,7,10)
        b,eb=perturbed_data(self.data,self.tr,self.te,.1,7,10)
        c,ec,scales=reconstruct(self.data,self.tr,.1,7,10)
        self.assertEqual(ea['events'],eb['events'][:5]); self.assertEqual(eb['events'],ec)
        pd.testing.assert_frame_equal(b.X,c.X)
        other,eother=perturbed_data(self.data,self.tr,self.te,.1,7,10)
        self.assertEqual(eb,eother); pd.testing.assert_frame_equal(b.X,other.X)

    def test_contamination_enters_utility_validation(self):
        changed,e=perturbed_data(self.data,self.tr,self.te,.1,7,10)
        game=FeatureUtility(changed.X.iloc[self.tr],changed.partition(self.tr,'toy'),changed.numeric,(),self.base['methods'][0],self.base,21,3)
        touched={v['row_id'] for v in e['events']}
        validation=[rid for fold in game.context['folds'] for rid in fold['validation_ids']]
        self.assertEqual(len(touched & set(validation)),9)
        self.assertEqual(game.context['frame_hash'],frame_hash(changed.X.iloc[self.tr]))
        self.assertNotEqual(game.context['frame_hash'],frame_hash(self.data.X.iloc[self.tr]))
        self.assertFalse(set(validation)&set(self.data.row_ids[self.te]))

    def test_clean_identity_and_partition_rejection(self):
        changed,e=perturbed_data(self.data,self.tr,self.te,0,7,10)
        pd.testing.assert_frame_equal(changed.X,self.data.X); self.assertFalse(e['events'])
        with self.assertRaises(ValueError): perturbed_data(self.data,self.tr,self.tr,0,7,10)

    def test_decisions_copied_without_retuning(self):
        j=read(ROOT/'configs/joint_pilot_v1.json')
        d=read(ROOT/'runs/joint-htru2-pilot-20261001-v1/fold-0/decisions.json')
        choices=frozen_choices(d,j); self.assertEqual(choices,d['selections'])
        choices['joint']['k']=99
        self.assertNotEqual(choices,d['selections'])
        broken={'selections':choices}
        with self.assertRaises(ValueError): frozen_choices(broken,j)

    def test_resume_rejects_drift(self):
        fingerprint={'code':'abc','config':'def','data':'ghi'}
        check_fingerprint(fingerprint,dict(fingerprint))
        for field in fingerprint:
            changed={**fingerprint,field:'changed'}
            with self.assertRaises(RuntimeError): check_fingerprint(fingerprint,changed)

if __name__=='__main__': unittest.main()
