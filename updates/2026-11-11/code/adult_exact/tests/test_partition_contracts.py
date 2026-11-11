"""No-fit partition/adapter equivalence contracts; real learning APIs are sentinels."""
from pathlib import Path
import ast, copy, json, sys, unittest
from unittest.mock import patch
import numpy as np
sys.dont_write_bytecode=True
STAGE=Path(__file__).resolve().parents[1]
PROJECT=STAGE.parent
sys.path.insert(0,str(STAGE))
import partitions, adapter
from private_science import engine

def function_tree(path,name):
    tree=ast.parse(path.read_text(encoding='utf8'))
    return next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)

class PartitionAndAdapterContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.parts=partitions.load_partitions()
        cls.tasks=partitions.load_timing_tasks()
        cls.config=json.loads((STAGE/'config.json').read_text(encoding='utf8'))

    def setUp(self):
        self.sentinals=[]
        for owner,name in [(engine.LogisticRegression,'fit'),(engine.LogisticRegression,'predict_proba'),
                           (engine.KMeans,'fit'),(engine.TabularEncoder,'fit')]:
            handle=patch.object(owner,name,side_effect=AssertionError('Real fitting/prediction is forbidden in no-fit contracts'))
            mocked=handle.start();self.sentinals.append(mocked);self.addCleanup(handle.stop)

    def tearDown(self):
        for mocked in self.sentinals:self.assertEqual(mocked.call_count,0)

    def tables(self):
        return [[{'candidate_id':i,'method':m,'k':k,'f1':.8,'actual_rho':1. if m=='full' else .5}
                 for i,(m,k) in enumerate((m,k) for m in ['full','robust_half'] for k in range(1,13))]
                for _ in range(3)]

    def test_read_api_and_all_75_tasks(self):
        self.assertEqual(len(self.parts['contexts']),20)
        self.assertEqual(len(self.tasks),75)
        for f in range(5):
            self.assertEqual(len(partitions.get_context(self.parts,f,'refit')['train']['positions']),8000)
            for i in range(3):self.assertEqual(len(partitions.get_context(self.parts,f,f'inner-{i}')['games']),4)

    def test_context_lookup_rejects_cross_fold(self):
        with self.assertRaises(ValueError):partitions.get_context(self.parts,0,'adult_reduced:outer1:refit')

    def test_utility_memberships_are_method_shared(self):
        for context in self.parts['contexts'].values():
            parent=set(context['train']['positions'])
            for record in context['games']:
                self.assertTrue(set(record['positions'])<=parent)
                self.assertEqual(set(record['permutation_seeds']),{'full','robust_half'})
                self.assertEqual(set(record['permutations']),{'full','robust_half'})
                for fold in record['utility_folds']:
                    self.assertFalse(set(fold['train_positions'])&set(fold['validation_positions']))
                    self.assertEqual(set(fold['train_positions'])|set(fold['validation_positions']),set(record['positions']))

    def test_reference_timing_overlap_is_only_explicit_case(self):
        for task in self.tasks:
            overlap=set(task['train_positions'])&set(task['prediction_positions'])
            if task['kind']=='full_reference_timing':
                self.assertTrue(task['prediction_overlap_allowed']);self.assertTrue(overlap)
            else:self.assertFalse(task['prediction_overlap_allowed']);self.assertFalse(overlap)

    def test_private_encoder_scientific_bodies_match_legacy_ast(self):
        old=ast.parse((PROJECT/'thesis_crossdataset_20261014/src/preprocessing.py').read_text(encoding='utf8'))
        new=ast.parse((STAGE/'private_science/preprocessing.py').read_text(encoding='utf8'))
        oldclass=next(x for x in old.body if isinstance(x,ast.ClassDef));newclass=next(x for x in new.body if isinstance(x,ast.ClassDef))
        for name in ['__init__','fit','transform','fit_transform','diagnostics']:
            a=copy.deepcopy(next(x for x in oldclass.body if isinstance(x,ast.FunctionDef) and x.name==name))
            b=copy.deepcopy(next(x for x in newclass.body if isinstance(x,ast.FunctionDef) and x.name==name))
            if name=='__init__':
                b.args.kwonlyargs=[];b.args.kw_defaults=[]
                b.body=[n for n in b.body if not (isinstance(n,ast.Assign) and ast.unparse(n.targets[0])=='self.session')]
            else:
                b.body=[n for n in b.body if not (isinstance(n,ast.If) and ast.unparse(n.test)=='self.session is None')
                        and not (isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and ast.unparse(n.value.func)=='self.session.require_execution')]
            self.assertEqual(ast.dump(a,include_attributes=False),ast.dump(b,include_attributes=False),name)

    def test_copied_arithmetic_modules_match_legacy_ast(self):
        pairs=[('contracts.py','src/contracts.py'),('evaluation.py','src/evaluation.py'),('shapley.py','experiments/shapley.py')]
        for private,legacy in pairs:
            a=ast.parse((PROJECT/'thesis_crossdataset_20261014'/legacy).read_text(encoding='utf8'))
            b=ast.parse((STAGE/'private_science'/private).read_text(encoding='utf8'))
            self.assertEqual(ast.dump(a,include_attributes=False),ast.dump(b,include_attributes=False),private)

    def test_four_decisions_equal_legacy_restricted_grid(self):
        tables=self.tables();actual=adapter.make_four_decisions(tables,self.config)
        legacy=PROJECT/'thesis_crossdataset_20261014/experiments/joint_selection.py'
        body=[function_tree(legacy,name) for name in ['choose_candidate','make_decisions']]
        namespace={'np':np};exec(compile(ast.Module(body=body,type_ignores=[]),'<pure-legacy-decisions>','exec'),namespace)
        expected=namespace['make_decisions'](actual['candidate_scores'],self.config,12)
        self.assertEqual(tuple(actual['selections']),adapter.ARMS)
        for arm in adapter.ARMS:self.assertEqual(actual['selections'][arm],expected['selections'][arm])
        self.assertEqual(actual['selections']['joint']['method'],'robust_half')
        self.assertEqual(actual['selections']['joint']['k'],1)

    def test_reject_extra_candidate(self):
        tables=self.tables();tables[0].append({'candidate_id':99,'method':'full','k':1,'f1':1.,'actual_rho':1.})
        with self.assertRaises(ValueError):adapter.make_four_decisions(tables,self.config)

    def test_reject_nonfinite_rho(self):
        tables=self.tables();tables[0][1]['actual_rho']=float('nan')
        with self.assertRaises(ValueError):adapter.make_four_decisions(tables,self.config)

    def test_reject_non12_features(self):
        with self.assertRaises(ValueError):adapter.make_four_decisions(self.tables(),self.config,d=11)

    def test_run_fold_requires_authorized_task_identity(self):
        class Session:
            role='main';task='fold-0'
            def require_execution(self):pass
            def artifact_path(self,relative):raise AssertionError('Must reject fold mismatch before resolving outputs')
        with self.assertRaises(PermissionError):adapter.run_fold(self.config,{},self.parts,1,'main',Session())

    def test_commit_preserves_interrupted_partial(self):
        folder=STAGE/'qa/partial_guard_fixture';folder.mkdir(exist_ok=True)
        target=folder/'result.json';partial=folder/'result.json.partial'
        fixture=b'preserve the original interrupted bytes\n'
        if partial.exists():self.assertEqual(partial.read_bytes(),fixture)
        else:partial.write_bytes(fixture)
        self.assertFalse(target.exists())
        with self.assertRaises(RuntimeError):engine.commit(target,{'new':'must not overwrite'})
        self.assertEqual(partial.read_bytes(),fixture);self.assertFalse(target.exists())

if __name__=='__main__':unittest.main()
