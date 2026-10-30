import tempfile, unittest
from pathlib import Path
from src.budget import Budget, BudgetStop

class BudgetTests(unittest.TestCase):
    def make(self,folder,**overrides):
        limits=dict(classifier_attempts=2,encoder_attempts=2,kmeans_attempts=2,wall_seconds=60,invocation_seconds=60)
        limits.update(overrides)
        return Budget(Path(folder)/'ledger.sqlite3',limits,Path(folder)/'STOP')
    def test_attempts_include_failed_and_started_and_cannot_be_raised(self):
        with tempfile.TemporaryDirectory() as folder:
            b=self.make(folder); b.reserve('logistic'); n=b.reserve('random_forest')
            b.db.execute("UPDATE operations SET status='failed' WHERE id=?",(n,));b.db.commit()
            with self.assertRaises(BudgetStop):b.reserve('logistic')
            b.close()
            with self.assertRaises(ValueError):self.make(folder,classifier_attempts=3)
    def test_stop_and_resume_total_counts(self):
        with tempfile.TemporaryDirectory() as folder:
            b=self.make(folder); b.reserve('logistic'); b.close()
            b=self.make(folder); self.assertEqual(sum(r['attempts'] for r in b.counts()),1)
            (Path(folder)/'STOP').write_text('stop')
            with self.assertRaises(BudgetStop):b.reserve('encoder')
            b.close()
    def test_cumulative_time_cannot_reset_on_resume(self):
        with tempfile.TemporaryDirectory() as folder:
            b=self.make(folder); b.db.execute("INSERT INTO sessions(status,seconds) VALUES ('complete',61)");b.db.commit();b.close()
            b=self.make(folder)
            with self.assertRaises(BudgetStop):b.reserve('kmeans')
            b.close()
    def test_interrupted_session_fails_closed(self):
        with tempfile.TemporaryDirectory() as folder:
            b=self.make(folder);b.db.execute("INSERT INTO sessions(status,seconds) VALUES ('active',0)");b.db.commit()
            with self.assertRaisesRegex(RuntimeError,'Interrupted'):
                with b.instrument():pass
            b.close()
    def test_replay_audit_reserves_protected_from_main(self):
        with tempfile.TemporaryDirectory() as folder:
            b=self.make(folder,classifier_attempts=5,audit_reserve_attempts=1,replay_reserve_attempts=2)
            b.tag='main:package';b.reserve('logistic');b.reserve('logistic')
            with self.assertRaises(BudgetStop):b.reserve('logistic')
            b.tag='replay:package';b.reserve('logistic');b.reserve('logistic')
            b.tag='audit:package';b.reserve('logistic')
            self.assertEqual(b.count('classifiers'),5)
            with self.assertRaises(BudgetStop):b.reserve('random_forest')
            b.close()
if __name__=='__main__':unittest.main()
