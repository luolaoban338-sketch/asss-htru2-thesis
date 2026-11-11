"""Pure no-fit checks for exact work and conservative duration arithmetic."""
from pathlib import Path
import copy
import math
import sys
import unittest

STAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STAGE))
from projection import remaining_counts, remaining_seconds, projection_terms, refresh_rates, total_seconds

RATES = {'t_utility': .01, 't_full_candidate': .02, 't_reference': .3,
         't_encoder': .001, 't_kmeans': .002, 't_game_setup': .004, 't_selector_setup': .005}


class ExactProjectionTests(unittest.TestCase):
    def test_initial_formula_preserves_typed_retry_costs(self):
        expected = 3.75 * (301350 * .01 + 1056 * .02 + 12 * .3
            + 42840 * .02 + 12 * .3 + 44132 * .001 + 152 * .002
            + 384 * .004 + 96 * .005) + 3600
        self.assertAlmostEqual(remaining_seconds(RATES), expected, places=9)
        self.assertAlmostEqual(total_seconds(RATES, 240.8535168, 100), expected + 340.8535168, places=9)

    def test_fold_credit_uses_each_actual_frozen_count(self):
        expected_remaining = (251259, 251004, 250968, 251394, 250866)
        for fold, expected in enumerate(expected_remaining):
            with self.subTest(fold=fold):
                result = remaining_counts([('main', f'fold-{fold}')])
                self.assertEqual(result['utility'], expected)
                self.assertEqual(result['other_cycles_including_direct_audit'], 980)
        self.assertEqual(remaining_counts([('replay', 'fold-0')])['utility'], 251259)

    def test_all_work_committed_leaves_full_retry_and_closure_reserve(self):
        done = [('main', f'fold-{f}') for f in range(5)] + [('replay', 'fold-0')]
        done += [('audit', f'main-fold-{f}') for f in range(5)] + [('audit', 'replay-fold-0')]
        counts = remaining_counts(done)
        for key in ('utility','other_cycles_including_direct_audit','complete_training_reference_and_audit',
                    'game_initializations_including_audit','selector_setups_including_audit'):
            self.assertEqual(counts[key], 0)
        self.assertEqual([counts[k] for k in ('retry_classifier_pool','retry_classifier_reference','retry_encoder','retry_kmeans')], [42840,12,44132,152])
        expected = 3.75 * (42840 * .02 + 12 * .3 + 44132 * .001 + 152 * .002) + 3600
        self.assertAlmostEqual(remaining_seconds(RATES, done), expected)

    def test_probe_completion_grants_no_production_credit(self):
        self.assertEqual(remaining_counts([('probe','fold4-full-reference-repeat2')]), remaining_counts())

    def test_partial_duplicate_unknown_and_source_less_completions_rejected(self):
        cases = [[('main','fold-0-partial')], [('main','fold-0'),('main','fold-0')],
                 [('replay','fold-1')], [('audit','main-fold-0')], [('main','fold-5')],
                 [('probe','fake')], [('main','fold-0','complete')], [(True,'fold-0')]]
        for done in cases:
            with self.subTest(done=done), self.assertRaises(ValueError):remaining_counts(done)

    def test_main_then_audit_credits_distinct_costs(self):
        main = remaining_counts([('main','fold-2')])
        audited = remaining_counts([('main','fold-2'),('audit','main-fold-2')])
        self.assertEqual(main['utility'], audited['utility'])
        self.assertEqual(main['other_cycles_including_direct_audit'] - audited['other_cycles_including_direct_audit'],100)
        self.assertEqual(main['complete_training_reference_and_audit'] - audited['complete_training_reference_and_audit'],1)

    def test_later_smaller_measurements_never_reduce_rates(self):
        updated = refresh_rates(RATES, [('t_utility',.1), ('t_utility', .001), ('t_encoder',.0001)])
        self.assertEqual(updated['t_utility'], .1)
        self.assertEqual(updated['t_encoder'], .001)
        self.assertGreater(remaining_seconds(updated), remaining_seconds(RATES))
        self.assertEqual(RATES['t_utility'], .01)

    def test_pool_and_reference_audit_maxima_stay_separate(self):
        pool = projection_terms(refresh_rates(RATES,[('t_direct_audit_cycle', .2)]))
        reference = projection_terms(refresh_rates(RATES,[('t_direct_audit_reference', .8)]))
        self.assertEqual(pool['retry_classifier_pool'], 42840*.2)
        self.assertEqual(pool['retry_classifier_reference'], 12*.3)
        self.assertEqual(reference['retry_classifier_pool'], 42840*.02)
        self.assertEqual(reference['retry_classifier_reference'], 12*.8)

    def test_each_required_rate_rejects_missing_zero_nonfinite_boolean(self):
        for key in RATES:
            for bad in (None, 0, -1, math.nan, math.inf, True):
                rates = dict(RATES); rates[key] = bad
                with self.subTest(key=key,bad=bad), self.assertRaises(ValueError):remaining_seconds(rates)

    def test_invalid_observations_and_probe_charges_rejected(self):
        for event in [('unknown',1),('t_encoder',math.nan),('t_encoder',-1),('t_encoder',True)]:
            with self.subTest(event=event), self.assertRaises(ValueError):refresh_rates(RATES,[event])
        for old,new in [(math.nan,1),(1,math.inf),(-1,1),(1,0),(True,1),(1,False)]:
            with self.subTest(old=old,new=new), self.assertRaises(ValueError):total_seconds(RATES,old,new)


if __name__=='__main__':unittest.main(verbosity=2)
