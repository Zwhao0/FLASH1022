"""Independent CPU audit boundaries, before any healthy model is selected."""
import copy
import unittest

from flash_v1.stage_forward_evidence import SCHEMA
from flash_v1.stage_forward_localize import (audit_healthy, fallback, fit, infer,
    validate_row, workload_diagnostics, _hash, semantic_key, context_binding, infer_bound)


def sample(rank=0, step=13):
    expected = dict(all_reduce=0, all_to_all_single=1, all_gather=0)
    microbatches = [dict(microbatch=index, stream_id=1,
        stage_inputs=[dict(shape=[1, 16], dtype='int64')],
        work={'0:experts.0': [dict(shape=[index, 128], dtype='bfloat16')],
              '0:shared_expert': [dict(shape=[16, 128], dtype='bfloat16')]},
        forward_elapsed_us=100., duration_us=90., event_pairs=3,
        collective_counts=dict(expected), communication_intervals=[
            dict(kind='all_to_all_single', start_us=10., end_us=20.),
            dict(kind='ep_metadata', start_us=10., end_us=20.)]) for index in range(2)]
    return dict(schema=SCHEMA, record_type='stage_forward_aux', rank=rank,
        iteration=step, epoch=1, expected_microbatches=2,
        expected_collectives=expected, expected_ep_metadata_spans=1,
        stage_metadata=dict(layout='legacy', pp_stage=0, tp_size=2),
        microbatches=microbatches, invalid_reasons=[], coverage_complete=True,
        duration_us=180., event_pairs=6)


class AuditTests(unittest.TestCase):
    def setUp(self):
        self.rows = {phase: [sample(rank, step) for rank in (0, 1, 2)
            for step in range(13, end + 1)]
            for phase, end in (('fit', 44), ('calibration', 28))}
        self.manifest = dict(expected_ranks=[0, 1, 2], window_size=8,
            fit=dict(session_id='healthy-fit', seed=11, healthy=True, layout='legacy', start=13, end=44),
            calibration=dict(session_id='healthy-cal', seed=12, healthy=True, layout='legacy', start=13, end=28))

    def test_audit_retains_zero_tokens_and_reports_unfitted_actual_work(self):
        result = audit_healthy(self.rows, self.manifest)
        self.assertEqual(result['status'], 'audit_only_no_model_selected')
        self.assertEqual((result['fit_rows'], result['calibration_rows']), (96, 48))
        self.assertEqual(len(result['groups']), 1)
        features = workload_diagnostics(self.rows['fit'][0])
        self.assertEqual(features['empty_routed_calls'], 1)
        self.assertEqual(features['routed_tokens'], 1)
        self.assertEqual(features['shared_tokens'], 32)
        model, audit = fit(self.rows, self.manifest)
        self.assertEqual(audit['status'], 'healthy_calibrated_not_independently_validated')
        self.assertEqual(model['min_score'], 6.)

    def test_coverage_flag_cannot_hide_tampered_union_or_missing_work(self):
        for modify in (lambda row: row.update(duration_us=181.),
            lambda row: row['microbatches'][1]['work'].pop('0:shared_expert'),
            lambda row: row['microbatches'][0]['communication_intervals'][0].update(end_us=101.),
            lambda row: row['microbatches'][0]['communication_intervals'].pop(),
            lambda row: row['microbatches'][0].update(event_pairs=2)):
            row = sample()
            modify(row)
            with self.assertRaises(ValueError):
                validate_row(row)

    def test_dense_three_dimensional_work_is_retained_and_counted(self):
        row = sample()
        for item in row['microbatches']:
            item['work']['23:'] = [dict(shape=[1, 16, 896], dtype='bfloat16')]
        self.assertIs(validate_row(row), row)
        self.assertEqual(workload_diagnostics(row)['dense_tokens'], 32)
        self.assertEqual(row['microbatches'][0]['work']['23:'][0]['shape'], [1, 16, 896])
        row['microbatches'][0]['work']['0:experts.0'][0]['shape'] = [1, 16, 896]
        with self.assertRaises(ValueError):
            validate_row(row)

    def test_healthy_grid_duplicates_missing_and_overlapping_seed_rejected(self):
        cases = []
        rows = copy.deepcopy(self.rows)
        rows['fit'].pop()
        cases.append((rows, self.manifest))
        rows = copy.deepcopy(self.rows)
        rows['calibration'].append(rows['calibration'][0])
        cases.append((rows, self.manifest))
        manifest = copy.deepcopy(self.manifest)
        manifest['calibration']['seed'] = 11
        cases.append((self.rows, manifest))
        for rows, manifest in cases:
            with self.assertRaises(ValueError):
                audit_healthy(rows, manifest)

    def test_dc89_accepted_result_is_never_read_or_changed_by_auxiliary(self):
        original = dict(status='localized', rank=7, reason='original')
        self.assertIs(fallback(original, None, {'invalid': True}, None), original)
        original = dict(status='rejected', rank=None, reason='original rejection')
        result = fallback(original, [], None, [0, 1])
        self.assertEqual(result['status'], 'rejected')
        self.assertEqual(result['reason'], 'original rejection')
        self.assertEqual(result['stage_aux']['reason'], 'stage_aux_model_not_frozen')
        with self.assertRaises(ValueError):
            infer([], {'unsupported': True}, [0, 1])

    def test_only_three_supported_steps_can_localize_unknowns_never_zero(self):
        model, _ = fit(self.rows, self.manifest)
        rows = [copy.deepcopy(row) for row in self.rows['calibration'] if row['iteration'] <= 20]
        for row in rows:
            if row['rank'] == 2 or row['iteration'] > 15:
                row['microbatches'][0]['work']['0:experts.0'][0]['shape'][0] = 3
            elif row['rank'] == 0:
                row['duration_us'] += 20
                for mb in row['microbatches']:
                    mb['duration_us'] += 10
                    mb['forward_elapsed_us'] += 10
        result = infer(rows, model, [0, 1, 2])
        self.assertEqual((result['status'], result['rank']), ('localized', 0))
        self.assertEqual(result['support'][0]['comparable_iterations'], 3)
        self.assertNotIn(2, result['rank_scores'])
        self.assertIn(2, result['unknown'])
        self.assertEqual(result['runner_up'], 1)
        self.assertEqual(result['comparable_same_semantic_peers'], [1])

    def test_candidate_requires_same_semantic_peer_and_true_runner_up(self):
        model, _ = fit(self.rows, self.manifest)
        rows = [copy.deepcopy(row) for row in self.rows['calibration'] if row['iteration'] <= 20]
        original = next(iter(model['baselines'].values()))
        for row in rows:
            if row['rank'] == 0:
                row['duration_us'] += 20
                for mb in row['microbatches']:
                    mb['duration_us'] += 10
                    mb['forward_elapsed_us'] += 10
            else:
                row['stage_metadata']['pp_stage'] = 1
                model['baselines'][semantic_key(row)] = copy.deepcopy(original)
        model['sha256'] = _hash({key: value for key, value in model.items() if key != 'sha256'})
        result = infer(rows, model, [0, 1, 2])
        self.assertEqual(result['reason'], 'candidate_lacks_comparable_same_semantic_peer')
        self.assertIsNone(result['rank'])
        model['margin'] = 1
        with self.assertRaises(ValueError):
            infer(rows, model, [0, 1, 2])

    def test_calibrated_domain_can_expand_without_changing_fit_baseline(self):
        for row in self.rows['calibration']:
            row['microbatches'][0]['work']['0:experts.0'][0]['shape'][0] = 3
        model, _ = fit(self.rows, self.manifest)
        reference = next(iter(model['baselines'].values()))
        self.assertEqual(reference['routed_token_domain'], [1, 4])
        self.assertEqual(reference['median_us'], 180.)
        self.assertEqual(reference['count'], 96)
        self.assertEqual(reference['calibration_count'], 48)

    def test_structural_completeness_is_required_even_when_work_is_unknown(self):
        model, _ = fit(self.rows, self.manifest)
        rows = [row for row in self.rows['calibration'] if row['iteration'] <= 20]
        result = infer(rows[:-1], model, [0, 1, 2])
        self.assertIsNone(result['rank'])
        self.assertIn('incomplete_rank_eight_step_structure', result['reason'])

    def test_external_binding_preserves_original_model_and_rejects_missing_mixed_context(self):
        model, _ = fit(self.rows, self.manifest)
        original_hash = model['sha256']
        binding = context_binding(model, 'scoped_dispatch', {'test': 'attested observer class'})
        rows = [copy.deepcopy(row) for row in self.rows['calibration'] if row['iteration'] <= 20]
        self.assertEqual(infer_bound(rows, model, [0, 1, 2], binding)['reason'],
                         'incompatible_or_missing_instrumentation_context')
        for row in rows:
            row['stage_metadata']['instrumentation_context'] = 'scoped_dispatch'
        result = infer_bound(rows, model, [0, 1, 2], binding)
        self.assertEqual(len(result['rank_scores']), 3)
        self.assertEqual(model['sha256'], original_hash)
        rows[0]['stage_metadata']['instrumentation_context'] = 'plain'
        self.assertEqual(infer_bound(rows, model, [0, 1, 2], binding)['reason'],
                         'incompatible_or_missing_instrumentation_context')
        rows[0]['stage_metadata']['instrumentation_context'] = None
        self.assertEqual(infer_bound(rows, model, [0, 1, 2], binding)['status'], 'rejected')
        with self.assertRaises(ValueError):
            context_binding(model, 'plain', {'wrong': 'context'})
        binding['instrumentation_context'] = 'plain'
        with self.assertRaises(ValueError):
            infer_bound(rows, model, [0, 1, 2], binding)

    def test_plain8_protocol_is_explicit_same_session_split_not_old_default(self):
        rows = {phase: [sample(rank, step) for rank in (0, 1, 2) for step in range(start, end+1)]
            for phase, start, end in [('fit', 85, 92), ('calibration', 104, 111)]}
        for phase in rows:
            for row in rows[phase]:
                row['stage_metadata']['instrumentation_context'] = 'plain'
        manifest = dict(expected_ranks=[0, 1, 2], window_size=8,
            protocol='plain_cost_fit8_cal8_v1',
            fit=dict(session_id='existing-cost', seed=77, healthy=True, layout='legacy', start=85, end=92),
            calibration=dict(session_id='existing-cost', seed=77, healthy=True, layout='legacy', start=104, end=111))
        model, audit = fit(rows, manifest)
        self.assertEqual(next(iter(model['baselines'].values()))['count'], 24)
        self.assertEqual(audit['protocol'], 'plain_cost_fit8_cal8_v1')
        binding = context_binding(model, 'plain', {'test': 'known plain run'})
        self.assertEqual(infer_bound(rows['calibration'], model, [0, 1, 2], binding)['status'], 'rejected')
        with self.assertRaises(ValueError):
            context_binding(model, 'scoped_dispatch', {'wrong': 'context'})
        del manifest['protocol']
        with self.assertRaises(ValueError):
            fit(rows, manifest)


if __name__ == '__main__':
    unittest.main()
