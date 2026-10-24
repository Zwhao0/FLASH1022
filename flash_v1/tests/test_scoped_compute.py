"""CPU-only filtering/population audit; does not certify GPU backward coverage."""
from types import SimpleNamespace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from flash_v1.scoped_compute import ScopedComputeSlowdown, attest_recorded_catalog
from flash_v1.tests.test_whole_compute import Operator, Tensor, Workload, setup


def make(cls=ScopedComputeSlowdown):
    state, original = setup()
    mode = cls(original.torch, workload_factory=Workload)
    mode.declarations = {name: dict(path=name, module_class='Fake', excluded_reason=None)
                         for name in ('layer0', 'layer0.attention', 'layer1.experts', 'unexecuted')}
    op, args = Operator(state), (Tensor(4, 8), Tensor(8, 16))
    mode.begin_calibration()
    with mode.training():
        with mode.scope('layer0', 'fwd'), mode.scope('layer0.attention', 'fwd'):
            mode.dispatch(op, args)
            mode.dispatch(op, args)
        with mode.scope('layer1.experts', 'fwd'):
            mode.dispatch(op, args)
        state.phase = 1
        with mode.scope('layer0', 'bwd'), mode.scope('layer0.attention', 'bwd'):
            mode.dispatch(op, args)
    mode.finish_calibration()
    mode.freeze_scope_catalog(gpu_numerical_and_coverage_validated=True)  # fake test fixture only
    state.phase = -1
    return state, mode, op, args


class ScopedComputeTests(unittest.TestCase):
    def test_complete_population_no_profile_selection(self):
        _, mode, _, _ = make()
        catalog = mode.scope_catalog()
        self.assertEqual(len(catalog['operators']), 3)
        self.assertEqual(catalog['component_operator_phases'], ['fwd'])
        self.assertEqual(catalog['phase_coverage']['bwd']['compute_calls'], 1)
        self.assertTrue(all(row['phase'] == 'fwd' for row in catalog['operators']))
        self.assertEqual({r['module_path'] for r in catalog['operators']},
                         {'layer0.attention', 'layer1.experts'})
        self.assertEqual([r['eligible'] for r in catalog['components']], [True, True, True, False])
        self.assertEqual(catalog['components'][-1]['excluded_reason'], 'no_allowlisted_cuda_compute_observed')
        self.assertEqual(catalog['catalog_sha256'], mode.scope_catalog()['catalog_sha256'])

    def test_fwd_phase_covers_multiple_components_but_not_optimizer_or_bwd(self):
        state, mode, op, args = make()
        before = state.events, state.syncs
        mode.begin_iteration(10, 2, dict(kind='stage', phase='fwd'))
        with mode.training():
            with mode.scope('layer0.attention', 'fwd'):
                mode.dispatch(op, args)
            with mode.scope('layer1.experts', 'fwd'):
                mode.dispatch(op, args)
            state.phase = 1
            with mode.scope('layer0.attention', 'bwd'):
                mode.dispatch(op, args)
        state.phase = -1
        mode.dispatch(op, args)  # optimizer/outside schedule
        row = mode.end_iteration()
        self.assertEqual(row['selected_compute_calls'], 2)
        self.assertAlmostEqual(row['requested_extra_ms'], .4)
        self.assertEqual((state.events, state.syncs), before)

    def test_component_subtree_and_operator_exact_slot(self):
        _, mode, op, args = make()
        mode.begin_iteration(11, 1, dict(kind='component', slot_id='layer0'))
        with mode.training(), mode.scope('layer0', 'fwd'), mode.scope('layer0.attention', 'fwd'):
            mode.dispatch(op, args)
        self.assertEqual(mode.end_iteration()['selected_compute_calls'], 1)
        slot = 'layer0.attention|fwd|aten.mm.default|1'
        mode.begin_iteration(12, 1, dict(kind='operator', slot_id=slot))
        with mode.training(), mode.scope('layer0.attention', 'fwd'):
            mode.dispatch(op, args)
            mode.dispatch(op, args)
        self.assertEqual(mode.end_iteration()['selected_compute_calls'], 1)

    def test_missing_site_reports_without_retargeting(self):
        _, mode, op, args = make()
        mode.begin_iteration(13, 1, dict(kind='operator', slot_id='layer0.attention|fwd|aten.mm.default|1'))
        with mode.training(), mode.scope('layer1.experts', 'fwd'):
            mode.dispatch(op, args)
        row = mode.end_iteration()
        self.assertTrue(row['selected_scope_missed'])
        self.assertEqual(row['injected_launches'], 0)
        self.assertEqual(row['requested_extra_ms'], 0)

    def test_whole_never_launches_outside_training_schedule(self):
        _, mode, op, args = make()
        mode.begin_iteration(15, 2, dict(kind='whole'))
        mode.dispatch(op, args)  # optimizer before schedule
        self.assertEqual(mode.engine.launches, [])
        with mode.training(), mode.scope('layer1.experts', 'fwd'):
            mode.dispatch(op, args)
        count = len(mode.engine.launches)
        self.assertGreater(count, 0)
        mode.dispatch(op, args)  # optimizer after schedule
        row = mode.end_iteration()
        self.assertEqual(len(mode.engine.launches), count)
        self.assertEqual(row['selected_compute_calls'], 1)
        self.assertAlmostEqual(row['requested_extra_ms'], .2)

    def test_frozen_catalog_roundtrip_and_tamper_rejection(self):
        _, mode, _, _ = make()
        _, original = setup()
        restored = ScopedComputeSlowdown(original.torch, workload_factory=Workload)
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'scopes.json'
            path.write_text(json.dumps(mode.frozen_scope_catalog))
            restored.load_scope_catalog(path)
            self.assertEqual(restored.frozen_scope_catalog, mode.frozen_scope_catalog)
            modified = dict(mode.frozen_scope_catalog, rank=8)
            path.write_text(json.dumps(modified))
            with self.assertRaisesRegex(ValueError, 'hash/rank'):
                restored.load_scope_catalog(path)

    def test_changed_operator_trace_invalidates_case(self):
        _, mode, op, args = make()
        mode.begin_iteration(14, 1, dict(kind='operator',
            slot_id='layer0.attention|fwd|aten.mm.default|0'))
        with mode.training(), mode.scope('layer0.attention', 'fwd'):
            mode.dispatch(op, args)  # calibration's scope had two mm calls
        row = mode.end_iteration()
        self.assertFalse(row['scope_execution_valid'])
        self.assertEqual(row['scope_trace_drift'], ['layer0.attention|fwd'])

    def test_bwd_stage_injects_without_any_backward_module_scope(self):
        state, mode, op, args = make()
        mode.begin_iteration(30, 2, dict(kind='stage', phase='bwd'), case_id='bwd')
        with mode.training():
            state.phase = -1
            mode.dispatch(op, args)
            state.phase = 1
            mode.dispatch(op, args)
        row = mode.end_iteration()
        self.assertEqual(row['selected_compute_calls'], 1)
        self.assertGreater(row['injected_launches'], 0)
        self.assertEqual({row['phase'] for row in row['operators']}, {'bwd'})

    def test_attach_never_replaces_kwargs_or_registers_backward_hooks(self):
        _, original = setup()
        mode = ScopedComputeSlowdown(original.torch, workload_factory=Workload)
        class Module:
            def named_modules(self):
                return [('', self)]
            def register_forward_pre_hook(self, fn, **kwargs):
                self.pre = fn
                return SimpleNamespace(remove=lambda: None)
            def register_forward_hook(self, fn, **kwargs):
                self.post = fn
                return SimpleNamespace(remove=lambda: None)
            def register_full_backward_pre_hook(self, *args, **kwargs):
                raise AssertionError('backward hooks change autograd views')
            register_full_backward_hook = register_full_backward_pre_hook
        module = Module()
        mode.attach(module)
        hidden = object()
        kwargs = dict(hidden_states=hidden)
        with mode.training():
            self.assertIsNone(module.pre(module, (), kwargs))
            self.assertIs(kwargs['hidden_states'], hidden)
            module.post(module, (), kwargs, hidden)
        self.assertEqual(len(mode.handles), 2)
        mode.detach()

    def test_cpu_attestation_requires_catalog_bound_real_proof_record(self):
        _, mode, _, _ = make()
        raw = mode.scope_catalog()
        with TemporaryDirectory() as directory:
            catalog_path, proof_path, output = [Path(directory) / name for name in
                                                ('raw.json', 'proof.json', 'validated.json')]
            catalog_path.write_text(json.dumps(raw))
            proof = dict(status='passed', numerical_validation='passed',
                actual_scope_coverage='passed', localizer_outputs_used=False,
                catalog_sha256=raw['catalog_sha256'], evidence='CPU fake fixture only')
            proof_path.write_text(json.dumps(proof))
            validated = attest_recorded_catalog(catalog_path, proof_path, output)
            _, original = setup()
            restored = ScopedComputeSlowdown(original.torch, workload_factory=Workload)
            self.assertEqual(restored.load_scope_catalog(output), validated)
            self.assertEqual(validated['raw_catalog_sha256'], raw['catalog_sha256'])
            self.assertIn('validation_proof_sha256', validated)
            proof['actual_scope_coverage'] = 'pending'
            proof_path.write_text(json.dumps(proof))
            with self.assertRaisesRegex(ValueError, 'pass numerical'):
                attest_recorded_catalog(catalog_path, proof_path)
            proof['actual_scope_coverage'] = 'passed'
            proof['catalog_sha256'] = 'different'
            proof_path.write_text(json.dumps(proof))
            with self.assertRaisesRegex(ValueError, 'bind this catalog'):
                attest_recorded_catalog(catalog_path, proof_path)

    def test_case_boundary_discards_credit_without_launch_or_cross_case_dose(self):
        _, mode, op, args = make()
        target = dict(kind='operator', slot_id='layer0.attention|fwd|aten.mm.default|0')
        mode.begin_iteration(20, .2, target, case_id='A')
        with mode.training(), mode.scope('layer0.attention', 'fwd'):
            mode.dispatch(op, args)  # .02 ms, below .05 quantum
        self.assertEqual(mode.engine.launches, [])
        self.assertAlmostEqual(mode.end_iteration()['pending_extra_ms'], .02)
        mode.begin_iteration(21, .2, target, case_id='B')
        self.assertEqual(mode.engine.launches, [])
        before = mode.end_iteration()
        self.assertEqual(before['pending_before_ms'], 0)
        self.assertAlmostEqual(before['pending_discard']['signed_ms'], .02)
        self.assertEqual(before['pending_discard']['policy'], 'discard_at_case_boundary_no_flush')
        with mode.training(), mode.scope('layer0.attention', 'fwd'):
            mode.dispatch(op, args)
        self.assertAlmostEqual(mode.end_iteration()['pending_extra_ms'], .02)
        self.assertEqual(mode.engine.launches, [])
        mode.begin_iteration(22, 0)
        self.assertAlmostEqual(mode.end_iteration()['pending_discard']['signed_ms'], .02)

    def test_explicit_target_and_frozen_population_are_required(self):
        _, mode, _, _ = make()
        with self.assertRaisesRegex(ValueError, 'explicit target'):
            mode.begin_iteration(16, 1)
        with self.assertRaisesRegex(ValueError, 'outside frozen'):
            mode.begin_iteration(16, 1, dict(kind='operator', slot_id='not_in_catalog'))
        with self.assertRaisesRegex(ValueError, 'outside frozen'):
            mode.begin_iteration(16, 1, dict(kind='component', slot_id='unexecuted'))

    def test_freeze_requires_explicit_gpu_validation(self):
        _, original = setup()
        mode = ScopedComputeSlowdown(original.torch, workload_factory=Workload)
        with self.assertRaisesRegex(RuntimeError, 'GPU numerical'):
            mode.freeze_scope_catalog()
        with self.assertRaisesRegex(ValueError, 'stage phase'):
            mode.begin_iteration(1, target=dict(kind='stage', phase='optimizer'))


class ScopeCheapGuardTests(unittest.TestCase):
    def test_unselected_scope_phase_and_optimizer_never_inspect_operator(self):
        from contextlib import ExitStack
        from unittest.mock import patch
        from flash_v1.whole_compute import WholeComputeSlowdown
        class DirectOnly:
            calls = 0
            def __str__(self):
                raise AssertionError('operator str outside selected domain')
            def __call__(self, *args, **kwargs):
                self.calls += 1
                return 'direct'
        for target, path, phase, training in [
            (dict(kind='component', slot_id='layer0'), 'layer1.experts', -1, True),
            (dict(kind='operator', slot_id='layer0.attention|fwd|aten.mm.default|0'), 'layer1.experts', -1, True),
            (dict(kind='component', slot_id='layer0'), 'layer0', 1, True),
            (dict(kind='operator', slot_id='layer0.attention|fwd|aten.mm.default|0'), 'layer0.attention', 1, True),
            (dict(kind='stage', phase='bwd'), 'layer0.attention', -1, True),
            (dict(kind='stage', phase='fwd'), 'layer0.attention', 1, True),
            (dict(kind='whole'), 'layer0.attention', -1, False),
            (dict(kind='stage', phase='fwd'), 'layer0.attention', -1, False),
            (dict(kind='component', slot_id='layer0'), 'layer0', -1, False),
            (dict(kind='operator', slot_id='layer0.attention|fwd|aten.mm.default|0'), 'layer0.attention', -1, False),
        ]:
            with self.subTest(target=target, phase=phase, training=training):
                state, mode, _, _ = make()
                mode.begin_iteration(40, 1, target)
                state.phase = phase
                direct = DirectOnly()
                with ExitStack() as stack:
                    for module, name in [('flash_v1.scoped_compute', '_tensors'),
                            ('flash_v1.scoped_compute', '_signature'),
                            ('flash_v1.whole_compute', '_tensors'),
                            ('flash_v1.whole_compute', '_signature'),
                            ('flash_v1.whole_compute', 'work_units')]:
                        stack.enter_context(patch(module+'.'+name, side_effect=AssertionError(name)))
                    stack.enter_context(patch.object(WholeComputeSlowdown, 'dispatch', side_effect=AssertionError('super dispatch')))
                    if training: stack.enter_context(mode.training())
                    stack.enter_context(mode.scope(path, 'bwd' if phase >= 0 else 'fwd'))
                    self.assertEqual(mode.dispatch(direct), 'direct')
                self.assertEqual(direct.calls, 1)
                self.assertEqual(mode.end_iteration()['scope_trace_drift'], [])

    def test_forward_hook_during_wrong_autograd_phase_is_not_armed(self):
        state, mode, op, args = make()
        mode.begin_iteration(46, 1, dict(kind='component', slot_id='layer0.attention'))
        state.phase = 1
        with mode.training(), mode.scope('layer0.attention', 'fwd'):
            mode.dispatch(op, args)
        row = mode.end_iteration()
        self.assertEqual(row['scope_trace_drift'], [])
        self.assertTrue(row['selected_scope_missed'])
        self.assertEqual(row['observed_sites'], {})

    def test_component_descendants_and_empty_selected_frame_keep_trace_guard(self):
        _, mode, op, args = make()
        mode.begin_iteration(41, 1, dict(kind='component', slot_id='layer0'))
        with mode.training(), mode.scope('layer0', 'fwd'):
            with mode.scope('layer0.attention', 'fwd'):
                mode.dispatch(op,args);mode.dispatch(op,args)
            # Not in the selected subtree; empty frame must not become drift.
        with mode.training(), mode.scope('layer1.experts', 'fwd'):
            pass
        self.assertTrue(mode.end_iteration()['scope_execution_valid'])
        self.assertEqual(mode.selected_calls, 2)
        mode.begin_iteration(42, 1, dict(kind='component', slot_id='layer0.attention'))
        with mode.training(), mode.scope('layer0.attention','fwd'):
            pass
        row=mode.end_iteration()
        self.assertEqual(row['scope_trace_drift'], ['layer0.attention|fwd'])
        self.assertTrue(row['selected_scope_missed'])

    def test_operator_domain_tracks_full_local_trace_but_not_child_domain(self):
        _,mode,op,args=make()
        mode.begin_iteration(43,1,dict(kind='operator',slot_id='layer0.attention|fwd|aten.mm.default|1'))
        with mode.training(),mode.scope('layer0.attention','fwd'):
            mode.dispatch(op,args)
            with mode.scope('layer0.attention.child','fwd'):
                mode.dispatch(op,args) # bypass, not parent's ordinal
            mode.dispatch(op,args)
        row=mode.end_iteration()
        self.assertTrue(row['scope_execution_valid'])
        self.assertEqual(row['selected_compute_calls'],1)
        self.assertEqual(set(row['observed_sites']),{
            'layer0.attention|fwd|aten.mm.default|0',
            'layer0.attention|fwd|aten.mm.default|1'})

    def test_functional_operator_ordinals_span_two_microbatches(self):
        state, original=setup()
        mode=ScopedComputeSlowdown(original.torch,workload_factory=Workload)
        op,args=Operator(state),(Tensor(4,8),Tensor(8,16))
        mode.begin_calibration()
        with mode.training():
            for _ in range(2):
                mode.dispatch(op,args);mode.dispatch(op,args)
        mode.finish_calibration();mode.freeze_scope_catalog(gpu_numerical_and_coverage_validated=True)
        mode.begin_iteration(44,1,dict(kind='operator',slot_id='<training_functional>|fwd|aten.mm.default|2'))
        with mode.training():
            for _ in range(2):
                with mode.scope('not_functional','fwd'):
                    mode.dispatch(op,args)
                mode.dispatch(op,args);mode.dispatch(op,args)
        row=mode.end_iteration()
        self.assertEqual(row['selected_compute_calls'],1)
        self.assertEqual(set(row['observed_sites']),{
            f'<training_functional>|fwd|aten.mm.default|{i}' for i in range(4)})
        self.assertTrue(row['scope_execution_valid'])

    def test_calibration_catalog_and_selected_fma_match_frozen_previous_source(self):
        import types
        snapshot=Path(__file__).resolve().parents[2]/'results/layer_campaign/scope_guard_v1/scoped_compute_before.py'
        previous=types.ModuleType('flash_v1._scope_before_cpu_test')
        previous.__file__=str(Path(__file__).resolve().parents[1]/'scoped_compute.py')
        exec(compile(snapshot.read_text(), previous.__file__, 'exec'),previous.__dict__)
        old_state,old,old_op,old_args=make(previous.ScopedComputeSlowdown)
        state,new,op,args=make()
        self.assertEqual(old.table,new.table)
        self.assertEqual(old.scope_catalog(),new.scope_catalog())
        self.assertEqual((old_state.events,old_state.syncs),(state.events,state.syncs))
        for mode,operator,arguments in [(old,old_op,old_args),(new,op,args)]:
            mode.begin_iteration(45,1,dict(kind='operator',slot_id='layer0.attention|fwd|aten.mm.default|1'))
            with mode.training():
                with mode.scope('layer1.experts','fwd'):mode.dispatch(operator,arguments)
                with mode.scope('layer0.attention','fwd'):
                    mode.dispatch(operator,arguments);mode.dispatch(operator,arguments)
        for key in ['selected_compute_calls','requested_extra_ms','injected_launches','operators']:
            self.assertEqual(old.end_iteration()[key],new.end_iteration()[key])
        self.assertEqual(old.engine.launches,new.engine.launches)


if __name__ == '__main__':
    unittest.main()
