"""Independent fault-only scopes over the existing calibrated FMA injector.

No detector inputs or PROFILE selection are consumed. Integration is deliberately
explicit: wrap the training schedule in training(), and attach module scopes only
for calibration/injection sessions. GPU numerical/coverage validation is required before freezing the catalog.
Components/operators cover FWD only; whole/stage retain FWD and BWD dispatch.
"""
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path

from flash_v1.whole_compute import COMPUTE_OPS, WholeComputeSlowdown, _signature, _tensors


def attest_recorded_catalog(catalog_path, proof_path, output_path=None):
    """CPU-only finalization after external numerical/coverage checks pass.

    Proof JSON must bind catalog_sha256 and contain status='passed',
    numerical_validation='passed', actual_scope_coverage='passed', and
    localizer_outputs_used=false. The proof itself carries actual check evidence;
    this function checks the attestation contract and hashes, not GPU correctness.
    """
    catalog = json.loads(Path(catalog_path).read_text())
    claimed = catalog.pop('catalog_sha256')
    actual = hashlib.sha256(json.dumps(catalog, sort_keys=True).encode()).hexdigest()
    if claimed != actual:
        raise ValueError('recorded scope catalog hash mismatch')
    if 'validation' in catalog:
        raise ValueError('expected raw unvalidated scope catalog')
    for filename, expected in catalog['source_sha256'].items():
        if hashlib.sha256(Path(__file__).with_name(filename).read_bytes()).hexdigest() != expected:
            raise ValueError('recorded scope catalog source mismatch')
    proof_bytes = Path(proof_path).read_bytes()
    proof = json.loads(proof_bytes)
    required = dict(status='passed', numerical_validation='passed',
                    actual_scope_coverage='passed', catalog_sha256=claimed)
    if any(proof.get(key) != value for key, value in required.items()):
        raise ValueError('proof must bind this catalog and pass numerical/actual scope coverage checks')
    if proof.get('localizer_outputs_used') is not False:
        raise ValueError('catalog proof must not use localizer outputs')
    catalog.update(validation='gpu_numerical_and_actual_scope_coverage_passed',
                   raw_catalog_sha256=claimed,
                   validation_proof_sha256=hashlib.sha256(proof_bytes).hexdigest())
    catalog['catalog_sha256'] = hashlib.sha256(json.dumps(catalog, sort_keys=True).encode()).hexdigest()
    if output_path is not None:
        Path(output_path).write_text(json.dumps(catalog, indent=2) + '\n')
    return catalog


class ScopedComputeSlowdown(WholeComputeSlowdown):
    def __init__(self, *args, **kwargs):
        self.training_active = False
        self.scope_stack, self.handles = [], []
        self.declarations, self.population, self.scope_traces = {}, {}, {}
        self.target = None
        self.scope_catalog_frozen = False
        self.active_case_key = None
        self.pending_discard_events = []
        super().__init__(*args, **kwargs)

    def begin_iteration(self, iteration, extra_ratio=0, target=None, case_id=None):
        if extra_ratio and target is None:
            raise ValueError('scoped injection requires an explicit target')
        if target is not None:
            if target.get('kind') not in ('whole', 'stage', 'component', 'operator'):
                raise ValueError('unknown scoped fault kind')
            if target.get('kind') == 'stage' and target.get('phase') not in ('fwd', 'bwd'):
                raise ValueError('stage phase must be fwd or bwd')
            if target.get('kind') in ('component', 'operator') and not target.get('slot_id'):
                raise ValueError('component/operator requires frozen slot_id')
            if extra_ratio and not self.scope_catalog_frozen:
                raise RuntimeError('freeze scope catalog before scoped injection')
            if extra_ratio and target['kind'] == 'operator' and target['slot_id'] not in {row['slot_id'] for row in self.frozen_scope_catalog['operators']}:
                raise ValueError('operator target is outside frozen population')
            if extra_ratio and target['kind'] == 'component':
                allowed = {row['slot_id'] for row in self.frozen_scope_catalog['components'] if row['eligible']}
                if target['slot_id'] not in allowed:
                    raise ValueError('component target is outside frozen eligible population')
        case_key = (json.dumps(dict(case_id=case_id, target=target, ratio=extra_ratio), sort_keys=True)
                    if extra_ratio else None)
        self.iteration_pending_discard = None
        if hasattr(self, 'pending') and case_key != self.active_case_key:
            if any(self.pending.values()):
                self.iteration_pending_discard = dict(iteration=iteration,
                    previous_case_key=self.active_case_key, next_case_key=case_key,
                    signed_ms=sum(self.pending.values()),
                    absolute_ms=sum(abs(value) for value in self.pending.values()),
                    buckets={':'.join(key): value for key, value in self.pending.items()},
                    policy='discard_at_case_boundary_no_flush')
                self.pending_discard_events.append(self.iteration_pending_discard)
            self.pending.clear()
        self.active_case_key = case_key
        self.target = target
        # Resolve the operator's module from the frozen catalog, never from a
        # guessed string prefix or a PROFILE-selected module.
        self.tracking_kind = target['kind'] if extra_ratio else None
        self.tracking_path = None
        self.tracking_phase = None
        if self.tracking_kind == 'operator':
            site = next(row for row in self.frozen_scope_catalog['operators']
                        if row['slot_id'] == target['slot_id'])
            self.tracking_path = site['module_path']
            self.tracking_phase = site['phase']
        elif self.tracking_kind == 'component':
            self.tracking_path, self.tracking_phase = target['slot_id'], 'fwd'
        elif self.tracking_kind == 'stage':
            self.tracking_phase = target['phase']
        self.case_id = case_id
        self.site_counts, self.iteration_sites = {}, {}
        self.selected_calls = 0
        self.trace_drift = []
        super().begin_iteration(iteration, extra_ratio)

    @contextmanager
    def training(self):
        if self.training_active:
            raise RuntimeError('nested training scope')
        self.training_active = True
        try:
            yield self
        finally:
            self.training_active = False
            if self.scope_stack:
                raise RuntimeError('unclosed module scope; catalog/injection invalid')

    @contextmanager
    def scope(self, path, phase):
        """Also usable by explicit semantic wrappers; never infer from PROFILE."""
        frame = dict(path=path, phase=phase, counts={}, trace=[])
        self.scope_stack.append(frame)
        if not self.calibrating and self.extra_ratio:
            frame['trace_armed'] = (self.training_active and self._tracking_domain(phase) and
                                    phase == ('bwd' if self.torch._C._current_graph_task_id() >= 0 else 'fwd'))
        try:
            yield self
        finally:
            if self.scope_stack.pop() is not frame:
                raise RuntimeError('non-nested module scopes')
            self._close_frame(frame)

    def _close_frame(self, frame):
        key = frame['path'] + '|' + frame['phase']
        trace = tuple(frame['trace'])
        if self.calibrating:
            self.scope_traces.setdefault(key, set()).add(trace)
        elif self.extra_ratio and frame.get('trace_armed', False) and trace not in self.scope_traces.get(key, set()):
            # Never reinterpret an ordinal after a dynamic branch changes order.
            self.trace_drift.append(key)

    def attach(self, model):
        """Observe all module FWD calls without replacing inputs or autograd nodes.

        No full-backward hooks: their inserted views change gradient accumulation
        behavior under this Megatron DDP adapter (failed GPU numerical pilot).
        """
        if self.handles:
            raise RuntimeError('already attached')
        for path, module in model.named_modules():
            path = path or '<model>'
            container = type(module).__name__ in ('ModuleList', 'ModuleDict', 'Identity')
            self.declarations[path] = dict(path=path, module_class=type(module).__name__,
                excluded_reason='structural_container_or_identity' if container else None)
            if container:
                continue
            def enter(phase, name=path):
                frame = dict(path=name, phase=phase, counts={}, trace=[])
                self.scope_stack.append(frame)
                if not self.calibrating and self.extra_ratio:
                    frame['trace_armed'] = (self.training_active and self._tracking_domain(phase) and
                                    phase == ('bwd' if self.torch._C._current_graph_task_id() >= 0 else 'fwd'))
            def leave(phase, name=path):
                if not self.scope_stack or self.scope_stack[-1]['path'] != name or self.scope_stack[-1]['phase'] != phase:
                    raise RuntimeError(f'non-nested {phase} module hook: {name}')
                self._close_frame(self.scope_stack.pop())
            def pre(_m, args, kwargs, start=enter):
                if self.training_active:
                    start('fwd')
                # Returning None preserves exact args/kwargs and Tensor identity.
            def post(_m, args, kwargs, output, stop=leave):
                if self.training_active:
                    stop('fwd')
            self.handles.extend([
                module.register_forward_pre_hook(pre, prepend=True, with_kwargs=True),
                module.register_forward_hook(post, with_kwargs=True, always_call=True)])
        return self

    def detach(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def _tracking_domain(self, phase):
        """Cheap scope-only predicate; no operator/tensor/signature inspection."""
        kind = self.tracking_kind
        if kind == 'whole':
            return True
        if kind == 'stage':
            return phase == self.tracking_phase
        if phase != 'fwd':
            return False
        frames = [frame for frame in self.scope_stack if frame['phase'] == phase]
        if kind == 'component':
            return any(frame['path'] == self.tracking_path for frame in frames)
        if kind == 'operator':
            path = frames[-1]['path'] if frames else '<training_functional>'
            return path == self.tracking_path
        return False

    def dispatch(self, func, args=(), kwargs=None):
        kwargs = kwargs or {}
        self.current_site = None
        dispatch_phase = None
        if not self.calibrating:
            if not self.extra_ratio or not self.training_active:
                return func(*args, **kwargs)
            # Components/operators are FWD-only. Reject their module domain
            # before even querying the autograd engine on an unrelated module.
            scoped_fwd = self.tracking_kind in ('component', 'operator')
            if scoped_fwd and not self._tracking_domain('fwd'):
                return func(*args, **kwargs)
            dispatch_phase = 'bwd' if self.torch._C._current_graph_task_id() >= 0 else 'fwd'
            if ((scoped_fwd and dispatch_phase != 'fwd') or
                    (not scoped_fwd and not self._tracking_domain(dispatch_phase))):
                return func(*args, **kwargs)
        parts = str(func).split('.')
        tensors = _tensors((args, kwargs))
        selected = (len(parts) > 1 and parts[0] == 'aten' and parts[1] in COMPUTE_OPS
                    and any(getattr(t.device, 'type', str(t.device)) == 'cuda' for t in tensors))
        self.current_site = None
        if selected and self.training_active and (self.calibrating or self.extra_ratio):
            phase = (dispatch_phase if dispatch_phase is not None else
                     'bwd' if self.torch._C._current_graph_task_id() >= 0 else 'fwd')
            # A hook stack claiming a different autograd phase is invalid evidence.
            frames = [frame for frame in self.scope_stack if frame['phase'] == phase]
            frame = frames[-1] if frames else None
            if frame is not None:
                frame['trace'].append(str(func))
            path = frame['path'] if frame else '<training_functional>'
            counters = frame['counts'] if frame else self.site_counts
            token = (phase, str(func))
            ordinal = counters.get(token, 0)
            counters[token] = ordinal + 1
            slot = f'{path}|{phase}|{func}|{ordinal}'
            ancestors = tuple(frame['path'] for frame in frames)
            signature = repr((_signature(args), _signature(kwargs)))
            self.current_site = dict(slot_id=slot, module_path=path, phase=phase,
                operator=str(func), ordinal=ordinal, ancestors=ancestors)
            if self.calibrating:
                row = self.population.setdefault(slot, dict(**self.current_site, calls=0, signatures=set(), timing_calls={}))
                row['calls'] += 1
                row['signatures'].add(signature)
                timing_key = repr((phase, str(func), _signature(args), _signature(kwargs)))
                row['timing_calls'][timing_key] = row['timing_calls'].get(timing_key, 0) + 1
            self.iteration_sites[slot] = self.iteration_sites.get(slot, 0) + 1
        return super().dispatch(func, args, kwargs)

    def _inject(self, key, op, units, phase):
        site, target = self.current_site, self.target or {'kind': 'whole'}
        kind = target['kind']
        matches = kind == 'whole' and self.training_active and site is not None
        if kind == 'stage':
            matches = bool(site and site['phase'] == target['phase'])
        elif kind == 'component':
            matches = bool(site and site['phase'] == 'fwd' and target['slot_id'] in site['ancestors'])
        elif kind == 'operator':
            matches = bool(site and site['phase'] == 'fwd' and site['slot_id'] == target['slot_id'])
        if matches:
            self.selected_calls += 1
            super()._inject(key, op, units, phase)

    def scope_catalog(self):
        sites = [dict(row, signatures=sorted(row['signatures']), ancestors=list(row['ancestors']))
                 for _, row in sorted(self.population.items()) if row['phase'] == 'fwd']
        for row in sites:
            row['estimated_healthy_total_ms'] = sum(
                self.table[key]['median_ms'] * calls for key, calls in row['timing_calls'].items()
                if key in self.table)
            row['timing_unresolved_calls'] = sum(calls for key, calls in row['timing_calls'].items()
                                                 if key not in self.table)
            row['estimated_healthy_per_call_ms'] = row['estimated_healthy_total_ms'] / row['calls']
        executed = {path for row in sites for path in row['ancestors']}
        components = [dict(row, slot_id=path, eligible=path in executed and not row['excluded_reason'],
                           excluded_reason=row['excluded_reason'] or (None if path in executed else 'no_allowlisted_cuda_compute_observed'))
                      for path, row in sorted(self.declarations.items())]
        for row in components:
            row['estimated_healthy_total_ms'] = sum(site['estimated_healthy_total_ms'] for site in sites
                                                     if row['path'] in site['ancestors'])
        payload = dict(schema=1, rank=self.rank, components=components, operators=sites,
            source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in (Path(__file__), Path(__file__).with_name('whole_compute.py'))},
            scope_traces={key: sorted(list(t) for t in traces) for key, traces in sorted(self.scope_traces.items())},
            phases=['fwd', 'bwd'], component_operator_phases=['fwd'],
            phase_coverage={phase: dict(
                compute_calls=sum(row['calls'] for row in self.population.values() if row['phase'] == phase),
                sites=sum(1 for row in self.population.values() if row['phase'] == phase))
                for phase in ('fwd', 'bwd')}, compute_allowlist=sorted(COMPUTE_OPS),
            population_basis='all_declared_modules_and_all_executed_allowlisted_cuda_FWD_sites_in_healthy_training',
            limitations=['healthy execution catalog, not all possible data dependent branches',
                        'operator ordinals require stable module-local execution order; uncalibrated traces invalidate case, selected miss is reported, never retargeted',
                        'component/operator scopes are FWD only; whole/stage dispatch retain FWD+BWD; no backward module hooks or input replacement',
                        'non-allowlisted CUDA operations and communication excluded; functional sites have no module component',
                        'site times are estimates from shared healthy phase/op/shape CUDA medians, not isolated site timing',
                        'same module called repeatedly shares its local ordinal sites; distinct Python callsites are not distinguished'])
        payload['catalog_sha256'] = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        return payload

    def freeze_scope_catalog(self, *, gpu_numerical_and_coverage_validated=False):
        if not gpu_numerical_and_coverage_validated:
            raise RuntimeError('GPU numerical and actual scope coverage audit required')
        if not self.population:
            raise RuntimeError('empty scoped population')
        self.scope_catalog_frozen = True
        self.frozen_scope_catalog = self.scope_catalog()
        self.frozen_scope_catalog.pop('catalog_sha256')
        self.frozen_scope_catalog['validation'] = 'gpu_numerical_and_actual_scope_coverage_passed'
        self.frozen_scope_catalog['catalog_sha256'] = hashlib.sha256(
            json.dumps(self.frozen_scope_catalog, sort_keys=True).encode()).hexdigest()
        return self.frozen_scope_catalog

    def load_scope_catalog(self, path):
        """Load a validated healthy catalog alongside load_calibration().

        Caller must pass the exact validated artifact from the same rank/source;
        no population is reconstructed from the selected formal case.
        """
        catalog = json.loads(Path(path).read_text())
        claimed = catalog.pop('catalog_sha256')
        actual = hashlib.sha256(json.dumps(catalog, sort_keys=True).encode()).hexdigest()
        if actual != claimed or catalog['rank'] != self.rank:
            raise ValueError('scope catalog hash/rank mismatch')
        for filename, expected in catalog['source_sha256'].items():
            if hashlib.sha256(Path(__file__).with_name(filename).read_bytes()).hexdigest() != expected:
                raise ValueError('scope catalog source mismatch')
        if catalog.get('validation') != 'gpu_numerical_and_actual_scope_coverage_passed':
            raise ValueError('scope catalog is not GPU validated')
        self.population = {row['slot_id']: dict(row, signatures=set(row['signatures']))
                           for row in catalog['operators']}
        self.scope_traces = {key: set(tuple(t) for t in traces)
                             for key, traces in catalog['scope_traces'].items()}
        self.declarations = {row['path']: {key: row[key] for key in
                             ('path', 'module_class', 'excluded_reason')}
                             for row in catalog['components']}
        self.scope_catalog_frozen = True
        self.frozen_scope_catalog = dict(catalog, catalog_sha256=claimed)
        return self.frozen_scope_catalog

    def end_iteration(self):
        return dict(super().end_iteration(), fault_scope=self.target, case_id=self.case_id,
                    pending_discard=self.iteration_pending_discard,
                    pending_discard_events=list(self.pending_discard_events),
                    selected_compute_calls=self.selected_calls,
                    scope_trace_drift=self.trace_drift, scope_execution_valid=not self.trace_drift,
                    selected_scope_missed=bool(self.extra_ratio and not self.selected_calls),
                    observed_sites=self.iteration_sites,
                    scope_catalog_sha256=getattr(self, 'frozen_scope_catalog', {}).get('catalog_sha256'))
