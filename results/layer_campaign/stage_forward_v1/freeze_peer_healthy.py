"""Healthy-only TP2 peer freeze; no native fault/diagnostic inputs are loaded."""
import copy
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from flash_v1.stage_forward_peer import (context_binding,fit,infer_bound,topology_pairs,
    verify_model,verify_topology)


def sha(data):return hashlib.sha256(data).hexdigest()


def main():
    base=Path(__file__).resolve().parent
    output=base/'peer_v1'
    if output.exists():raise FileExistsError('preserve existing peer healthy freeze')
    source_hashes={}
    def read(path,lines=False):
        data=path.read_bytes();source_hashes[str(path.relative_to(ROOT))]=sha(data)
        return [json.loads(line) for line in data.splitlines()] if lines else json.loads(data)
    scoped_manifest=read(base/'health/manifest.json')
    plain_manifest=read(base/'context_amendment/plain_reuse_manifest.json')
    context_proofs={context:read(base/name) for context,name in [
        ('plain','model_plain_work_union_context_binding.json'),
        ('scoped_dispatch','model_scoped_context_binding.json')]}
    read(base/'healthy_model_freeze.json')
    read(base/'context_amendment/freeze.json')
    legacy_derived=base/'derived_revalidated/v2s_stage_forward_v1_fit_legacy_01'
    read(legacy_derived/'revalidation_manifest.json')
    cohorts={'plain':dict(fit=[],calibration=[]),'scoped_dispatch':dict(fit=[],calibration=[])}
    topologies={'plain':{},'scoped_dispatch':{}}
    mappings=[];prepared=[]
    for context,manifest in [('scoped_dispatch',scoped_manifest),('plain',plain_manifest)]:
        for layout,sources in sorted(manifest['by_layout'].items()):
            for phase,source in sources.items():
                session=source['session_id'];rawdir=ROOT/'results/data16_sessions'/session
                topology_path=rawdir/'host2/topology.json';topology=read(topology_path)
                pairs=topology_pairs(topology)
                if layout in topologies[context]:
                    assert topology_pairs(topologies[context][layout])==pairs
                else:topologies[context][layout]=topology
                census_files=list(rawdir.glob('host*/stage_forward_manifest_rank*.json'))
                assert len(census_files)==16
                census_hash={}
                for path in census_files:
                    read(path);census_hash[int(path.stem.split('rank')[-1])]=source_hashes[str(path.relative_to(ROOT))]
                if context=='plain':
                    parent=base/'context_amendment/derived_plain'/session
                    paths=sorted(parent.glob('host*/stage_forward_aux_rank*.jsonl'))
                    selected=[(path,row) for path in paths for row in read(path,True)
                              if source['start']<=row['iteration']<=source['end']]
                elif layout=='legacy' and phase=='fit':
                    paths=sorted(legacy_derived.glob('host*/stage_forward_aux_rank*.jsonl'))
                    selected=[(path,row) for path in paths for row in read(path,True)]
                else:
                    path=base/'health'/f'{phase}_{layout}_rows.json'
                    selected=[(path,row) for row in read(path)]
                cohort=[]
                for path,row in selected:
                    assert source['start']<=row['iteration']<=source['end']
                    assert row['stage_metadata']['layout']==layout
                    declared=row['stage_metadata'].get('instrumentation_context')
                    assert declared in (None,context)
                    pair=next(pair for pair in pairs if row['rank'] in pair)
                    derived=copy.deepcopy(row)
                    derived['stage_metadata']['instrumentation_context']=context
                    derived['stage_metadata']['tp_group_ranks']=pair
                    check=copy.deepcopy(derived)
                    del check['stage_metadata']['tp_group_ranks']
                    if declared is None:del check['stage_metadata']['instrumentation_context']
                    assert check==row
                    original_canonical=json.dumps(row,sort_keys=True,separators=(',',':')).encode()
                    encoded=json.dumps(derived,sort_keys=True,separators=(',',':')).encode()
                    changes=[dict(path='stage_metadata.tp_group_ranks',operation='add',value=pair)]
                    if declared is None:changes.append(dict(path='stage_metadata.instrumentation_context',operation='add',value=context))
                    mappings.append(dict(context=context,layout=layout,phase=phase,session=session,
                        rank=row['rank'],iteration=row['iteration'],source=str(path.relative_to(ROOT)),
                        source_canonical_row_sha256=sha(original_canonical),derived_canonical_row_sha256=sha(encoded),
                        topology_file_sha256=source_hashes[str(topology_path.relative_to(ROOT))],
                        census_sha256=census_hash[row['rank']],context_binding_sha256=context_proofs[context]['sha256'],changes=changes))
                    cohorts[context][phase].append(derived);cohort.append(encoded+b'\n')
                prepared.append((Path('derived_health')/context/layout/(phase+'_rows.jsonl'),b''.join(cohort)))
    models={};audits={};bindings={}
    for context,manifest in [('plain',plain_manifest),('scoped_dispatch',scoped_manifest)]:
        model,audit=fit(cohorts[context],manifest,topologies[context],context)
        verify_model(model)
        for layout,topology in topologies[context].items():verify_topology(model,topology,layout)
        binding=context_binding(model,dict(healthy_context_binding_sha256=context_proofs[context]['sha256'],
            source_files=source_hashes,topology_and_census_attested_all_ranks=True,
            derived_metadata_only=['instrumentation_context','tp_group_ranks']))
        for layout,sources in manifest['by_layout'].items():
            source=sources['calibration']
            for start in range(source['start'],source['end']+1,8):
                rows=[row for row in cohorts[context]['calibration'] if row['stage_metadata']['layout']==layout and start<=row['iteration']<start+8]
                decision=infer_bound(rows,model,list(range(16)),binding)
                assert decision['status']=='rejected' and decision['coverage']['comparable_pair_count']==8
        models[context]=model;audits[context]=audit;bindings[context]=binding
    freeze=dict(status='healthy_only_tp2_peer_frozen_before_native_replay',model_type=models['plain']['model_type'],
        source_files_sha256=source_hashes,implementation_sha256=sha((ROOT/'flash_v1/stage_forward_peer.py').read_bytes()),
        script_sha256=sha(Path(__file__).read_bytes()),relative_floor=.15,support=3,margin=2,
        persistent_support_rule='at least3 distinct matched iterations each cross calibrated signed threshold in candidate direction; median must also cross',
        models={context:dict(model_sha256=model['sha256'],binding_sha256=bindings[context]['sha256'],
            min_score=model['min_score'],max_healthy_calibration_absolute_z=model['max_healthy_calibration_absolute_z'],
            fit_rows=len(cohorts[context]['fit']),calibration_rows=len(cohorts[context]['calibration'])) for context,model in models.items()},
        no_fault_inputs_loaded=True,original_models_and_raw_preserved=True,
        limits=['Plain fit8/cal8 are same-session reused cost windows, not independent validation.',
                'Earlier native healthy comparisons are development evidence; fresh diagnostics remain necessary.',
                'TP2 single-rank faults only; pair-common slowdowns cancel; CPU enqueue bias remains in residuals.'])
    output.mkdir()
    for relative,data in prepared:
        path=output/relative;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
    for context,label in [('plain','plain'),('scoped_dispatch','scoped')]:
        for name,value in [(f'model_{label}.json',models[context]),
            (f'model_{label}_context_binding.json',bindings[context]),(f'healthy_fit_audit_{label}.json',audits[context])]:
            (output/name).write_text(json.dumps(value,sort_keys=True,indent=2)+'\n')
    (output/'derived_metadata_mapping.json').write_text(json.dumps(mappings,indent=2)+'\n')
    (output/'freeze.json').write_text(json.dumps(freeze,indent=2)+'\n')
    print(json.dumps(dict(status=freeze['status'],models=freeze['models'],derived_rows=len(mappings)),indent=2))


if __name__=='__main__':main()
