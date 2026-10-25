"""CPU raw replay of the complete preexisting sparse3 cohort; no fit or selection."""
import collections, hashlib, importlib.util, json, sys, time
from pathlib import Path
ROOT=Path('/Users/henry/Desktop/FLASH1022');sys.path.insert(0,str(ROOT))
HERE=Path(__file__).resolve().parent
OLD=ROOT/'results/layer_campaign/stage_forward_v1/sparse3_offline'
BASE=OLD.parent/'sparse_aux_offline';sys.path.insert(0,str(BASE))
spec=importlib.util.spec_from_file_location('raw_cohort_loader',BASE/'replay.py');loader=importlib.util.module_from_spec(spec);spec.loader.exec_module(loader)
from flash_v1 import online_v2, stage_forward_sparse_peer, stage_forward_sampling, stage_forward_peer

def read(p):return json.loads(Path(p).read_text())
def save(p,v):Path(p).write_text(json.dumps(v,indent=2,sort_keys=True)+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
class RedisRows:
    def __init__(self,core,aux):self.core=core;self.aux=aux
    def lrange(self,key,start,end):
        rows=self.aux if 'stage_forward_aux' in key else self.core
        # All transported rows on node0, node1 empty; identity semantics unchanged.
        if key.endswith('node1'):return []
        # Redis limit is per node. Production splits ranks across nodes.
        node=int(key[-1]);rows=[r for r in rows if r['rank']<8] if node==0 else [r for r in rows if r['rank']>=8]
        return [json.dumps(r) for r in rows[start:(None if end==-1 else end+1)]]
# Node assignment needs both rank groups without dropping node1.
class Transport:
    def __init__(self,core,aux):self.core=core;self.aux=aux
    def lrange(self,key,start,end):
        rows=self.aux if 'stage_forward_aux' in key else self.core
        rows=[r for r in rows if (r['rank']<8)==key.endswith('node0')]
        return [json.dumps(r) for r in rows[start:(None if end==-1 else end+1)]]
model=read(ROOT/'results/v2_semantic/linear_candidate_model.json');assert model['sha256'].startswith('dc89');online_v2.verify_model(model)
ACTIVE={};core_counts=collections.Counter()
def predict(rows,start,epoch,unused_core):
    context=rows[0]['stage_metadata']['instrumentation_context'];auxmodel,binding=loader.models[context]
    selected=[dict(r,sampling_contract=stage_forward_sampling.contract(epoch,start)) for r in rows if r['iteration'] in {start,start+2,start+6}]
    aux=stage_forward_sparse_peer.infer_bound(selected,auxmodel,list(range(16)),binding)
    session=ACTIVE.get('session');core=None;public=None;coverage='aux_only_no_matching_raw_core'
    if session:
        directory=ROOT/'results/data16_sessions'/session/'host2';profile=directory/f'online_profile_{epoch}'
        ep=profile/'evidence.json';mp=profile/'evidence_markers.json'
        if ep.exists() and mp.exists():
            core_rows=read(ep);markers=read(mp)
            if {r['iteration'] for r in core_rows}==set(range(start,start+8)):
                out=HERE/'production_replay'/session/f'window_{start}' ;out.mkdir(parents=True,exist_ok=True)
                topology=read(directory/'topology.json');save(out/'topology.json',topology)
                original_config=read(directory/'configuration.json') if (directory/'configuration.json').exists() else {}
                label='plain' if context=='plain' else 'scoped'
                cfg=dict(parallel_layout=rows[0]['stage_metadata']['layout'],monitor_profile_transition=original_config.get('monitor_profile_transition',False),stage_forward_aux_enabled=True,
                    stage_forward_aux_model=str(loader.P/f'model_{label}.json'),stage_forward_aux_context_binding=str(loader.P/f'model_{label}_context_binding.json'),stage_forward_aux_sampling=stage_forward_sampling.CONFIG)
                save(out/'configuration.json',cfg)
                public=online_v2.diagnose(Transport(core_rows+markers,selected),out,dict(start=start,end=start+7,epoch=epoch),model)
                cp=out/f'online_profile_{epoch}/original_dc89.json';core=read(cp) if cp.exists() else None
                coverage='production_diagnose_raw_core_aux' if public is not None else 'production_diagnose_no_result'
    core_counts[coverage]+=1
    return dict(fixed3=dict(aux=loader.compact(aux),core=core,public=public,replay_coverage=coverage))
loader.predict=predict

def run():
    if (HERE/'prediction_freeze.json').exists():raise RuntimeError('preserve frozen prediction')
    manifest=read(OLD/'predictions_without_truth.json')
    data=dict(scopes=manifest['scopes'],sessions=[],canonical_health=[],errors=[])
    # Prior prediction is used ONLY as the immutable exhaustive session/source ledger.
    for prior in manifest['sessions']:
        if not prior:continue
        session=prior['session'];ACTIVE['session']=session
        try:data['sessions'].append(loader.load_session(session))
        except (ValueError,KeyError,FileNotFoundError) as e:data['errors'].append(dict(session=session,reason=str(e)))
    ACTIVE.clear()
    for prior in manifest['canonical_health']:
        path=ROOT/prior['source'];rows=[json.loads(l) for l in path.read_text().splitlines()];start=prior['start'];selected=[r for r in rows if start<=r['iteration']<=prior['end']]
        data['canonical_health'].append(dict(source=prior['source'],source_sha256=sha(path),start=start,end=prior['end'],results=predict(selected,start,selected[0]['epoch'],None)))
    save(HERE/'predictions_without_truth.json',data)
    sources=['flash_v1/online_v2.py','flash_v1/localize_v2_linear.py','flash_v1/stage_forward_sparse_peer.py','flash_v1/stage_forward_sparse_math.py','flash_v1/stage_forward_peer.py','flash_v1/profile_transition.py','results/v2_semantic/linear_candidate_model.json']
    save(HERE/'prediction_freeze.json',dict(status='all_predictions_frozen_before_label_scoring',created_unix=time.time(),predictions_sha256=sha(HERE/'predictions_without_truth.json'),replay_source_sha256=sha(__file__),cohort_ledger_sha256=sha(OLD/'predictions_without_truth.json'),sources={p:sha(ROOT/p) for p in sources},core_model_sha256=model['sha256'],mask=[1,3,7],replay_coverage=dict(core_counts),method='Fresh raw inference; static missing metadata derived by existing approved loader. Production sparse3 infer_bound and online_v2.diagnose with fake Redis transport; original thresholds/min3/fusion. No model training, tuning, or accepted-only selection.'))
    print(json.dumps(dict(predictions_frozen=True,replay_coverage=dict(core_counts))))
if __name__=='__main__':run()
