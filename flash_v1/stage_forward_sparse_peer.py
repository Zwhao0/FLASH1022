"""Same frozen peer math; runtime rows require explicit sparse3 manifests."""
from .stage_forward_sampling import CONFIG
from .stage_forward_sparse_math import infer_sparse

def infer_bound(rows,model,expected_ranks,binding):
    rows=list(rows)
    if not rows:return dict(status='rejected',rank=None,reason='missing_sparse_auxiliary')
    manifest=rows[0].get('sampling_contract')
    if (not isinstance(manifest,dict) or {key:manifest.get(key) for key in CONFIG}!=CONFIG
            or type(manifest.get('start')) is not int or manifest.get('end')!=manifest['start']+7
            or any(row.get('sampling_contract')!=manifest or row.get('epoch')!=manifest.get('epoch') for row in rows)):
        return dict(status='rejected',rank=None,reason='missing_or_incompatible_sparse_sampling_contract')
    result=infer_sparse(rows,model,expected_ranks,binding,dict(schema='offline_sparse_aux_v1',window_size=8,mask='fixed3',start=manifest['start'],epoch=manifest['epoch']))
    result['sampling_contract']=dict(manifest,unobserved_offsets=[2,4,5,6,8],unobserved_is_healthy=False)
    return result
