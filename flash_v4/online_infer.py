"""CPU-only frozen inference, isolated from the actuator/execution process."""
import argparse
import json
from pathlib import Path
import time

from flash_v4.frozen_localizer import features, decide, verify
from flash_v4.health_centered_fusion import fingerprint, write_new
from flash_v4.offline_localizer import _read_observations


def run(root,topology,case,manual=False):
    model_root=Path('results/v4/frozen/r11_kernel_match_v3')
    receipt=json.loads((model_root/'freeze_receipt.json').read_text())
    verify([receipt['model']]+receipt['source_files'])
    model=json.loads((model_root/'model.json').read_text())
    paths=sorted(root.glob('node*/observations/rank_*.jsonl'))
    data=_read_observations(root)
    branches,coverage,cpu_ms=features(data,model['references'][topology])
    result=decide(branches,model['calibration']['by_topology'][topology])
    result.update(case=case,manual_integration_only=manual,prediction_wall_time=time.time(),
                  feature_score_cpu_ms=cpu_ms,coverage_by_rank=coverage,model=receipt['model'],
                  iteration_ids=list(data[0]['iterations']),fault_labels_read=False,
                  input_files=[fingerprint(p) for p in paths],inference_source=fingerprint(Path(__file__)))
    write_new(root/'frozen_prediction.json',result)
    print(json.dumps({k:result[k] for k in ('case','alarm_rank','abstention_reasons','iteration_ids')}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input-root',type=Path,required=True)
    p.add_argument('--topology',choices=('legacy','node_local_pp'),required=True)
    p.add_argument('--case',required=True);p.add_argument('--manual',action='store_true')
    a=p.parse_args();run(a.input_root,a.topology,a.case,a.manual)
