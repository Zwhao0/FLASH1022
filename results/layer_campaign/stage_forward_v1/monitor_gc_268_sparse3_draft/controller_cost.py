"""Cost-only controller: full BEAST/recent3 and core hint computation, two manual windows."""
import hashlib
import json
import sys
from pathlib import Path
from flash_v1 import online,online_v2


def main():
    output=Path(sys.argv[sys.argv.index('--output')+1]);plan=json.loads((output/'plan.json').read_text())
    if not plan.get('monitor_sameprocess_cost') or plan.get('max_profiles')!=3:
        raise ValueError('explicit cost controller contract/max_profiles3 required')
    requests={};sent=set();natural=0;children=0;sample_iterations=[]
    original=online.add_monitor_sample;transition=online.profile_transition_event
    def record(name,row):
        with (output/name).open('a') as stream:stream.write(json.dumps(row,sort_keys=True)+'\n')
    def controlled(detector,iteration,duration,confirm_recent=False):
        nonlocal natural
        sample_iterations.append(iteration)
        candidate=original(detector,iteration,duration,confirm_recent)
        if candidate is not None:
            natural+=1;record('cost_natural_candidates.jsonl',dict(iteration=iteration,candidate=candidate,requested=False))
        if iteration in requests and iteration not in sent:
            sent.add(iteration)
            return dict(iteration=iteration,source='controlled_sameprocess_cost',purpose='C logical39 manual request; no fault input')
        return None
    def no_child(*args,**kwargs):
        nonlocal children
        event=transition(*args,**kwargs)
        if event is not None:
            children+=1;record('cost_ignored_children.jsonl',dict(event=event,requested=False))
        return None
    online.add_monitor_sample=controlled;online.profile_transition_event=no_child
    try:online_v2.main()
    finally:
        online.add_monitor_sample=original;online.profile_transition_event=transition
        (output/'cost_controller_counts.json').write_text(json.dumps(dict(
            full_monitor_candidate_calls=True,monitor_sample_iterations=sample_iterations,monitor_sample_calls=len(sample_iterations),natural_candidates=natural,ignored_child_candidates=children,
            manual_requests_sent=sorted(sent),manual_limit=0,controller_budget=3,
            no_profile_is_not_zero_false_gpu_evidence=True),sort_keys=True)+'\n')


if __name__=='__main__':main()
