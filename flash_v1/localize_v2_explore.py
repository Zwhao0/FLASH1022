"""CPU-only exploratory diagnostics; never imported by online localization.

Inference consumes only raw PROFILE and topology. Labels are read separately
by evaluate(), after predictions have been materialized. Scores are invariant
under consistent rank permutation and never train per-rank classifiers.
"""
from __future__ import annotations
import argparse
from collections import defaultdict, Counter
import gzip
import json
import random
from pathlib import Path
from statistics import median


def read_rows(session):
    rows=[]
    for host in ('host1','host2'):
        with gzip.open(session/host/'raw_profile.jsonl.gz','rt') as stream:
            rows.extend(json.loads(line) for line in stream)
    return rows


def features(rows, topology):
    """Comparable communication durations, with exact matching and no clocks."""
    clique_members=defaultdict(set)
    for row in rows: clique_members[str(row['clique'])].add(row['rank'])
    groups=defaultdict(list)
    for group in topology['groups']:
        groups[tuple(sorted(group['members']))].append(group['name'])
    paired=defaultdict(dict)
    for row in rows:
        members=tuple(sorted(clique_members[str(row['clique'])]))
        names=set(groups[members])
        if len(names)!=1: continue
        family=next(iter(names))
        key=(members,family,row['iteration'],row['phase_code'],row['operation_index'],row['kind'],row['count'],row['dtype'])
        if row['rank'] in paired[key]: raise ValueError('duplicate matched sample')
        paired[key][row['rank']]=row['duration_us']
    indices=defaultdict(set)
    for key in paired: indices[(key[0],key[1],key[3])].add(key[4])
    first={}
    for key, values in indices.items():
        ordered=sorted(values)
        first[key]={index for pos,index in enumerate(ordered) if pos==0 or index!=ordered[pos-1]+1}
    values=defaultdict(list)
    coverage=Counter()
    for key,by_rank in paired.items():
        members,family,iteration,phase,index,*_=key
        if set(by_rank)!=set(members): coverage['incomplete_pairs']+=1;continue
        coverage[family]+=1
        for rank,duration in by_rank.items():
            signed=(sum(by_rank.values())-duration)/(len(members)-1)-duration
            values[rank,f'{family}:phase{phase&255}:all'].append(signed)
            if index in first[(members,family,phase)]:
                values[rank,f'{family}:phase{phase&255}:first'].append(signed)
    result=defaultdict(dict)
    for (rank,name),numbers in values.items(): result[rank][name]=median(numbers)
    return dict(result),dict(coverage)


def predict(rank_features, names, floor_us=1000., ratio=2.):
    scores={rank:max([v for name,v in fs.items() if name in names] or [0.]) for rank,fs in rank_features.items()}
    ordered=sorted(scores,key=lambda rank:scores[rank],reverse=True)
    if len(ordered)<2:return {'rank':None,'scores':scores,'reason':'coverage'}
    winner,second=ordered[:2]
    accepted=scores[winner]>floor_us and scores[winner]>ratio*max(floor_us,scores[second])
    return {'rank':winner if accepted else None,'scores':scores,'reason':'accepted' if accepted else 'ambiguous_or_small'}


VARIANTS={
 'tp_bwd_first':['tensor_parallel:phase2:first'],
 'tp_all_phases':['tensor_parallel:phase%d:%s'%(phase,sel) for phase in (1,2,4) for sel in ('all','first')],
 'tp_dp_all':['%s:phase%d:%s'%(family,phase,sel) for family,phases in [('tensor_parallel',(0,1,2,4)),('dense_world',(0,)),('expert_dp',(0,))] for phase in phases for sel in ('all','first')],
 'tp_pp_all':['tensor_parallel:phase%d:%s'%(phase,sel) for phase in (1,2,4) for sel in ('all','first')]+['pipeline:phase0:all'],
 'pp_only':['pipeline:phase0:all'],
 'dp_only':['%s:phase0:all'%family for family in ('dense_world','expert_dp')],
}


def infer_session(session):
    rows=read_rows(session)
    topology=json.loads((session/'host2/topology.json').read_text())
    events=[json.loads(line) for line in (session/'host2/events.jsonl').read_text().splitlines()]
    windows=[(event['start'],event['end']) for event in events if event.get('event')=='profile_schedule']
    records=[]
    for start,end in windows:
        selected=[row for row in rows if start<=row['iteration']<=end]
        fs,coverage=features(selected,topology)
        records.append(dict(start=start,end=end,features=fs,coverage=coverage,predictions={name:predict(fs,names) for name,names in VARIANTS.items()}))
    return records


def evaluate(session,records):
    """Separate scoring stage; all predictions already fixed before truth opens."""
    plan=json.loads((session/'plan.json').read_text())
    faults=[interval for interval in plan['intervals'] if interval['segment']=='fault']
    cases=[]
    used=set()
    for interval in faults:
        matches=[(idx,rec) for idx,rec in enumerate(records) if min(interval['end'],rec['end'])-max(interval['start'],rec['start'])+1>=4]
        item=dict(target=interval['target_ranks'][0],mode=interval['fault_mode'],start=interval['start'],end=interval['end'],profile_present=bool(matches),predictions={})
        if matches:
            idx,record=matches[0];used.add(idx)
            item['predictions']=record['predictions'];item['window']=[record['start'],record['end']]
        cases.append(item)
    return dict(session=session.name,layout=plan['parallel_layout'],cases=cases,unmatched_windows=[rec for i,rec in enumerate(records) if i not in used])


def summarize(cases,variant):
    counts=Counter(total=len(cases))
    for case in cases:
        if not case['profile_present']: counts['no_profile']+=1;continue
        rank=case['predictions'][variant]['rank']
        counts['correct' if rank==case['target'] else 'rejected' if rank is None else 'wrong']+=1
        scores=case['predictions'][variant]['scores']
        top=max(scores.values())
        winners=[int(key) for key,value in scores.items() if value==top]
        counts['forced_top_correct' if winners==[case['target']] else 'forced_top_wrong_or_tied']+=1
    return dict(counts)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=Path('results'));args=parser.parse_args()
    out=args.root/'localize_v2_explore';out.mkdir(exist_ok=True)
    sessions=[]
    permutation_checks=[]
    for name in ('mixed32_legacy_01','mixed32_node_local_pp_01','whole32_legacy_01','whole32_node_local_pp_01'):
        path=args.root/'data16_sessions'/name
        records=infer_session(path)
        rows=read_rows(path)
        topology=json.loads((path/'host2/topology.json').read_text())
        selected=[row for row in rows if records[0]['start']<=row['iteration']<=records[0]['end']]
        original,_=features(selected,topology)
        permutation=list(range(16));random.Random(9327).shuffle(permutation)
        shuffled=[dict(row,rank=permutation[row['rank']]) for row in selected]
        mapped_topology=dict(topology,groups=[dict(group,members=[permutation[r] for r in group['members']]) for group in topology['groups']])
        remapped,_=features(shuffled,mapped_topology)
        equivariant=all(remapped[permutation[r]]==fs for r,fs in original.items())
        assert equivariant
        permutation_checks.append(dict(session=name,rank_permutation_equivariant=equivariant))
        (out/f'{name}_unlabeled.json').write_text(json.dumps(records,indent=2)+'\n')
        sessions.append(evaluate(path,records))
    summary={variant:{session['session']:summarize(session['cases'],variant) for session in sessions} for variant in VARIANTS}
    modes=sorted(set(case['mode'] for session in sessions for case in session['cases']))
    by_mode={variant:{mode:summarize([case for session in sessions for case in session['cases'] if case['mode']==mode],variant) for mode in modes} for variant in VARIANTS}
    report=dict(metric_definitions={'summary':'Offline case-associated windows only; not end-to-end online accuracy. no_profile kept in all-case denominator. Windows may begin before fault detection; early-trigger mixed legacy first window is retained for diagnostic exploration only.','strict_online_frozen_reference':{'mixed32':{'correct':14,'total':32},'whole32':{'correct':32,'total':32}},'tp_bwd_first':'Exploratory signed-wait proxy with fixed 1000us/2x thresholds; not a replay of frozen production classifier.','conditional_denominator':{'mixed32':31,'whole32':32},'strict_blind_requirement':'All preregistered cases, frozen detector->PROFILE->localizer end-to-end; new independent random sessions.'},permutation_checks=permutation_checks,split_protocol={'fitting':'none; no rank templates; fixed feature variants and thresholds','session_layout_folds':[{'reference_layout':ref,'held_out_layout':held,'sessions':[s['session'] for s in sessions if s['layout']==held]} for ref,held in [('legacy','node_local_pp'),('node_local_pp','legacy')]],'mode_folds':modes,'interpretation':'Whole sessions stay intact. All four sessions are now development evidence, not untouched validation; no threshold/variant is claimed selected on held-out data. Independent new blind sessions required.'},status='development_only_no_independent_blind_claim',floor_us=1000,ratio=2,summary=summary,by_mode=by_mode,sessions=sessions,
      limitations=['Each layout/mode has only one session; windows are correlated, not independent trials.', 'Sessions/layouts/modes are reported separately, with no supervised fitting or tuning on held-out labels.', 'Fixed exploratory thresholds are not calibrated against independent healthy PROFILE.', 'Phase1/4 is a first-communicator-call side effect, not complete per-layer coverage.', 'pipeline features are kind6 AllReduce on pipeline-group cliques, not actual PP Send/Recv; no EP collective/P2P evidence, local compute timing, or synchronized cross-host arrival.'])
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(summary=summary,by_mode=by_mode),indent=2))

if __name__=='__main__':main()
