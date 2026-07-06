#!/usr/bin/env bash
set -euo pipefail

export GRAPH=${GRAPH:-outputs/persona_chat_cost_aware/persona_chat/graph_50/behavioral_hybrid_graph.json}
export SUM=${SUM:-outputs/persona_chat_cost_aware/persona_chat/graph_50/summary_hypernodes.json}
export Q=${Q:-outputs/persona_chat_cost_aware/persona_chat/data/questions.jsonl}
export OUT=${OUT:-outputs/persona_chat_cost_aware/persona_chat/summary_gate_probe}
export MAXQ=${MAXQ:-1000}
mkdir -p "$OUT"

python -m py_compile hypermem/summary_gate_retrieval.py

python - <<'PY'
import csv, json, os, sys, time
from pathlib import Path
ROOT=Path.cwd()
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from examples import profile_centric_hypergraph_eval as base
from hypermem.profile_centric_hypergraph import ProfileCentricHypergraphMemory
from hypermem.summary_hypernodes import load_summary_hypernodes, retrieve_summary_hypernodes
from hypermem.summary_gate_retrieval import retrieve_summary_gate
from hypermem.cost_aware_retrieval import retrieve_budget_aware

graph=Path(os.environ['GRAPH'])
sum_path=Path(os.environ['SUM'])
q_path=Path(os.environ['Q'])
out=Path(os.environ['OUT'])
maxq=int(os.environ.get('MAXQ','1000'))
mem=ProfileCentricHypergraphMemory.load(graph)
sums=load_summary_hypernodes(sum_path)
qs=base.normalize_questions(base.read_json_or_jsonl(q_path))[:maxq]
methods=['profile_full','adaptive_tiny','summary_first','summary_gate','summary_gate_hint']
rows=[]
for m in methods:
    t0=time.time()
    for i,q in enumerate(qs,1):
        s=time.time()
        if m=='profile_full':
            r=mem.retrieve(q['question'],top_k_edges=3,top_k_facts=8,max_tokens=450,use_utility=False)
        elif m=='adaptive_tiny':
            r=retrieve_budget_aware(mem,q['question'],top_k_edges=2,top_k_facts=4,max_tokens=110,use_utility=False,top_k_topics=2,top_k_episodes=3,budget_ratio=1.0)
        elif m=='summary_first':
            r=retrieve_summary_hypernodes(mem,q['question'],sums,mode='summary_first',top_k_summaries=3,top_k_facts=8,max_tokens=158,expand_ratio=0.45)
        elif m=='summary_gate':
            r=retrieve_summary_gate(mem,q['question'],sums,top_k_summaries=2,top_k_facts=4,max_tokens=110,include_one_summary_hint=False)
        else:
            r=retrieve_summary_gate(mem,q['question'],sums,top_k_summaries=1,top_k_facts=3,max_tokens=110,include_one_summary_hint=True)
        row,_,_,_=base.row_from_result(m,q,r,update_used=False)
        row['retrieval_ms']=round((time.time()-s)*1000,3)
        ctrl=next((x for x in r.debug_scores if x.get('path') in ('summary_gate','summary_hypernode_controller')),{})
        row['candidate_summaries']=ctrl.get('candidate_summaries',len(sums) if m.startswith('summary') else 0)
        row['selected_summaries']=ctrl.get('selected_summaries',0)
        row['candidate_facts']=ctrl.get('candidate_facts',len(r.selected_facts))
        row['expanded_facts']=ctrl.get('expanded_facts',0)
        rows.append(row)
    print(f'[done] {m} avg={(time.time()-t0)/max(1,len(qs)):.4f}s/q', flush=True)
by={}
for r in rows: by.setdefault(r['method'],[]).append(r)
def avg(rs,k): return sum(float(x.get(k,0)) for x in rs)/max(1,len(rs))
fields=['method','n','accuracy','recall','tokens','retrieval_ms','num_facts','candidate_summaries','selected_summaries','candidate_facts','expanded_facts']
with (out/'summary_gate_probe.csv').open('w',encoding='utf-8',newline='') as f:
    w=csv.DictWriter(f,fieldnames=fields); w.writeheader()
    for m,rs in by.items():
        s=base.summarize(rs)
        w.writerow({'method':m,'n':len(rs),'accuracy':s['accuracy'],'recall':s['recall'],'tokens':s['tokens'],'retrieval_ms':round(avg(rs,'retrieval_ms'),3),'num_facts':round(avg(rs,'num_facts'),3),'candidate_summaries':round(avg(rs,'candidate_summaries'),3),'selected_summaries':round(avg(rs,'selected_summaries'),3),'candidate_facts':round(avg(rs,'candidate_facts'),3),'expanded_facts':round(avg(rs,'expanded_facts'),3)})
print((out/'summary_gate_probe.csv').read_text(encoding='utf-8'))
PY
