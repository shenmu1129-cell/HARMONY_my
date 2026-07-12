#!/usr/bin/env bash
set -euo pipefail

GRAPH=${GRAPH:-outputs/persona_chat_cost_aware/persona_chat/graph_50/behavioral_hybrid_graph.json}
SUM=${SUM:-outputs/persona_chat_cost_aware/persona_chat/graph_50/summary_hypernodes.json}
Q=${Q:-outputs/persona_chat_cost_aware/persona_chat/data/questions.jsonl}
OUT=${OUT:-outputs/persona_chat_cost_aware/persona_chat/summary_probe}
MAXQ=${MAXQ:-1000}
mkdir -p "$OUT"

python - <<'PY'
import csv, json, time, os, sys
from pathlib import Path
ROOT=Path.cwd()
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from examples import profile_centric_hypergraph_eval as base
from hypermem.profile_centric_hypergraph import ProfileCentricHypergraphMemory
from hypermem.summary_hypernodes import load_summary_hypernodes, retrieve_summary_hypernodes
from hypermem.cost_aware_retrieval import retrieve_budget_aware

graph=Path(os.environ.get('GRAPH','outputs/persona_chat_cost_aware/persona_chat/graph_50/behavioral_hybrid_graph.json'))
sum_path=Path(os.environ.get('SUM','outputs/persona_chat_cost_aware/persona_chat/graph_50/summary_hypernodes.json'))
q_path=Path(os.environ.get('Q','outputs/persona_chat_cost_aware/persona_chat/data/questions.jsonl'))
out=Path(os.environ.get('OUT','outputs/persona_chat_cost_aware/persona_chat/summary_probe'))
maxq=int(os.environ.get('MAXQ','1000'))
mem=ProfileCentricHypergraphMemory.load(graph)
sums=load_summary_hypernodes(sum_path)
qs=base.normalize_questions(base.read_json_or_jsonl(q_path))[:maxq]
methods=['profile_full','adaptive_tiny','summary_first','summary_adaptive']
rows=[]
for m in methods:
    for q in qs:
        t=time.time()
        if m=='profile_full':
            r=mem.retrieve(q['question'], top_k_edges=3, top_k_facts=8, max_tokens=450, use_utility=False)
        elif m=='adaptive_tiny':
            r=retrieve_budget_aware(mem,q['question'],top_k_edges=2,top_k_facts=4,max_tokens=110,use_utility=False,top_k_topics=2,top_k_episodes=3,budget_ratio=1.0)
        else:
            mode=m
            r=retrieve_summary_hypernodes(mem,q['question'],sums,mode=mode,top_k_summaries=3,top_k_facts=8,max_tokens=158,expand_ratio=0.45)
        row,_,_,_=base.row_from_result(m,q,r,update_used=False)
        row['retrieval_ms']=round((time.time()-t)*1000,3)
        row['selected_summaries']=next((x.get('selected_summaries',0) for x in r.debug_scores if 'selected_summaries' in x),0)
        row['expanded_facts']=next((x.get('expanded_facts',0) for x in r.debug_scores if 'expanded_facts' in x),0)
        rows.append(row)
by={}
for r in rows: by.setdefault(r['method'],[]).append(r)
def avg(rs,k): return sum(float(x.get(k,0)) for x in rs)/max(1,len(rs))
fields=['method','n','accuracy','recall','tokens','retrieval_ms','num_facts','selected_summaries','expanded_facts']
with (out/'summary_probe.csv').open('w',encoding='utf-8',newline='') as f:
    w=csv.DictWriter(f,fieldnames=fields); w.writeheader()
    for m,rs in by.items():
        s=base.summarize(rs)
        w.writerow({'method':m,'n':len(rs),'accuracy':s['accuracy'],'recall':s['recall'],'tokens':s['tokens'],'retrieval_ms':round(avg(rs,'retrieval_ms'),3),'num_facts':round(avg(rs,'num_facts'),3),'selected_summaries':round(avg(rs,'selected_summaries'),3),'expanded_facts':round(avg(rs,'expanded_facts'),3)})
print((out/'summary_probe.csv').read_text(encoding='utf-8'))
PY
