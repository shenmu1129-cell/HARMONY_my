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
from hypermem.profile_centric_hypergraph import ProfileCentricHypergraphMemory, ProfileRetrievalResult, estimate_tokens, HashedEmbeddingModel
from hypermem.summary_hypernodes import load_summary_hypernodes, retrieve_summary_hypernodes
from hypermem.summary_gate_retrieval import retrieve_summary_gate
from hypermem.cost_aware_retrieval import retrieve_budget_aware

graph=Path(os.environ['GRAPH']); sum_path=Path(os.environ['SUM']); q_path=Path(os.environ['Q']); out=Path(os.environ['OUT'])
maxq=int(os.environ.get('MAXQ','1000'))
mem=ProfileCentricHypergraphMemory.load(graph)
sums=load_summary_hypernodes(sum_path)
qs=base.normalize_questions(base.read_json_or_jsonl(q_path))[:maxq]
methods=['profile_full','adaptive_tiny','summary_first','summary_gate','summary_gate_wide','summary_gate_hybrid']

def pack_ranked_facts(query, facts, top_k=4, max_tokens=110):
    qemb=mem.embedding_model.encode(query)
    seen=set(); ranked=[]
    for f in facts:
        if f.fact_id in seen: continue
        seen.add(f.fact_id)
        score=HashedEmbeddingModel.cosine(qemb, f.embedding)
        ranked.append((f,score))
    ranked.sort(key=lambda x:x[1], reverse=True)
    chosen=[]; toks=0
    for f,score in ranked:
        cost=estimate_tokens(f.content)
        if chosen and toks+cost>max_tokens: continue
        chosen.append(f); toks+=cost
        if len(chosen)>=top_k or toks>=max_tokens: break
    return chosen

def get_result(method,q):
    text=q['question']
    if method=='profile_full':
        return mem.retrieve(text,top_k_edges=3,top_k_facts=8,max_tokens=450,use_utility=False)
    if method=='adaptive_tiny':
        return retrieve_budget_aware(mem,text,top_k_edges=2,top_k_facts=4,max_tokens=110,use_utility=False,top_k_topics=2,top_k_episodes=3,budget_ratio=1.0)
    if method=='summary_first':
        return retrieve_summary_hypernodes(mem,text,sums,mode='summary_first',top_k_summaries=3,top_k_facts=8,max_tokens=158,expand_ratio=0.45)
    if method=='summary_gate':
        return retrieve_summary_gate(mem,text,sums,top_k_summaries=2,top_k_facts=4,max_tokens=110,include_one_summary_hint=False)
    if method=='summary_gate_wide':
        return retrieve_summary_gate(mem,text,sums,top_k_summaries=4,top_k_facts=4,max_tokens=110,include_one_summary_hint=False)
    gate=retrieve_summary_gate(mem,text,sums,top_k_summaries=4,top_k_facts=8,max_tokens=220,include_one_summary_hint=False)
    tiny=retrieve_budget_aware(mem,text,top_k_edges=2,top_k_facts=4,max_tokens=110,use_utility=False,top_k_topics=2,top_k_episodes=3,budget_ratio=1.0)
    facts=pack_ranked_facts(text, list(gate.selected_facts)+list(tiny.selected_facts), top_k=4, max_tokens=110)
    return ProfileRetrievalResult(query=text, channel='summary_gate_hybrid', selected_edges=[], selected_facts=facts, score=1.0, tokens=estimate_tokens([f.content for f in facts]), fallback_used=False, sufficient=bool(facts), debug_scores=[{'path':'summary_gate_hybrid','candidate_summaries':len(sums),'selected_summaries':4,'candidate_facts':len({f.fact_id for f in list(gate.selected_facts)+list(tiny.selected_facts)}),'expanded_facts':len(facts),'token_budget':110}])

rows=[]
for m in methods:
    t0=time.time()
    for q in qs:
        s=time.time(); r=get_result(m,q)
        row,_,_,_=base.row_from_result(m,q,r,update_used=False)
        row['retrieval_ms']=round((time.time()-s)*1000,3)
        ctrl=next((x for x in r.debug_scores if x.get('path') in ('summary_gate','summary_hypernode_controller','summary_gate_hybrid')),{})
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
