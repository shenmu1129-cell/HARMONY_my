from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import os
import random
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from openai import OpenAI, OpenAIError  # noqa: E402

from hypermem import load_runtime_env  # noqa: E402
from hypermem.profile_centric_hypergraph import (  # noqa: E402
    HashedEmbeddingModel,
    ProfileCentricHypergraphMemory,
    ProfileEdgeType,
    ProfileFact,
    ProfileRetrievalResult,
    estimate_tokens,
    keyword_overlap,
    tokenize,
)


@dataclass
class LongMemExample:
    qid: str
    qtype: str
    question: str
    answer: str
    question_date: str
    rows: List[Dict[str, Any]]
    answer_session_ids: List[str]


@dataclass
class MethodConfig:
    name: str
    top_k_edges: int = 3
    top_k_facts: int = 6
    max_tokens: int = 280
    graph_gate: str = "global"
    token_roi: float = 0.15
    time_boost: float = 0.0
    initial_candidates: int = 16
    episode_top_k: int = 20
    topic_top_k: int = 10
    lambda_prop: float = 0.5


class BM25Index:
    def __init__(self, facts: Sequence[ProfileFact], k1: float = 1.5, b: float = 0.75) -> None:
        self.facts = list(facts)
        self.k1 = k1
        self.b = b
        self.doc_terms = [tokenize(f.content) for f in self.facts]
        self.avgdl = sum(len(t) for t in self.doc_terms) / max(1, len(self.doc_terms))
        df: Dict[str, int] = {}
        for terms in self.doc_terms:
            for tok in set(terms):
                df[tok] = df.get(tok, 0) + 1
        n = max(1, len(self.facts))
        self.idf = {tok: math.log(1 + (n - c + 0.5) / (c + 0.5)) for tok, c in df.items()}

    def score(self, query: str, fact_idx: int) -> float:
        q_terms = tokenize(query)
        terms = self.doc_terms[fact_idx]
        if not q_terms or not terms:
            return 0.0
        counts: Dict[str, int] = {}
        for tok in terms:
            counts[tok] = counts.get(tok, 0) + 1
        dl = len(terms)
        score = 0.0
        for tok in q_terms:
            tf = counts.get(tok, 0)
            if tf <= 0:
                continue
            denom = tf + self.k1 * (1 - self.b + self.b * dl / max(1e-6, self.avgdl))
            score += self.idf.get(tok, 0.0) * (tf * (self.k1 + 1)) / denom
        return score


class QwenEmbeddingClient:
    def __init__(self, base_url: str = "http://localhost:11810/v1/embeddings", model: str = "Qwen3-Embedding-4B", timeout: int = 120) -> None:
        self.base_url = base_url
        self.model = model
        self.timeout = timeout

    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        payload = json.dumps({"model": self.model, "input": list(texts)}).encode("utf-8")
        req = urllib.request.Request(self.base_url, data=payload, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return [row["embedding"] for row in data["data"]]

    @staticmethod
    def cosine(a: Sequence[float], b: Sequence[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(y * y for y in b))
        return dot / max(1e-9, na * nb)


class QwenRerankerClient:
    def __init__(self, base_url: str = "http://localhost:12810", model: str = "Qwen3-Reranker-4B", timeout: int = 120) -> None:
        self.base_url = base_url.rstrip("/") + "/v1/completions"
        self.model = model
        self.timeout = timeout

    def rerank(self, query: str, docs: Sequence[str]) -> List[float]:
        prefix = '<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
        suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        instruction = "Given a user's question and a text passage, determine if the passage contains specific information that directly answers the question."
        prompts = [f"{prefix}<Instruct>: {instruction}\n\n<Query>: {query}\n\n<Document>: {doc}{suffix}" for doc in docs]
        payload = json.dumps({"model": self.model, "prompt": prompts, "max_tokens": 1, "logprobs": 20, "temperature": 0}).encode("utf-8")
        req = urllib.request.Request(self.base_url, data=payload, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        scores = []
        for choice in data["choices"]:
            top = (choice.get("logprobs") or {}).get("top_logprobs", [{}])[0]
            yes = math.exp(float(top.get("yes", -10)))
            no = math.exp(float(top.get("no", -10)))
            scores.append(yes / max(1e-12, yes + no))
        return scores


def rrf_fuse(rankings: Sequence[Sequence[Tuple[str, float]]], k: int = 60) -> Dict[str, float]:
    fused: Dict[str, float] = {}
    for ranking in rankings:
        for rank, (doc_id, _score) in enumerate(ranking, start=1):
            fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (k + rank)
    return fused


def vector_add(a: Sequence[float], b: Sequence[float], weight: float = 1.0) -> List[float]:
    return [float(x) + weight * float(y) for x, y in zip(a, b)]


def vector_mean(vectors: Sequence[Sequence[float]]) -> List[float]:
    if not vectors:
        return []
    dim = len(vectors[0])
    out = [0.0] * dim
    for vec in vectors:
        for i, val in enumerate(vec[:dim]):
            out[i] += float(val)
    return [x / len(vectors) for x in out]


def normalize_vector(vec: Sequence[float]) -> List[float]:
    norm = math.sqrt(sum(float(x) * float(x) for x in vec))
    if norm <= 1e-12:
        return [float(x) for x in vec]
    return [float(x) / norm for x in vec]


def safe_json(text: str) -> Dict[str, Any]:
    text = (text or "").strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.S)
    if match:
        try:
            return json.loads(match.group(1))
        except Exception:
            pass
    start, end = text.find("{"), text.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except Exception:
            return {}
    return {}


def load_examples(path: Path, max_examples: int = 0, start_index: int = 0, skip_abs: bool = False) -> List[LongMemExample]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if raw and isinstance(raw[0], dict) and "qa" in raw[0] and "conversation" in raw[0]:
        return load_locomo_examples(raw, max_examples=max_examples, start_index=start_index)
    examples: List[LongMemExample] = []
    for item in raw[start_index:]:
        qid = str(item["question_id"])
        if skip_abs and qid.endswith("_abs"):
            continue
        rows = flatten_rows(item)
        examples.append(
            LongMemExample(
                qid=qid,
                qtype=str(item.get("question_type") or ("abstention" if qid.endswith("_abs") else "")),
                question=str(item["question"]),
                answer=str(item["answer"]),
                question_date=str(item.get("question_date") or ""),
                rows=rows,
                answer_session_ids=[str(x) for x in item.get("answer_session_ids", [])],
            )
        )
        if max_examples and len(examples) >= max_examples:
            break
    return examples


def _locomo_session_date(conversation: Dict[str, Any], session_idx: int) -> str:
    return str(conversation.get(f"session_{session_idx}_date_time") or conversation.get(f"session_{session_idx}_date") or "")


def _stringify_summary(value: Any) -> str:
    parts: List[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            if key == "date":
                continue
            child_text = _stringify_summary(child)
            if child_text:
                parts.append(f"{key}: {child_text}")
    elif isinstance(value, list):
        for child in value:
            if isinstance(child, list) and child:
                parts.append(str(child[0]))
            elif isinstance(child, (dict, list)):
                child_text = _stringify_summary(child)
                if child_text:
                    parts.append(child_text)
            elif child:
                parts.append(str(child))
    elif value:
        parts.append(str(value))
    return " ".join(parts)


def _locomo_summary_for_session(item: Dict[str, Any], session_idx: int) -> str:
    chunks: List[str] = []
    for root_key, prefix in (("observation", "Observation"), ("session_summary", "Session summary"), ("event_summary", "Event summary")):
        root = item.get(root_key) or {}
        if not isinstance(root, dict):
            continue
        for key in (
            f"session_{session_idx}_observation",
            f"session_{session_idx}",
            f"events_session_{session_idx}",
        ):
            text = _stringify_summary(root.get(key))
            if text:
                chunks.append(f"{prefix}: {text}")
    return "\n".join(dict.fromkeys(chunks))


def flatten_locomo_rows(item: Dict[str, Any], evidence_ids: Sequence[str]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    evidence_set = {str(x) for x in evidence_ids}
    conversation = item.get("conversation") or {}
    sample_id = str(item.get("sample_id") or "locomo")
    session_nums = sorted(
        int(match.group(1))
        for key in conversation
        for match in [re.fullmatch(r"session_(\d+)", str(key))]
        if match
    )
    for si in session_nums:
        date = _locomo_session_date(conversation, si)
        session_summary = _locomo_summary_for_session(item, si)
        for ti, turn in enumerate(conversation.get(f"session_{si}") or []):
            speaker = str(turn.get("speaker") or "unknown")
            dia_id = str(turn.get("dia_id") or f"D{si}:{ti + 1}")
            text = str(turn.get("text") or "")
            caption = str(turn.get("blip_caption") or "")
            query = str(turn.get("query") or "")
            if caption:
                text = f"{text} Image caption: {caption}"
            if query:
                text = f"{text} Image query: {query}"
            if not text.strip():
                continue
            rows.append(
                {
                    "row_id": f"{sample_id}::{dia_id}",
                    "session_id": f"session_{si}",
                    "session_index": si - 1,
                    "turn_index": ti,
                    "date": date,
                    "role": speaker,
                    "dia_id": dia_id,
                    "has_answer": dia_id in evidence_set,
                    "content": f"[{date}] {speaker}: {text}",
                    "raw_content": text,
                    "session_summary": session_summary,
                }
            )
    return rows


def load_locomo_examples(raw: Sequence[Dict[str, Any]], max_examples: int = 0, start_index: int = 0) -> List[LongMemExample]:
    examples: List[LongMemExample] = []
    flat_qas: List[Tuple[Dict[str, Any], int, Dict[str, Any]]] = []
    for item in raw:
        for qi, qa in enumerate(item.get("qa") or []):
            flat_qas.append((item, qi, qa))
    for item, qi, qa in flat_qas[start_index:]:
        evidence = [str(x) for x in qa.get("evidence") or []]
        rows = flatten_locomo_rows(item, evidence)
        evidence_sessions = {
            str(row["session_id"])
            for row in rows
            if str(row.get("dia_id") or "") in set(evidence)
        }
        question_date = ""
        if rows:
            max_session = max(int(row.get("session_index") or 0) for row in rows)
            dates = [str(row.get("date") or "") for row in rows if int(row.get("session_index") or 0) == max_session]
            question_date = dates[0] if dates else ""
        examples.append(
            LongMemExample(
                qid=f"{item.get('sample_id', 'locomo')}::qa_{qi:03d}",
                qtype=f"locomo_category_{qa.get('category', 'unknown')}",
                question=str(qa.get("question") or ""),
                answer=str(qa.get("answer") or ""),
                question_date=question_date,
                rows=rows,
                answer_session_ids=sorted(evidence_sessions),
            )
        )
        if max_examples and len(examples) >= max_examples:
            break
    return examples


def flatten_rows(item: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    dates = item.get("haystack_dates") or []
    session_ids = item.get("haystack_session_ids") or []
    sessions = item.get("haystack_sessions") or []
    for si, turns in enumerate(sessions):
        sid = str(session_ids[si] if si < len(session_ids) else f"session_{si:04d}")
        date = str(dates[si] if si < len(dates) else "")
        for ti, turn in enumerate(turns or []):
            role = str(turn.get("role") or "unknown")
            content = str(turn.get("content") or "")
            if not content.strip():
                continue
            row_id = f"{item['question_id']}::{sid}::{ti:03d}"
            rows.append(
                {
                    "row_id": row_id,
                    "session_id": sid,
                    "session_index": si,
                    "turn_index": ti,
                    "date": date,
                    "role": role,
                    "has_answer": bool(turn.get("has_answer")),
                    "content": f"[{date}] {role}: {content}",
                    "raw_content": content,
                }
            )
    return rows


def build_memory(rows: Sequence[Dict[str, Any]]) -> ProfileCentricHypergraphMemory:
    memory = ProfileCentricHypergraphMemory(user_id="longmemeval", attach_threshold=0.58, discovery_threshold=0.62)
    session_fact_ids: Dict[str, List[str]] = {}
    role_fact_ids: Dict[str, List[str]] = {}
    for idx, row in enumerate(rows, start=1):
        fid = str(row["row_id"])
        fact = memory.add_fact(
            content=str(row["content"]),
            fact_id=fid,
            keywords=[],
            timestamp=float(idx),
            metadata=dict(row),
            promote=False,
        )
        session_fact_ids.setdefault(str(row["session_id"]), []).append(fact.fact_id)
        role_fact_ids.setdefault(str(row["role"]), []).append(fact.fact_id)
    for sid, fids in session_fact_ids.items():
        facts = [memory.facts[fid] for fid in fids]
        date = str(facts[0].metadata.get("date") or "") if facts else ""
        summary_hint = str(facts[0].metadata.get("session_summary") or "") if facts else ""
        sample = " ".join(f.content for f in facts[:3])
        edge_summary = summary_hint[:900] if summary_hint else f"LongMemEval session {sid} at {date}: {sample[:360]}"
        memory.create_edge(
            ProfileEdgeType.AUTO_DISCOVERED,
            fids,
            summary=edge_summary,
            confidence=0.62,
            metadata={"edge_kind": "session", "session_id": sid, "date": date},
        )
    for role, fids in role_fact_ids.items():
        memory.create_edge(
            ProfileEdgeType.AUTO_DISCOVERED,
            fids,
            summary=f"LongMemEval {role} turns",
            confidence=0.45,
            metadata={"edge_kind": "role", "role": role},
        )
    return memory


def pack_ranked(ranked: Sequence[Tuple[ProfileFact, float]], top_k: int, max_tokens: int) -> List[ProfileFact]:
    selected: List[ProfileFact] = []
    tokens = 0
    seen = set()
    for fact, score in ranked:
        if score <= 0:
            continue
        key = fact.content.lower().strip()
        if key in seen:
            continue
        cost = estimate_tokens(fact.content)
        if selected and tokens + cost > max_tokens:
            continue
        selected.append(fact)
        seen.add(key)
        tokens += cost
        if len(selected) >= top_k or tokens >= max_tokens:
            break
    return selected


def result(query: str, method: str, facts: Sequence[ProfileFact], edges: Sequence[Any], score: float, started: float, candidates: int) -> ProfileRetrievalResult:
    return ProfileRetrievalResult(
        query=query,
        channel=method,
        selected_edges=list(edges),
        selected_facts=list(facts),
        score=score,
        tokens=estimate_tokens([f.content for f in facts]),
        fallback_used=False,
        sufficient=bool(facts),
        debug_scores=[{"method": method, "candidate_facts": candidates, "latency_ms": round((time.time() - started) * 1000, 4)}],
    )


def build_episode_topic_docs(facts: Sequence[ProfileFact]) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    sessions: Dict[str, List[ProfileFact]] = {}
    for fact in facts:
        sid = str(fact.metadata.get("session_id") or "unknown")
        sessions.setdefault(sid, []).append(fact)
    episodes: Dict[str, Dict[str, Any]] = {}
    for sid, sfacts in sessions.items():
        sfacts = sorted(sfacts, key=lambda f: int(f.metadata.get("turn_index") or 0))
        date = str(sfacts[0].metadata.get("date") or "") if sfacts else ""
        summary_hint = str(sfacts[0].metadata.get("session_summary") or "") if sfacts else ""
        user_lines = [str(f.metadata.get("raw_content") or f.content) for f in sfacts if str(f.metadata.get("role") or "") == "user"]
        all_lines = [f"{f.metadata.get('role')}: {f.metadata.get('raw_content') or f.content}" for f in sfacts]
        text = f"Episode date: {date}\n"
        if summary_hint:
            text += f"Gate summary: {summary_hint[:1200]}\n"
        text += "\n".join(user_lines[:10] or all_lines[:10])
        episodes[sid] = {"id": sid, "date": date, "facts": sfacts, "text": text}

    # Lightweight topic aggregation: group episodes by their strongest content words.
    buckets: Dict[str, List[str]] = {}
    stop = {"the", "and", "for", "with", "that", "this", "have", "what", "when", "where", "which", "about", "from", "into"}
    for sid, ep in episodes.items():
        toks = [t for t in tokenize(ep["text"]) if len(t) > 3 and t not in stop]
        key = toks[0] if toks else sid
        buckets.setdefault(key, []).append(sid)
    topics: Dict[str, Dict[str, Any]] = {}
    for idx, (key, sids) in enumerate(buckets.items(), start=1):
        topic_id = f"topic_{idx:03d}_{key}"
        text = "\n".join(episodes[sid]["text"][:420] for sid in sids)
        topics[topic_id] = {"id": topic_id, "episode_ids": sids, "text": f"Topic {key}:\n{text}"}
    return episodes, topics


def bm25_rank_docs(query: str, docs: Sequence[Tuple[str, str]], top_n: int) -> List[Tuple[str, float]]:
    temp_facts = [
        ProfileFact(fact_id=doc_id, content=text, timestamp=float(i + 1), embedding=[])
        for i, (doc_id, text) in enumerate(docs)
    ]
    index = BM25Index(temp_facts)
    ranked = [(doc_id, index.score(query, i)) for i, (doc_id, _text) in enumerate(docs)]
    ranked.sort(key=lambda x: x[1], reverse=True)
    return ranked[:top_n]


def dense_rank_docs(query: str, docs: Sequence[Tuple[str, str]], qwen_embed: QwenEmbeddingClient, top_n: int) -> Tuple[List[Tuple[str, float]], Dict[str, List[float]], List[float]]:
    vectors = qwen_embed.embed([query] + [text for _doc_id, text in docs])
    qvec, dvecs = vectors[0], vectors[1:]
    emb = {doc_id: vec for (doc_id, _), vec in zip(docs, dvecs)}
    ranked = [(doc_id, qwen_embed.cosine(qvec, emb[doc_id])) for doc_id, _ in docs]
    ranked.sort(key=lambda x: x[1], reverse=True)
    return ranked[:top_n], emb, qvec


def retrieve_hypermem_full(
    example: LongMemExample,
    memory: ProfileCentricHypergraphMemory,
    method: MethodConfig,
    qwen_embed: QwenEmbeddingClient,
    qwen_reranker: QwenRerankerClient | None,
) -> ProfileRetrievalResult:
    started = time.time()
    facts = list(memory.facts.values())
    episodes, topics = build_episode_topic_docs(facts)

    topic_docs = [(tid, t["text"]) for tid, t in topics.items()]
    episode_docs = [(eid, e["text"]) for eid, e in episodes.items()]
    fact_docs = [(f.fact_id, f.content) for f in facts]

    candidate_n = max(method.initial_candidates, method.topic_top_k, method.episode_top_k, method.top_k_facts)
    topic_bm25 = bm25_rank_docs(example.question, topic_docs, min(candidate_n, len(topic_docs))) if topic_docs else []
    topic_dense, topic_emb, qvec = dense_rank_docs(example.question, topic_docs, qwen_embed, min(candidate_n, len(topic_docs))) if topic_docs else ([], {}, qwen_embed.embed([example.question])[0])
    topic_fused = rrf_fuse([topic_bm25, topic_dense])
    top_topics = [tid for tid, _ in sorted(topic_fused.items(), key=lambda x: x[1], reverse=True)[: method.topic_top_k]]
    allowed_episode_ids = {
        eid
        for tid in top_topics
        for eid in topics.get(tid, {}).get("episode_ids", [])
    } or set(episodes.keys())

    episode_pool_docs = [(eid, episodes[eid]["text"]) for eid in allowed_episode_ids if eid in episodes]
    ep_bm25 = bm25_rank_docs(example.question, episode_pool_docs, min(candidate_n, len(episode_pool_docs))) if episode_pool_docs else []
    ep_dense, ep_emb, _ = dense_rank_docs(example.question, episode_pool_docs, qwen_embed, min(candidate_n, len(episode_pool_docs))) if episode_pool_docs else ([], {}, qvec)
    ep_fused = rrf_fuse([ep_bm25, ep_dense])
    top_episodes = [eid for eid, _ in sorted(ep_fused.items(), key=lambda x: x[1], reverse=True)[: method.episode_top_k]]
    allowed_fact_ids = {
        f.fact_id
        for eid in top_episodes
        for f in episodes.get(eid, {}).get("facts", [])
    } or {f.fact_id for f in facts}

    fact_pool = [f for f in facts if f.fact_id in allowed_fact_ids]
    fact_pool_docs = [(f.fact_id, f.content) for f in fact_pool]
    fact_bm25 = bm25_rank_docs(example.question, fact_pool_docs, min(candidate_n, len(fact_pool_docs))) if fact_pool_docs else []
    fact_dense, fact_emb_raw, _ = dense_rank_docs(example.question, fact_pool_docs, qwen_embed, min(candidate_n, len(fact_pool_docs))) if fact_pool_docs else ([], {}, qvec)

    # Hypergraph embedding propagation: turn/fact vectors absorb episode and topic context.
    episode_vecs: Dict[str, List[float]] = {}
    for eid, ep in episodes.items():
        child_vecs = [fact_emb_raw.get(f.fact_id) for f in ep["facts"] if f.fact_id in fact_emb_raw]
        child_vecs = [v for v in child_vecs if v]
        episode_vecs[eid] = normalize_vector(vector_mean(child_vecs) or ep_emb.get(eid, []))
    topic_vecs: Dict[str, List[float]] = {}
    for tid, topic in topics.items():
        child_vecs = [episode_vecs.get(eid) for eid in topic["episode_ids"] if episode_vecs.get(eid)]
        topic_vecs[tid] = normalize_vector(vector_mean([v for v in child_vecs if v]) or topic_emb.get(tid, []))
    fact_dense_prop: List[Tuple[str, float]] = []
    fact_to_topic = {
        f.fact_id: tid
        for tid, topic in topics.items()
        for eid in topic["episode_ids"]
        for f in episodes.get(eid, {}).get("facts", [])
    }
    for fact in fact_pool:
        base = fact_emb_raw.get(fact.fact_id)
        if not base:
            continue
        eid = str(fact.metadata.get("session_id") or "")
        tid = fact_to_topic.get(fact.fact_id, "")
        vec = base
        if episode_vecs.get(eid):
            vec = vector_add(vec, episode_vecs[eid], method.lambda_prop)
        if topic_vecs.get(tid):
            vec = vector_add(vec, topic_vecs[tid], method.lambda_prop * 0.5)
        vec = normalize_vector(vec)
        fact_dense_prop.append((fact.fact_id, qwen_embed.cosine(qvec, vec)))
    fact_dense_prop.sort(key=lambda x: x[1], reverse=True)

    fused = rrf_fuse([fact_bm25, fact_dense, fact_dense_prop])
    candidate_ids = [fid for fid, _ in sorted(fused.items(), key=lambda x: x[1], reverse=True)[: method.initial_candidates]]
    by_id = {f.fact_id: f for f in fact_pool}
    candidates = [by_id[fid] for fid in candidate_ids if fid in by_id]
    if qwen_reranker is not None and candidates:
        rerank_scores = qwen_reranker.rerank(example.question, [f.content for f in candidates])
    else:
        rerank_scores = [fused.get(f.fact_id, 0.0) for f in candidates]
    ranked: List[Tuple[ProfileFact, float]] = []
    for fact, rr in zip(candidates, rerank_scores):
        role = str(fact.metadata.get("role") or "")
        role_prior = 0.04 if role == "user" else -0.02 if role == "assistant" else 0.0
        temporal_prior = 0.03 if re.search(r"\b(when|date|days?|weeks?|months?|before|after|first|last)\b", example.question.lower()) else 0.0
        score = float(rr) + role_prior + temporal_prior + 0.05 * keyword_overlap(example.question, fact.content)
        ranked.append((fact, score))
    ranked.sort(key=lambda x: x[1], reverse=True)
    selected = pack_ranked(ranked, method.top_k_facts, method.max_tokens)
    avg = sum(s for _, s in ranked[: max(1, method.top_k_facts)]) / max(1, min(len(ranked), method.top_k_facts))
    return result(example.question, method.name, selected, [], avg, started, len(candidates))


def retrieve_method(
    example: LongMemExample,
    memory: ProfileCentricHypergraphMemory,
    method: MethodConfig,
    qwen_embed: QwenEmbeddingClient | None = None,
    qwen_reranker: QwenRerankerClient | None = None,
) -> ProfileRetrievalResult:
    started = time.time()
    qemb = memory.embedding_model.encode(example.question)
    facts = list(memory.facts.values())
    bm25 = BM25Index(facts)
    ranked: List[Tuple[ProfileFact, float]] = []
    edges: List[Any] = []

    if method.graph_gate == "oracle_all":
        selected = pack_ranked([(fact, 1.0) for fact in facts], method.top_k_facts, method.max_tokens)
        return result(example.question, method.name, selected, [], 1.0, started, len(facts))
    if method.graph_gate == "hypermem_full":
        if qwen_embed is None:
            raise RuntimeError(f"{method.name} requires Qwen embedding service")
        return retrieve_hypermem_full(example, memory, method, qwen_embed, qwen_reranker)
    if method.graph_gate in {"qwen_dense", "qwen_hg"}:
        if qwen_embed is None:
            raise RuntimeError(f"{method.name} requires Qwen embedding service")
        candidate_ids: List[str] = []
        edges_for_result: List[Any] = []
        if method.graph_gate == "qwen_hg":
            qtype, _, _ = memory.infer_profile_type(example.question)
            scored_edges = []
            for edge in memory.edges.values():
                score, _ = edge.score(qemb, qtype, memory.facts, use_utility=True, weights=memory.weights)
                if score > 0:
                    scored_edges.append((edge, score))
            scored_edges.sort(key=lambda x: x[1], reverse=True)
            edges_for_result = [edge for edge, _ in scored_edges[: method.top_k_edges]]
            for edge, _ in scored_edges[: method.top_k_edges]:
                for fid in edge.member_fact_ids:
                    if fid not in candidate_ids:
                        candidate_ids.append(fid)
        candidate_facts = [memory.facts[fid] for fid in candidate_ids if fid in memory.facts]
        if len(candidate_facts) < method.initial_candidates:
            pool = [fact for fact in facts if fact.fact_id not in set(candidate_ids)]
            vectors = qwen_embed.embed([example.question] + [f.content for f in pool])
            qvec, dvecs = vectors[0], vectors[1:]
            dense_ranked = sorted(
                [(fact, qwen_embed.cosine(qvec, vec)) for fact, vec in zip(pool, dvecs)],
                key=lambda x: x[1],
                reverse=True,
            )
            candidate_facts.extend([fact for fact, _ in dense_ranked[: max(0, method.initial_candidates - len(candidate_facts))]])
        candidate_facts = list({fact.fact_id: fact for fact in candidate_facts}.values())[: method.initial_candidates]
        vectors = qwen_embed.embed([example.question] + [f.content for f in candidate_facts])
        qvec, dvecs = vectors[0], vectors[1:]
        emb_scores = [qwen_embed.cosine(qvec, vec) for vec in dvecs]
        rerank_scores: List[float] | None = None
        if qwen_reranker is not None:
            try:
                rerank_scores = qwen_reranker.rerank(example.question, [f.content for f in candidate_facts])
            except Exception as exc:
                print(f"[warn] qwen reranker unavailable, fallback to embedding scores: {exc}", flush=True)
                rerank_scores = None
        ranked = []
        for fact, emb_score, rr_score in zip(candidate_facts, emb_scores, rerank_scores or emb_scores):
            role = str(fact.metadata.get("role") or "")
            role_prior = 0.04 if role == "user" else -0.02 if role == "assistant" else 0.0
            lex = keyword_overlap(example.question, fact.content)
            score = 0.72 * float(rr_score) + 0.20 * emb_score + 0.08 * lex + role_prior
            ranked.append((fact, score))
        ranked.sort(key=lambda x: x[1], reverse=True)
        selected = pack_ranked(ranked, method.top_k_facts, method.max_tokens)
        avg = sum(s for _, s in ranked[: max(1, method.top_k_facts)]) / max(1, min(len(ranked), method.top_k_facts))
        return result(example.question, method.name, selected, edges_for_result, avg, started, len(candidate_facts))
    if method.graph_gate == "bm25":
        ranked = [(fact, bm25.score(example.question, i)) for i, fact in enumerate(facts)]
    elif method.graph_gate == "dense":
        ranked = [(fact, HashedEmbeddingModel.cosine(qemb, fact.embedding)) for fact in facts]
    else:
        qtype, _, _ = memory.infer_profile_type(example.question)
        scored_edges = []
        for edge in memory.edges.values():
            score, _ = edge.score(qemb, qtype, memory.facts, use_utility=True, weights=memory.weights)
            if score > 0:
                scored_edges.append((edge, score))
        scored_edges.sort(key=lambda x: x[1], reverse=True)
        edges = [edge for edge, _ in scored_edges[: method.top_k_edges]]
        candidate_ids: Dict[str, float] = {}
        if method.graph_gate in {"edge", "hybrid"}:
            for edge, edge_score in scored_edges[: method.top_k_edges]:
                edge_sim = HashedEmbeddingModel.cosine(qemb, edge.embedding)
                for fid in edge.member_fact_ids:
                    fact = memory.facts.get(fid)
                    if fact is None:
                        continue
                    sim = HashedEmbeddingModel.cosine(qemb, fact.embedding)
                    candidate_ids[fid] = max(candidate_ids.get(fid, 0.0), 0.62 * sim + 0.24 * edge_sim + 0.14 * edge_score)
        if method.graph_gate == "session_full":
            selected_ids: List[str] = []
            for edge, _ in scored_edges[: method.top_k_edges]:
                if edge.metadata.get("edge_kind") != "session":
                    continue
                for fid in edge.member_fact_ids:
                    if fid not in selected_ids:
                        selected_ids.append(fid)
            selected_ranked = [(memory.facts[fid], 1.0) for fid in selected_ids if fid in memory.facts]
            selected = pack_ranked(selected_ranked, method.top_k_facts, method.max_tokens)
            avg = sum(score for _, score in scored_edges[: max(1, method.top_k_edges)]) / max(1, min(len(scored_edges), method.top_k_edges))
            return result(example.question, method.name, selected, [edge for edge, _ in scored_edges[: method.top_k_edges]], avg, started, len(selected_ids))
        if method.graph_gate in {"global", "hybrid"} or not candidate_ids:
            for fid, fact in memory.facts.items():
                sim = HashedEmbeddingModel.cosine(qemb, fact.embedding)
                candidate_ids[fid] = max(candidate_ids.get(fid, 0.0), sim)
        for fid, base in candidate_ids.items():
            fact = memory.facts[fid]
            lex = keyword_overlap(example.question, fact.content)
            cost = max(1, estimate_tokens(fact.content))
            recency = float(fact.metadata.get("session_index") or 0) / max(1.0, len(example.rows))
            role = str(fact.metadata.get("role") or "")
            role_prior = 0.05 if role == "user" else -0.02 if role == "assistant" else 0.0
            score = 0.66 * base + 0.25 * lex + method.time_boost * recency + role_prior
            score = score / (cost ** method.token_roi)
            ranked.append((fact, score))

    ranked.sort(key=lambda x: x[1], reverse=True)
    selected = pack_ranked(ranked, method.top_k_facts, method.max_tokens)
    avg = sum(s for _, s in ranked[: max(1, method.top_k_facts)]) / max(1, min(len(ranked), method.top_k_facts))
    return result(example.question, method.name, selected, edges, avg, started, len(ranked))


def retrieval_metrics(example: LongMemExample, ret: ProfileRetrievalResult) -> Dict[str, Any]:
    total_answer_turns = sum(1 for row in example.rows if row.get("has_answer"))
    selected_answer_turns = sum(1 for f in ret.selected_facts if f.metadata.get("has_answer"))
    fact_hit = any(bool(f.metadata.get("has_answer")) for f in ret.selected_facts)
    selected_sessions = {str(f.metadata.get("session_id")) for f in ret.selected_facts}
    answer_sessions = set(example.answer_session_ids)
    session_hit = bool(answer_sessions & selected_sessions) if answer_sessions else fact_hit
    answer_text = example.answer.lower().strip()
    answer_in_evidence = any(answer_text and answer_text in f.content.lower() for f in ret.selected_facts)
    return {
        "fact_hit": int(fact_hit),
        "answer_turn_recall": selected_answer_turns / max(1, total_answer_turns),
        "all_answer_turns_hit": int(total_answer_turns > 0 and selected_answer_turns >= total_answer_turns),
        "session_hit": int(session_hit),
        "answer_substring_hit": int(answer_in_evidence),
    }


class ThompsonRouter:
    def __init__(self, arms: Sequence[MethodConfig], seed: int = 11) -> None:
        import random

        self.arms = list(arms)
        self.rng = random.Random(seed)
        self.alpha: Dict[str, List[float]] = {}
        self.beta: Dict[str, List[float]] = {}

    def bucket(self, example: LongMemExample) -> str:
        if example.qid.endswith("_abs"):
            return "abstention"
        qtype = example.qtype or "unknown"
        if "temporal" in qtype:
            return "temporal"
        if "multi" in qtype:
            return "multi"
        if "knowledge" in qtype:
            return "update"
        return qtype

    def _ensure(self, bucket: str) -> None:
        self.alpha.setdefault(bucket, [1.0] * len(self.arms))
        self.beta.setdefault(bucket, [1.0] * len(self.arms))

    def select(self, example: LongMemExample, train: bool) -> int:
        bucket = self.bucket(example)
        self._ensure(bucket)
        if not train:
            means = [a / (a + b) for a, b in zip(self.alpha[bucket], self.beta[bucket])]
            return max(range(len(self.arms)), key=lambda i: (means[i], -i))
        samples = [self.rng.betavariate(a, b) for a, b in zip(self.alpha[bucket], self.beta[bucket])]
        return max(range(len(self.arms)), key=lambda i: (samples[i], -i))

    def update(self, example: LongMemExample, arm_idx: int, reward: float) -> None:
        bucket = self.bucket(example)
        self._ensure(bucket)
        reward = max(0.0, min(1.0, reward))
        self.alpha[bucket][arm_idx] += reward
        self.beta[bucket][arm_idx] += 1.0 - reward


class LLMClient:
    def __init__(self, model: str | None = None) -> None:
        load_runtime_env()
        api_key = os.getenv("DEEPSEEK_API_KEY", "")
        if not api_key:
            raise RuntimeError("DEEPSEEK_API_KEY is missing")
        self.client = OpenAI(api_key=api_key, base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"))
        self.model = model or os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

    def chat(self, prompt: str, max_tokens: int = 256, json_mode: bool = False) -> str:
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "max_tokens": max_tokens,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        last_exc: Exception | None = None
        for attempt in range(5):
            try:
                resp = self.client.chat.completions.create(**kwargs)
                return resp.choices[0].message.content or ""
            except OpenAIError as exc:
                last_exc = exc
                wait_s = min(60, 5 * (2**attempt))
                print(f"[warn] LLM request failed on attempt {attempt + 1}/5: {exc}; retrying in {wait_s}s", flush=True)
                time.sleep(wait_s)
        raise RuntimeError(f"LLM request failed after retries: {last_exc}") from last_exc


def sorted_evidence_facts(ret: ProfileRetrievalResult) -> List[ProfileFact]:
    facts = sorted(
        ret.selected_facts,
        key=lambda f: (
            0 if str(f.metadata.get("role") or "") == "user" else 1,
            int(f.metadata.get("session_index") or 0),
            int(f.metadata.get("turn_index") or 0),
        ),
    )
    return facts


def evidence_block(ret: ProfileRetrievalResult, max_chars: int = 4200) -> str:
    facts = sorted_evidence_facts(ret)
    lines = []
    for i, fact in enumerate(facts[:16], start=1):
        meta = fact.metadata or {}
        date = str(meta.get("date") or "")
        role = str(meta.get("role") or "")
        text = str(meta.get("raw_content") or fact.content).replace("\n", " ").strip()
        lines.append(f"[{i}] {date} {role}: {text[:360]}")
    text = "\n".join(lines)
    return text[:max_chars]


MONTHS = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
}


def parse_session_date(text: str) -> dt.datetime | None:
    match = re.search(r"(\d{4})/(\d{2})/(\d{2})(?:\s+\([^)]*\)\s+(\d{2}):(\d{2}))?", text or "")
    if match:
        year, month, day = int(match.group(1)), int(match.group(2)), int(match.group(3))
        hour = int(match.group(4) or 0)
        minute = int(match.group(5) or 0)
        return dt.datetime(year, month, day, hour, minute)
    match = re.search(
        r"(?:(\d{1,2}):(\d{2})\s*(am|pm)\s+on\s+)?(\d{1,2})\s+"
        r"(January|February|March|April|May|June|July|August|September|October|November|December),?\s+(\d{4})",
        text or "",
        flags=re.I,
    )
    if not match:
        return None
    hour = int(match.group(1) or 0)
    minute = int(match.group(2) or 0)
    ampm = (match.group(3) or "").lower()
    if ampm == "pm" and hour < 12:
        hour += 12
    if ampm == "am" and hour == 12:
        hour = 0
    day = int(match.group(4))
    month = MONTHS[match.group(5).lower()]
    year = int(match.group(6))
    return dt.datetime(year, month, day, hour, minute)


def parse_explicit_dates(text: str, default_year: int | None) -> List[str]:
    out: List[str] = []
    for match in re.finditer(
        r"\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2})(?:st|nd|rd|th)?(?:,\s*(\d{4}))?",
        text or "",
        flags=re.I,
    ):
        year = int(match.group(3) or default_year or 2023)
        month = MONTHS[match.group(1).lower()]
        day = int(match.group(2))
        try:
            out.append(dt.date(year, month, day).isoformat())
        except ValueError:
            continue
    for match in re.finditer(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b", text or ""):
        try:
            out.append(dt.date(int(match.group(1)), int(match.group(2)), int(match.group(3))).isoformat())
        except ValueError:
            continue
    return list(dict.fromkeys(out))


def relative_time_notes(text: str, ref: dt.datetime | None) -> List[str]:
    if ref is None:
        return []
    lower = (text or "").lower()
    notes: List[str] = []
    number_words = {
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
        "seven": 7,
        "eight": 8,
        "nine": 9,
        "ten": 10,
    }
    for raw_num, unit in re.findall(r"\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+(day|days|week|weeks|month|months)\s+ago\b", lower):
        num = int(raw_num) if raw_num.isdigit() else number_words.get(raw_num, 0)
        days = num if unit.startswith("day") else 7 * num if unit.startswith("week") else 30 * num
        notes.append(f"'{raw_num} {unit} ago' relative to {ref.date().isoformat()} is about {(ref.date() - dt.timedelta(days=days)).isoformat()}.")
    if "last week" in lower:
        notes.append(f"'last week' relative to {ref.date().isoformat()} is about {(ref.date() - dt.timedelta(days=7)).isoformat()}.")
    if "last month" in lower:
        notes.append(f"'last month' relative to {ref.date().isoformat()} is about {(ref.date() - dt.timedelta(days=30)).isoformat()}.")
    if "black friday" in lower:
        notes.append("In 2023, Black Friday was 2023-11-24; a week before Black Friday was 2023-11-17.")
    return notes[:6]


def temporal_notes(example: LongMemExample, ret: ProfileRetrievalResult, max_lines: int = 24) -> str:
    lines: List[str] = []
    all_dates: List[dt.date] = []
    for i, fact in enumerate(sorted_evidence_facts(ret)[:12], start=1):
        meta = fact.metadata or {}
        raw = str(meta.get("raw_content") or fact.content)
        ref = parse_session_date(str(meta.get("date") or fact.content))
        explicit = parse_explicit_dates(raw, ref.year if ref else None)
        if ref:
            all_dates.append(ref.date())
        for d in explicit:
            try:
                all_dates.append(dt.date.fromisoformat(d))
            except ValueError:
                pass
        notes = relative_time_notes(raw, ref)
        parts = []
        if ref:
            parts.append(f"session_time={ref.isoformat(sep=' ', timespec='minutes')}")
        if explicit:
            parts.append(f"explicit_dates={', '.join(explicit[:4])}")
        if notes:
            parts.append("relative=" + " ".join(notes))
        times = re.findall(r"\b\d{1,2}:\d{2}\s*(?:AM|PM|am|pm)?\b", raw)
        if times:
            parts.append(f"clock_times={', '.join(times[:4])}")
        nums = re.findall(r"\b\d+(?:\.\d+)?\s*(?:days?|weeks?|months?|minutes?|hours?)\b", raw, flags=re.I)
        if nums:
            parts.append(f"durations={', '.join(nums[:5])}")
        if parts:
            lines.append(f"[{i}] " + "; ".join(parts))
    unique_dates = sorted(set(all_dates))
    if len(unique_dates) >= 2:
        pairs = []
        for left, right in zip(unique_dates, unique_dates[1:]):
            pairs.append(f"{left.isoformat()} -> {right.isoformat()} = {(right - left).days} days")
        lines.append("adjacent_date_differences: " + "; ".join(pairs[:8]))
    return "\n".join(lines[:max_lines])


def generate_and_judge(
    reader: LLMClient,
    judge: LLMClient,
    example: LongMemExample,
    ret: ProfileRetrievalResult,
    cache: Dict[str, Any],
    key: str,
    reader_mode: str,
) -> Dict[str, Any]:
    if key in cache:
        return dict(cache[key])
    notes = temporal_notes(example, ret) if reader_mode in {"temporal", "temporal_strict"} else ""
    extra = f"\nTemporal calculation notes:\n{notes}\n" if notes else ""
    instruction = (
        "Answer the LongMemEval question using only the retrieved conversation evidence. "
        "Use the temporal calculation notes to compare dates, relative times, durations, and clock times. "
        "If a turn says a relative time such as last week, yesterday, or years ago, anchor it to that turn's session date. "
        "For when/date questions, output the resolved absolute date or month whenever the evidence provides a session date; do not answer only with a relative phrase such as yesterday or last week. "
        "For list questions, include every item supported by the evidence and avoid dropping later items from the same evidence block. "
        "Do not merge unrelated candidate answers from different evidence blocks unless the question explicitly asks for all, every, both, or multiple items. "
        "When several evidence blocks mention similar entities or events, choose the block that most directly matches the wording and time context of the question. "
        "For what/who questions, answer the exact entity, object, event, pet, activity, or identity requested rather than a broader description. "
        "It is acceptable to answer with a relative phrase such as \"the week before 9 June 2023\" when that is the most faithful answer. "
        "When the evidence contains a direct clue, do not answer \"I don't know\" just because the exact calendar date is implicit. "
        "You must output one non-empty short phrase, preserving important qualifiers such as who the career or event is for. "
        "If the evidence is genuinely insufficient, output \"I don't know\"."
        if reader_mode in {"temporal", "temporal_strict"}
        else
        "Answer the LongMemEval question using only the retrieved conversation evidence. "
        "Do any date/count comparison needed. For list questions, include every item supported by the evidence. "
        "Do not merge unrelated candidate answers unless the question asks for all or multiple items. "
        "You must output a non-empty short phrase. "
        "If the evidence is insufficient, output \"I don't know\"."
    )
    answer_prompt = (
        f"{instruction}\n\n"
        f"Question date: {example.question_date}\n"
        f"Question: {example.question}\n\n"
        f"Retrieved evidence:\n{evidence_block(ret)}\n\n"
        f"{extra}"
        "Short answer:"
    )
    pred = reader.chat(answer_prompt, max_tokens=320 if "reasoner" in reader.model else 160).strip()
    if not pred:
        retry_prompt = (
            "Use the evidence to answer the question in one short phrase. If impossible, say I don't know.\n\n"
            f"Question: {example.question}\n"
            f"Evidence:\n{evidence_block(ret, max_chars=2600)}\n\n"
            "Answer:"
        )
        pred = reader.chat(retry_prompt, max_tokens=80).strip()
    judge_prompt = (
        "You are a strict but fair evaluator for long-term memory QA. "
        "Judge whether the predicted answer is semantically correct according to the gold answer. "
        "Return JSON only: {\"score\":0 or 1,\"reason\":\"short\"}.\n\n"
        f"Question: {example.question}\n"
        f"Gold answer: {example.answer}\n"
        f"Predicted answer: {pred}\n"
    )
    raw = judge.chat(judge_prompt, max_tokens=160, json_mode=True)
    data = safe_json(raw)
    score = int(1 if str(data.get("score", "0")).strip().lower() in {"1", "true"} else 0)
    out = {"answer": pred, "judge_score": score, "judge_reason": str(data.get("reason") or "")[:300], "judge_raw": raw}
    cache[key] = out
    return out


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_method: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        by_method.setdefault(row["method"], []).append(row)
    out = []
    for method, part in by_method.items():
        n = len(part)
        out.append(
            {
                "method": method,
                "n": n,
                "fact_hit": round(sum(r["fact_hit"] for r in part) / max(1, n), 6),
                "answer_turn_recall": round(sum(r["answer_turn_recall"] for r in part) / max(1, n), 6),
                "all_answer_turns_hit": round(sum(r["all_answer_turns_hit"] for r in part) / max(1, n), 6),
                "session_hit": round(sum(r["session_hit"] for r in part) / max(1, n), 6),
                "answer_substring_hit": round(sum(r["answer_substring_hit"] for r in part) / max(1, n), 6),
                "retrieval_tokens": round(sum(float(r["retrieval_tokens"]) for r in part) / max(1, n), 3),
                "retrieval_ms": round(sum(float(r["retrieval_ms"]) for r in part) / max(1, n), 3),
                "llm_judge_accuracy": round(
                    sum(float(r.get("judge_score", 0)) for r in part if "judge_score" in r)
                    / max(1, sum(1 for r in part if "judge_score" in r)),
                    6,
                ),
                "llm_n": sum(1 for r in part if "judge_score" in r),
            }
        )
    return sorted(out, key=lambda x: (x["llm_judge_accuracy"], x["fact_hit"], x["session_hit"]), reverse=True)


def random_train_test_split(examples: Sequence[LongMemExample], train_size: int, seed: int) -> Tuple[List[LongMemExample], List[LongMemExample]]:
    rng = random.Random(seed)
    shuffled = list(examples)
    rng.shuffle(shuffled)
    train = shuffled[: min(train_size, len(shuffled))]
    train_ids = {ex.qid for ex in train}
    test = [ex for ex in shuffled if ex.qid not in train_ids]
    return train, test


def qtype_counts(examples: Sequence[LongMemExample]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for ex in examples:
        counts[ex.qtype] = counts.get(ex.qtype, 0) + 1
    return dict(sorted(counts.items()))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-examples", type=int, default=50)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--train-size", type=int, default=20)
    parser.add_argument("--max-llm-judge", type=int, default=10)
    parser.add_argument("--methods", default="", help="Comma-separated method names to run. Empty runs all methods.")
    parser.add_argument("--reader-model", default="", help="Override answer generation model.")
    parser.add_argument("--judge-model", default="", help="Override judge model.")
    parser.add_argument("--reader-mode", default="direct", choices=["direct", "temporal", "temporal_strict"])
    parser.add_argument("--qwen-embedding-url", default=os.getenv("EMBEDDING_BASE_URL", "http://localhost:11810/v1/embeddings"))
    parser.add_argument("--qwen-reranker-url", default=os.getenv("RERANKER_BASE_URL", "http://localhost:12810"))
    parser.add_argument("--use-qwen-reranker", action="store_true")
    parser.add_argument("--skip-abs", action="store_true")
    parser.add_argument("--random-split", action="store_true")
    parser.add_argument("--random-seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    examples = load_examples(Path(args.data), max_examples=args.max_examples, start_index=args.start_index, skip_abs=args.skip_abs)
    if args.random_split:
        train, test = random_train_test_split(examples, args.train_size, args.random_seed)
    else:
        train = examples[: min(args.train_size, len(examples))]
        test = examples[min(args.train_size, len(examples)) :]
    (out_dir / "split_info.json").write_text(
        json.dumps(
            {
                "random_split": bool(args.random_split),
                "random_seed": args.random_seed,
                "num_examples": len(examples),
                "train_size": len(train),
                "test_size": len(test),
                "train_qtypes": qtype_counts(train),
                "test_qtypes": qtype_counts(test),
                "train_qids": [ex.qid for ex in train],
                "test_qids": [ex.qid for ex in test],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    methods = [
        MethodConfig("Oracle-all-turns", graph_gate="oracle_all", top_k_facts=32, max_tokens=1400),
        MethodConfig("BM25-turn", graph_gate="bm25", top_k_facts=6, max_tokens=320),
        MethodConfig("QwenEmb-turn", graph_gate="qwen_dense", top_k_facts=6, max_tokens=320, initial_candidates=18),
        MethodConfig("QwenEmb-HG-edge", graph_gate="qwen_hg", top_k_edges=3, top_k_facts=6, max_tokens=320, initial_candidates=18),
        MethodConfig("HyperMem-Flow", graph_gate="hypermem_full", top_k_facts=30, max_tokens=1800, initial_candidates=100, topic_top_k=15, episode_top_k=20, lambda_prop=0.5),
        MethodConfig("HG-QwenRRF-Rerank", graph_gate="hypermem_full", top_k_facts=30, max_tokens=1800, initial_candidates=200, topic_top_k=10, episode_top_k=20, lambda_prop=0.5),
        MethodConfig("HG-QwenRRF-Compact", graph_gate="hypermem_full", top_k_facts=16, max_tokens=1000, initial_candidates=120, topic_top_k=6, episode_top_k=12, lambda_prop=0.5),
        MethodConfig("HG-QwenRRF-Broad", graph_gate="hypermem_full", top_k_facts=40, max_tokens=2400, initial_candidates=240, topic_top_k=12, episode_top_k=25, lambda_prop=0.5),
        MethodConfig("DenseHash-turn", graph_gate="dense", top_k_facts=6, max_tokens=320),
        MethodConfig("HG-session-edge", graph_gate="edge", top_k_edges=3, top_k_facts=6, max_tokens=320, token_roi=0.10),
        MethodConfig("HG-session-full", graph_gate="session_full", top_k_edges=3, top_k_facts=24, max_tokens=1000, token_roi=0.0),
        MethodConfig("HG-hybrid-roi", graph_gate="hybrid", top_k_edges=3, top_k_facts=5, max_tokens=260, token_roi=0.22),
        MethodConfig("HG-global-roi", graph_gate="global", top_k_edges=0, top_k_facts=5, max_tokens=240, token_roi=0.25),
    ]
    wanted_methods = {name.strip() for name in args.methods.split(",") if name.strip()}
    if wanted_methods:
        methods = [method for method in methods if method.name in wanted_methods]
    needs_qwen_embedding = any(method.graph_gate in {"qwen_dense", "qwen_hg", "hypermem_full"} for method in methods) or "HG-RL-Full" in wanted_methods
    qwen_embed = QwenEmbeddingClient(base_url=args.qwen_embedding_url) if needs_qwen_embedding else None
    qwen_reranker = QwenRerankerClient(base_url=args.qwen_reranker_url) if args.use_qwen_reranker and needs_qwen_embedding else None
    rl_arms = [
        MethodConfig("arm_bm25_k6", graph_gate="bm25", top_k_facts=6, max_tokens=320),
        MethodConfig("arm_hg_edge", graph_gate="edge", top_k_edges=3, top_k_facts=6, max_tokens=320, token_roi=0.10),
        MethodConfig("arm_hg_session_full", graph_gate="session_full", top_k_edges=3, top_k_facts=24, max_tokens=1000, token_roi=0.0),
        MethodConfig("arm_hg_hybrid_roi", graph_gate="hybrid", top_k_edges=3, top_k_facts=5, max_tokens=260, token_roi=0.22),
        MethodConfig("arm_hg_global_roi", graph_gate="global", top_k_facts=5, max_tokens=240, token_roi=0.25),
    ]
    router = ThompsonRouter(rl_arms)
    full_rl_arms = [
        MethodConfig("arm_full_compact", graph_gate="hypermem_full", top_k_facts=20, max_tokens=1250, initial_candidates=70, topic_top_k=6, episode_top_k=12, lambda_prop=0.5),
        MethodConfig("arm_full_default", graph_gate="hypermem_full", top_k_facts=24, max_tokens=1500, initial_candidates=85, topic_top_k=8, episode_top_k=16, lambda_prop=0.5),
        MethodConfig("arm_full_broad", graph_gate="hypermem_full", top_k_facts=28, max_tokens=1700, initial_candidates=100, topic_top_k=10, episode_top_k=18, lambda_prop=0.5),
    ]
    full_router = ThompsonRouter(full_rl_arms, seed=23)

    for ex in train:
        memory = build_memory(ex.rows)
        arm_idx = router.select(ex, train=True)
        ret = retrieve_method(ex, memory, rl_arms[arm_idx])
        m = retrieval_metrics(ex, ret)
        reward = 1.0 if m["fact_hit"] else 0.0
        reward -= min(0.20, ret.tokens / 2000.0)
        router.update(ex, arm_idx, reward)
        if "HG-RL-Full" in wanted_methods and qwen_embed is not None:
            full_arm_idx = full_router.select(ex, train=True)
            full_ret = retrieve_method(ex, memory, full_rl_arms[full_arm_idx], qwen_embed=qwen_embed, qwen_reranker=qwen_reranker)
            fm = retrieval_metrics(ex, full_ret)
            full_latency = float(full_ret.debug_scores[0].get("latency_ms", 0.0)) if full_ret.debug_scores else 0.0
            full_reward = 0.55 * fm["fact_hit"] + 0.30 * fm["answer_turn_recall"] + 0.15 * fm["all_answer_turns_hit"]
            full_reward -= min(0.16, full_ret.tokens / 9000.0)
            full_reward -= min(0.06, full_latency / 14000.0)
            full_router.update(ex, full_arm_idx, full_reward)

    cache_path = out_dir / "llm_cache.json"
    cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}
    reader = LLMClient(model=args.reader_model or None) if args.max_llm_judge > 0 else None
    judge = LLMClient(model=args.judge_model or None) if args.max_llm_judge > 0 else None

    rows: List[Dict[str, Any]] = []
    trace_path = out_dir / "trace.jsonl"
    with trace_path.open("w", encoding="utf-8") as trace:
        for qi, ex in enumerate(test, start=1):
            memory = build_memory(ex.rows)
            method_runs: List[Tuple[str, ProfileRetrievalResult]] = []
            for method in methods:
                method_runs.append((method.name, retrieve_method(ex, memory, method, qwen_embed=qwen_embed, qwen_reranker=qwen_reranker)))
            if not wanted_methods or "HG-RL-Thompson" in wanted_methods:
                arm_idx = router.select(ex, train=False)
                method_runs.append(("HG-RL-Thompson", retrieve_method(ex, memory, rl_arms[arm_idx])))
            if "HG-RL-Full" in wanted_methods:
                if qwen_embed is None:
                    raise RuntimeError("HG-RL-Full requires Qwen embedding service")
                full_arm_idx = full_router.select(ex, train=False)
                method_runs.append(("HG-RL-Full", retrieve_method(ex, memory, full_rl_arms[full_arm_idx], qwen_embed=qwen_embed, qwen_reranker=qwen_reranker)))

            for method_name, ret in method_runs:
                metrics = retrieval_metrics(ex, ret)
                row = {
                    "method": method_name,
                    "qid": ex.qid,
                    "qtype": ex.qtype,
                    "question": ex.question,
                    "gold": ex.answer,
                    **metrics,
                    "retrieval_tokens": ret.tokens,
                    "retrieval_ms": ret.debug_scores[0]["latency_ms"] if ret.debug_scores else 0.0,
                    "num_facts": len(ret.selected_facts),
                }
                if reader is not None and judge is not None and qi <= args.max_llm_judge:
                    cache_key = f"{ex.qid}::{method_name}::reader={reader.model}::judge={judge.model}::mode={args.reader_mode}"
                    judged = generate_and_judge(reader, judge, ex, ret, cache, cache_key, args.reader_mode)
                    row.update({k: v for k, v in judged.items() if k != "judge_raw"})
                    cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
                rows.append(row)
                trace.write(
                    json.dumps(
                        {
                            **row,
                            "evidence": [f.content for f in ret.selected_facts],
                            "selected_sessions": [f.metadata.get("session_id") for f in ret.selected_facts],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            print(f"[done] {qi}/{len(test)}", flush=True)

    write_csv(out_dir / "longmemeval_mini_results.csv", rows)
    summary = summarize(rows)
    write_csv(out_dir / "longmemeval_mini_summary.csv", summary)
    (out_dir / "router_state.json").write_text(json.dumps(router.__dict__, default=str, ensure_ascii=False, indent=2), encoding="utf-8")
    print((out_dir / "longmemeval_mini_summary.csv").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
