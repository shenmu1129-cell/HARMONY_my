"""LLM-based multi-membership profile hypergraph builder."""

from __future__ import annotations

import json
import math
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None  # type: ignore[assignment]

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None  # type: ignore[assignment]

from hypermem.profile_centric_hypergraph import (
    HashedEmbeddingModel,
    ProfileCentricHypergraphMemory,
    ProfileEdgeType,
    ProfileFact,
    ProfileHyperedgeUnit,
    clamp,
    keyword_overlap,
)

GENERIC_FEATURE_NAMES = {"研究", "时间", "心理", "工具", "目标", "偏好", "状态", "其他", "general", "misc", "用户", "内容"}


@dataclass
class LLMFeature:
    feature_id: str
    feature_name: str
    feature_type: str
    description: str
    assigned_fact_ids: List[str]
    positive_triggers: List[str] = field(default_factory=list)
    negative_triggers: List[str] = field(default_factory=list)
    confidence: float = 0.55
    edge_type: ProfileEdgeType = ProfileEdgeType.AUTO_DISCOVERED

    def text(self) -> str:
        return " ".join([self.feature_name, self.feature_type, self.description, " ".join(self.positive_triggers)]).strip()


def _progress(iterable, *, total: Optional[int], desc: str, enabled: bool):
    if enabled and tqdm is not None:
        return tqdm(iterable, total=total, desc=desc, dynamic_ncols=True)
    return iterable


def _dedupe(items: Iterable[str]) -> List[str]:
    return list(dict.fromkeys(str(x) for x in items if str(x).strip()))


def _safe_json(text: str) -> Dict[str, Any]:
    text = (text or "").strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.S)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass
    s, e = text.find("{"), text.rfind("}")
    if s >= 0 and e > s:
        try:
            return json.loads(text[s : e + 1])
        except Exception:
            return {}
    return {}


def _edge_type(ftype: str, name: str, desc: str) -> ProfileEdgeType:
    text = " ".join([ftype, name, desc]).lower()
    if any(x in text for x in ["preference", "偏好", "喜欢", "不喜欢", "style"]):
        return ProfileEdgeType.PREFERENCE
    if any(x in text for x in ["goal", "目标", "投稿", "aaai", "paper", "论文"]):
        return ProfileEdgeType.GOAL
    if any(x in text for x in ["habit", "习惯", "workflow", "流程"]):
        return ProfileEdgeType.HABIT
    if any(x in text for x in ["current", "当前", "状态", "psych", "心理", "担忧", "焦虑"]):
        return ProfileEdgeType.CURRENT_STATE
    if any(x in text for x in ["temporal", "time", "时间", "演化", "变化", "阶段"]):
        return ProfileEdgeType.TEMPORAL_EVOLUTION
    if any(x in text for x in ["domain", "knowledge", "知识", "算法", "强化学习", "embedding"]):
        return ProfileEdgeType.DOMAIN_KNOWLEDGE
    if any(x in text for x in ["tool", "工具", "github", "codex", "服务器", "conda", "bash"]):
        return ProfileEdgeType.TOOL_USAGE
    if any(x in text for x in ["writing", "写作", "表达", "汇报", "ppt"]):
        return ProfileEdgeType.WRITING_STYLE
    if any(x in text for x in ["research", "研究", "memory", "hypermem", "rag", "locomo"]):
        return ProfileEdgeType.RESEARCH_FOCUS
    return ProfileEdgeType.AUTO_DISCOVERED


class LLMFeatureClient:
    def __init__(self, api_key_env: str, base_url: str, model: str, temperature: float, max_features: int, max_features_per_fact: int) -> None:
        if OpenAI is None:
            raise RuntimeError("openai package is required for llm_batch mode")
        api_key = os.getenv(api_key_env)
        if not api_key:
            raise RuntimeError(f"missing environment variable: {api_key_env}")
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.temperature = temperature
        self.max_features = max_features
        self.max_features_per_fact = max_features_per_fact

    def induce(self, facts: Sequence[ProfileFact], existing_edges: Sequence[ProfileHyperedgeUnit]) -> List[LLMFeature]:
        valid_ids = {f.fact_id for f in facts}
        payload = {
            "facts": [{"fact_id": f.fact_id, "content": f.content[:700], "keywords": f.keywords[:8]} for f in facts],
            "existing_features": [
                {
                    "edge_id": e.edge_id,
                    "feature_name": e.metadata.get("feature_name", e.summary[:80]),
                    "feature_type": e.metadata.get("feature_type", e.edge_type.value),
                    "description": e.metadata.get("feature_description", e.summary[:180]),
                    "members": len(e.member_fact_ids),
                }
                for e in existing_edges[:80]
                if e.status == "active"
            ],
            "limits": {"max_features": self.max_features, "max_features_per_fact": self.max_features_per_fact},
        }
        prompt = (
            "你是长期用户记忆超图构建器。请从 facts 中自由归纳共同用户画像特征，不要使用预设固定特征。"
            "一个 fact 可以属于多个特征。特征必须具体，避免'研究/时间/心理/工具/目标/偏好/状态'这类空泛词。"
            "若可复用 existing_features，请把 feature_id 写为已有 edge_id；否则创建新 feature_id。"
            "每个特征必须包含清晰边界 positive_triggers 和 negative_triggers。只输出 JSON："
            "{\"features\":[{\"feature_id\":\"...\",\"feature_name\":\"...\",\"feature_type\":\"...\","
            "\"description\":\"...\",\"positive_triggers\":[\"...\"],\"negative_triggers\":[\"...\"],"
            "\"assigned_fact_ids\":[\"...\"],\"confidence\":0.7}]}\n"
            f"输入：{json.dumps(payload, ensure_ascii=False)}"
        )
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
        )
        data = _safe_json(resp.choices[0].message.content or "")
        out: List[LLMFeature] = []
        for i, row in enumerate(data.get("features") or []):
            if not isinstance(row, dict):
                continue
            name = str(row.get("feature_name") or "").strip()
            ftype = str(row.get("feature_type") or "other").strip()
            desc = str(row.get("description") or "").strip()
            assigned = [x for x in _dedupe(row.get("assigned_fact_ids") or []) if x in valid_ids]
            if not name or not desc or not assigned:
                continue
            try:
                conf = float(row.get("confidence", 0.55))
            except Exception:
                conf = 0.55
            out.append(
                LLMFeature(
                    feature_id=str(row.get("feature_id") or f"llm_feature_{i+1:03d}"),
                    feature_name=name,
                    feature_type=ftype,
                    description=desc,
                    assigned_fact_ids=assigned,
                    positive_triggers=_dedupe(row.get("positive_triggers") or []),
                    negative_triggers=_dedupe(row.get("negative_triggers") or []),
                    confidence=clamp(conf, lo=0.0, hi=0.95),
                    edge_type=_edge_type(ftype, name, desc),
                )
            )
        return out[: self.max_features]

    def suggest_merges(self, edges: Sequence[ProfileHyperedgeUnit]) -> List[List[str]]:
        active = [e for e in edges if e.status == "active"]
        if len(active) <= 1:
            return []
        payload = [
            {
                "edge_id": e.edge_id,
                "feature_name": e.metadata.get("feature_name", e.summary[:80]),
                "feature_type": e.metadata.get("feature_type", e.edge_type.value),
                "description": e.metadata.get("feature_description", e.summary[:160]),
                "members": len(e.member_fact_ids),
            }
            for e in active[:120]
        ]
        prompt = "找出语义重复或边界高度重叠的用户画像超边，只输出 JSON：{\"merge_groups\":[[\"edge_id1\",\"edge_id2\"]]}\n输入：" + json.dumps(payload, ensure_ascii=False)
        resp = self.client.chat.completions.create(model=self.model, messages=[{"role": "user", "content": prompt}], temperature=0.0)
        data = _safe_json(resp.choices[0].message.content or "")
        return [g for g in data.get("merge_groups", []) if isinstance(g, list) and len(g) >= 2]


class LLMBatchProfileHypergraphBuilder:
    def __init__(
        self,
        memory: ProfileCentricHypergraphMemory,
        batch_size: int = 80,
        canonical_threshold: float = 0.72,
        consolidate_every: int = 3,
        llm_consolidation_rounds: int = 1,
        max_edge_facts: int = 400,
        api_key_env: str = "DEEPSEEK_API_KEY",
        base_url: str = "https://api.deepseek.com",
        model: str = "deepseek-chat",
        temperature: float = 0.2,
        max_features_per_batch: int = 12,
        max_features_per_fact: int = 4,
        show_progress: bool = False,
    ) -> None:
        self.memory = memory
        self.batch_size = max(1, batch_size)
        self.canonical_threshold = canonical_threshold
        self.consolidate_every = max(0, consolidate_every)
        self.llm_consolidation_rounds = max(0, llm_consolidation_rounds)
        self.max_edge_facts = max(50, max_edge_facts)
        self.max_features_per_fact = max_features_per_fact
        self.show_progress = show_progress
        self.client = LLMFeatureClient(api_key_env, base_url, model, temperature, max_features_per_batch, max_features_per_fact)
        self.created_edges = 0
        self.reused_edges = 0
        self.rejected_features = 0
        self.rejected_assignments = 0
        self.merges = 0
        self.splits = 0

    def build(self, rows: Sequence[Dict[str, Any]]) -> None:
        started = time.time()
        print(f"[llm-build] start rows={len(rows)} batch_size={self.batch_size} canonical_threshold={self.canonical_threshold}", flush=True)
        self._add_facts(rows)
        fact_ids = list(self.memory.facts.keys())
        batches = [fact_ids[i : i + self.batch_size] for i in range(0, len(fact_ids), self.batch_size)]
        it = _progress(enumerate(batches, start=1), total=len(batches), desc="[llm-build] induce batches", enabled=self.show_progress)
        for batch_idx, batch in it:
            stats = self._process_batch(batch_idx, batch)
            if self.consolidate_every and batch_idx % self.consolidate_every == 0:
                self.consolidate(use_llm=False, reason=f"periodic_{batch_idx}")
            if tqdm is not None and hasattr(it, "set_postfix"):
                it.set_postfix(active=self.memory.active_edge_count(), created=stats["created"], reused=stats["reused"], features=stats["features"])
        for r in range(self.llm_consolidation_rounds):
            self.consolidate(use_llm=True, reason=f"llm_round_{r+1}")
        self.consolidate(use_llm=False, reason="final")
        print(
            f"[llm-build] done elapsed={time.time() - started:.2f}s facts={len(self.memory.facts)} edges={len(self.memory.edges)} "
            f"active={self.memory.active_edge_count()} created={self.created_edges} reused={self.reused_edges} "
            f"rejected_features={self.rejected_features} rejected_assignments={self.rejected_assignments} merges={self.merges} splits={self.splits}",
            flush=True,
        )

    def _add_facts(self, rows: Sequence[Dict[str, Any]]) -> None:
        it = _progress(enumerate(rows), total=len(rows), desc="[llm-build] add raw facts", enabled=self.show_progress)
        for i, row in it:
            content = row.get("content") or row.get("text") or row.get("fact") or row.get("summary") or ""
            if not content:
                continue
            self.memory.add_fact(
                content=str(content),
                fact_id=str(row.get("fact_id") or row.get("id") or f"fact_{i+1:06d}"),
                keywords=[str(x) for x in row.get("keywords", [])] if isinstance(row.get("keywords", []), list) else [],
                timestamp=float(row.get("timestamp") or row.get("time_index") or i + 1),
                metadata=dict(row),
                promote=False,
            )

    def _process_batch(self, batch_idx: int, fact_ids: Sequence[str]) -> Dict[str, int]:
        facts = [self.memory.facts[fid] for fid in fact_ids if fid in self.memory.facts]
        features = self.client.induce(facts, self._active_edges())
        touched: Set[str] = set()
        per_fact: Dict[str, int] = {}
        created = reused = 0
        for feature in features:
            if not self._valid_feature(feature):
                self.rejected_features += 1
                continue
            members = []
            for fid in feature.assigned_fact_ids:
                per_fact[fid] = per_fact.get(fid, 0) + 1
                if per_fact[fid] <= self.max_features_per_fact:
                    members.append(fid)
                else:
                    self.rejected_assignments += 1
            members = _dedupe(members)
            if not members:
                continue
            edge = self._canonicalize(feature)
            if edge is None:
                edge = self._create_edge(feature, members, batch_idx)
                created += 1
            else:
                self._append_members(edge, members)
                reused += 1
            touched.add(edge.edge_id)
        for eid in touched:
            edge = self.memory.edges.get(eid)
            if edge and edge.status == "active":
                self._refresh(edge)
        self.created_edges += created
        self.reused_edges += reused
        return {"features": len(features), "created": created, "reused": reused}

    def _active_edges(self) -> List[ProfileHyperedgeUnit]:
        return [e for e in self.memory.edges.values() if e.status == "active"]

    def _valid_feature(self, f: LLMFeature) -> bool:
        name = f.feature_name.strip().lower()
        return name not in GENERIC_FEATURE_NAMES and len(f.feature_name) >= 4 and len(f.description) >= 12 and bool(f.assigned_fact_ids) and f.confidence >= 0.35

    def _canonicalize(self, f: LLMFeature) -> Optional[ProfileHyperedgeUnit]:
        if f.feature_id in self.memory.edges and self.memory.edges[f.feature_id].status == "active":
            return self.memory.edges[f.feature_id]
        femb = self.memory.embedding_model.encode(f.text())
        best = None
        best_score = 0.0
        for e in self._active_edges():
            if e.metadata.get("feature_type") != f.feature_type:
                continue
            etext = " ".join([str(e.metadata.get("feature_name", "")), str(e.metadata.get("feature_description", "")), e.summary, " ".join(e.keywords)])
            score = 0.45 * keyword_overlap(f.text(), etext) + 0.55 * HashedEmbeddingModel.cosine(femb, e.embedding)
            if score > best_score:
                best_score, best = score, e
        if best is not None and best_score >= self.canonical_threshold:
            best.metadata.setdefault("canonicalized_from", []).append({"feature_id": f.feature_id, "name": f.feature_name, "score": round(best_score, 6)})
            best.confidence_score = clamp(max(best.confidence_score, f.confidence))
            return best
        return None

    def _create_edge(self, f: LLMFeature, members: Sequence[str], batch_idx: int) -> ProfileHyperedgeUnit:
        return self.memory.create_edge(
            f.edge_type,
            list(dict.fromkeys(members)),
            summary=f"{f.feature_name}: {f.description}",
            confidence=f.confidence,
            metadata={
                "construction": "llm_batch_profile_feature_induction",
                "feature_id": f.feature_id,
                "feature_name": f.feature_name,
                "feature_type": f.feature_type,
                "feature_description": f.description,
                "positive_triggers": f.positive_triggers,
                "negative_triggers": f.negative_triggers,
                "batch_idx": batch_idx,
                "multi_membership": True,
                "boundary_control": "llm_positive_negative_schema",
                "canonicalization": "llm_reuse_then_similarity",
            },
        )

    def _append_members(self, edge: ProfileHyperedgeUnit, fact_ids: Sequence[str]) -> None:
        seen = set(edge.member_fact_ids)
        for fid in fact_ids:
            if fid not in seen:
                edge.member_fact_ids.append(fid)
                seen.add(fid)

    def consolidate(self, use_llm: bool, reason: str) -> None:
        merges = self._local_merge()
        if use_llm:
            merges += self._llm_merge()
        splits = self._split_large_edges()
        self.merges += merges
        self.splits += splits
        if merges or splits or self.show_progress:
            print(f"[llm-consolidate] reason={reason} merges={merges} splits={splits} active={self.memory.active_edge_count()}", flush=True)

    def _local_merge(self) -> int:
        edges = self._active_edges()
        merges = 0
        for i, left in enumerate(edges):
            if left.status != "active":
                continue
            for right in edges[i + 1 :]:
                if right.status != "active" or left.metadata.get("feature_type") != right.metadata.get("feature_type"):
                    continue
                if self._edge_similarity(left, right) >= min(0.93, self.canonical_threshold + 0.12):
                    self._append_members(left, right.member_fact_ids)
                    right.status = "merged"
                    right.metadata["merged_into"] = left.edge_id
                    self._refresh(left)
                    merges += 1
        return merges

    def _llm_merge(self) -> int:
        groups = self.client.suggest_merges(self._active_edges())
        merges = 0
        for group in groups:
            base = self.memory.edges.get(str(group[0]))
            if base is None or base.status != "active":
                continue
            for oid in group[1:]:
                other = self.memory.edges.get(str(oid))
                if other is None or other.status != "active":
                    continue
                self._append_members(base, other.member_fact_ids)
                other.status = "merged"
                other.metadata["merged_into"] = base.edge_id
                merges += 1
            self._refresh(base)
        return merges

    def _edge_similarity(self, a: ProfileHyperedgeUnit, b: ProfileHyperedgeUnit) -> float:
        at = " ".join([str(a.metadata.get("feature_name", "")), str(a.metadata.get("feature_description", "")), a.summary])
        bt = " ".join([str(b.metadata.get("feature_name", "")), str(b.metadata.get("feature_description", "")), b.summary])
        return 0.45 * keyword_overlap(at, bt) + 0.55 * HashedEmbeddingModel.cosine(a.embedding, b.embedding)

    def _split_large_edges(self) -> int:
        splits = 0
        for edge in list(self._active_edges()):
            if len(edge.member_fact_ids) <= self.max_edge_facts:
                continue
            chunks = [edge.member_fact_ids[i : i + self.max_edge_facts] for i in range(0, len(edge.member_fact_ids), self.max_edge_facts)]
            base_name = str(edge.metadata.get("feature_name", edge.edge_type.value))
            for idx, chunk in enumerate(chunks, start=1):
                meta = dict(edge.metadata)
                meta.update({"parent_edge_id": edge.edge_id, "feature_id": f"{edge.edge_id}:part:{idx}", "feature_name": f"{base_name} #{idx}", "split_reason": "max_edge_facts"})
                child = self.memory.create_edge(edge.edge_type, list(chunk), summary=f"{base_name} #{idx}", confidence=edge.confidence_score, metadata=meta)
                self._refresh(child)
            edge.status = "merged"
            edge.metadata["split_into_parts"] = len(chunks)
            splits += len(chunks)
        return splits

    def _refresh(self, edge: ProfileHyperedgeUnit) -> None:
        facts = [self.memory.facts[fid] for fid in edge.member_fact_ids if fid in self.memory.facts]
        reps = facts if len(facts) <= 120 else facts[:60] + facts[-60:]
        name = str(edge.metadata.get("feature_name", edge.edge_type.value))
        desc = str(edge.metadata.get("feature_description", ""))
        preview = "；".join(f.content.strip() for f in reps[:3] if f.content.strip())
        edge.summary = f"{name}: {desc}" + (f" | examples: {preview}" if preview else "")
        edge.keywords = self.memory.extract_keywords(" ".join([edge.summary] + [f.text() for f in reps]))
        edge.embedding = self.memory.edge_embedding(edge.summary, reps)
        edge.coherence_score = 0.55
        edge.stability_score = clamp(0.45 + 0.08 * math.log1p(len(facts)))
        if facts:
            max_ts = max([f.timestamp for f in self.memory.facts.values()] or [1.0])
            edge.freshness_score = clamp(0.35 + 0.65 * (max(f.timestamp for f in facts) / max_ts))
        edge.metadata["num_member_facts"] = len(facts)
        edge.updated_at = time.time()


def build_llm_profile_hypergraph_from_rows(memory: ProfileCentricHypergraphMemory, rows: Sequence[Dict[str, Any]], **kwargs: Any) -> LLMBatchProfileHypergraphBuilder:
    builder = LLMBatchProfileHypergraphBuilder(memory, **kwargs)
    builder.build(rows)
    return builder
