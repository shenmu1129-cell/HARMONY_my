"""Batch-wise profile feature induction for profile-centric hypergraphs.

This builder replaces the slow fact-by-fact online attach strategy with:
    facts -> batch profile feature induction -> feature canonicalization ->
    profile hyperedges -> periodic consolidation.

The implementation is dependency-light and deterministic by default. It is written
so the rule-based feature induction can later be replaced by an LLM extractor
without changing the downstream hypergraph interface.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

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


@dataclass(frozen=True)
class FeatureRule:
    """A canonical profile feature boundary used by the local inducer."""

    signature: str
    feature_name: str
    feature_type: str
    edge_type: ProfileEdgeType
    description: str
    positive_triggers: Tuple[str, ...]
    negative_triggers: Tuple[str, ...] = ()
    base_confidence: float = 0.45


@dataclass
class FeatureSpec:
    """A feature induced from a fact or a batch of facts."""

    signature: str
    feature_name: str
    feature_type: str
    edge_type: ProfileEdgeType
    description: str
    positive_triggers: List[str] = field(default_factory=list)
    negative_triggers: List[str] = field(default_factory=list)
    confidence: float = 0.50
    source: str = "rule"

    def text(self) -> str:
        return " ".join([self.feature_name, self.feature_type, self.description, " ".join(self.positive_triggers)]).strip()


FEATURE_RULES: Tuple[FeatureRule, ...] = (
    FeatureRule(
        signature="research:llm_memory",
        feature_name="LLM记忆与超图研究主线",
        feature_type="research_focus",
        edge_type=ProfileEdgeType.RESEARCH_FOCUS,
        description="用户围绕 LLM memory、HyperMem、A-MEM、MeMo、LoCoMo、RAG 或超图记忆开展的研究内容。",
        positive_triggers=("memory", "hypermem", "a-mem", "memo", "locomo", "rag", "超图", "超边", "记忆"),
        negative_triggers=("显示器", "键盘", "拓展坞"),
    ),
    FeatureRule(
        signature="goal:publication_aaai",
        feature_name="AAAI/论文投稿目标",
        feature_type="publication_goal",
        edge_type=ProfileEdgeType.GOAL,
        description="用户关于 AAAI、ACL、TITS、TCSVT 等论文投稿、创新性和实验支撑的目标与约束。",
        positive_triggers=("aaai", "acl", "投稿", "论文", "审稿", "rebuttal", "创新", "实验", "baseline", "竞争力"),
        negative_triggers=("买", "显示器", "typec"),
    ),
    FeatureRule(
        signature="psych:innovation_anxiety",
        feature_name="创新不足与审稿风险担忧",
        feature_type="psychological_state",
        edge_type=ProfileEdgeType.CURRENT_STATE,
        description="用户对创新不足、审稿人质疑、实验不够或方案像工程拼接的担忧。",
        positive_triggers=("担心", "风险", "不足", "质疑", "审稿人", "不好", "不够", "工程", "竞争力", "焦虑"),
    ),
    FeatureRule(
        signature="temporal:current_plan",
        feature_name="当前阶段与方案演化",
        feature_type="temporal_state",
        edge_type=ProfileEdgeType.TEMPORAL_EVOLUTION,
        description="用户当前阶段、最近变化、后续打算以及从旧方案到新方案的演化。",
        positive_triggers=("当前", "现在", "最近", "后续", "之前", "后来", "演化", "改成", "下一步", "阶段"),
    ),
    FeatureRule(
        signature="tool:server_github_codex",
        feature_name="服务器/GitHub/Codex实验流程",
        feature_type="tool_usage",
        edge_type=ProfileEdgeType.TOOL_USAGE,
        description="用户在服务器、conda、bash、GitHub、Codex、日志和运行命令上的工程实验流程。",
        positive_triggers=("github", "codex", "服务器", "conda", "bash", "脚本", "clone", "commit", "pull", "日志", "运行", "python", "cuda"),
    ),
    FeatureRule(
        signature="preference:reviewer_style",
        feature_name="审稿人式冷静分析偏好",
        feature_type="preference",
        edge_type=ProfileEdgeType.PREFERENCE,
        description="用户偏好直接、冷静、审稿人视角的评价，重视风险、弱点和可救方向。",
        positive_triggers=("喜欢", "希望", "不喜欢", "审稿人", "风险", "弱点", "冷静", "直接", "评价", "分析"),
    ),
    FeatureRule(
        signature="domain:rl_reward_utility",
        feature_name="强化学习/效用奖励记忆排序",
        feature_type="domain_knowledge",
        edge_type=ProfileEdgeType.DOMAIN_KNOWLEDGE,
        description="用户关于 reward utility、bandit、强化学习、PPO、utility score 与记忆价值判断的讨论。",
        positive_triggers=("强化学习", "reward", "utility", "bandit", "ppo", "rl", "价值", "排序", "反馈", "效用"),
    ),
    FeatureRule(
        signature="writing:paper_expression",
        feature_name="论文表达与汇报材料组织",
        feature_type="writing_style",
        edge_type=ProfileEdgeType.WRITING_STYLE,
        description="用户关于论文段落、PPT、汇报、审稿回复、表格和表达精炼的写作需求。",
        positive_triggers=("ppt", "汇报", "写成", "润色", "表达", "表格", "公式", "md", "文档", "总结"),
    ),
)

GENERIC_FEATURE_NAMES = {"研究", "时间", "心理", "工具", "目标", "偏好", "状态", "其他", "general", "misc"}


def _progress(iterable, *, total: Optional[int], desc: str, enabled: bool):
    if enabled and tqdm is not None:
        return tqdm(iterable, total=total, desc=desc, dynamic_ncols=True)
    return iterable


def _contains_any(text: str, triggers: Sequence[str]) -> List[str]:
    lower = text.lower()
    return [trigger for trigger in triggers if trigger.lower() in lower]


def _dedupe(items: Iterable[str]) -> List[str]:
    return list(dict.fromkeys(str(x) for x in items if str(x).strip()))


def _auto_feature_from_fact(fact: ProfileFact, memory: ProfileCentricHypergraphMemory) -> FeatureSpec:
    keywords = fact.keywords or memory.extract_keywords(fact.content, max_keywords=6)
    useful = [kw for kw in keywords if len(kw) > 1][:4]
    source = str(fact.metadata.get("source", "auto")) if fact.metadata else "auto"
    if useful:
        name = "自动画像特征:" + "/".join(useful[:3])
        signature = "auto:" + ":".join(useful[:3])
        description = "由未命中预设画像规则的事实自动归纳出的局部画像特征。"
        triggers = useful
    else:
        name = f"自动画像特征:{source}"
        signature = f"auto:{source}"
        description = "由未命中预设画像规则的事实按数据来源自动归纳出的局部画像特征。"
        triggers = [source]
    return FeatureSpec(
        signature=signature,
        feature_name=name,
        feature_type="auto_discovered",
        edge_type=ProfileEdgeType.AUTO_DISCOVERED,
        description=description,
        positive_triggers=triggers,
        confidence=0.40,
        source="auto",
    )


class BatchProfileHypergraphBuilder:
    """Fast batch-wise constructor for profile-centric hyperedges.

    Risk controls implemented here:
      1. feature canonicalization: exact signatures plus similarity-based matching;
      2. feature boundary control: positive/negative trigger checks and generic-name rejection;
      3. periodic consolidation: duplicate/similar merge and oversized edge split.
    """

    def __init__(
        self,
        memory: ProfileCentricHypergraphMemory,
        batch_size: int = 200,
        canonical_threshold: float = 0.62,
        min_feature_support: int = 1,
        consolidate_every: int = 5,
        max_edge_facts: int = 600,
        max_coherence_pairs: int = 512,
        show_progress: bool = False,
    ) -> None:
        self.memory = memory
        self.batch_size = max(1, batch_size)
        self.canonical_threshold = canonical_threshold
        self.min_feature_support = max(1, min_feature_support)
        self.consolidate_every = max(0, consolidate_every)
        self.max_edge_facts = max(50, max_edge_facts)
        self.max_coherence_pairs = max(32, max_coherence_pairs)
        self.show_progress = show_progress
        self.created_edges = 0
        self.reused_edges = 0
        self.rejected_assignments = 0
        self.consolidation_merges = 0
        self.consolidation_splits = 0

    def build(self, rows: Sequence[Dict[str, Any]]) -> None:
        started = time.time()
        print(
            f"[feature-build] start rows={len(rows)} batch_size={self.batch_size} "
            f"canonical_threshold={self.canonical_threshold} min_support={self.min_feature_support} "
            f"max_edge_facts={self.max_edge_facts}",
            flush=True,
        )
        self._add_facts_without_promotion(rows)
        fact_ids = list(self.memory.facts.keys())
        batches = [fact_ids[i : i + self.batch_size] for i in range(0, len(fact_ids), self.batch_size)]
        iterator = _progress(enumerate(batches, start=1), total=len(batches), desc="[feature-build] induce batches", enabled=self.show_progress)
        for batch_idx, batch_fact_ids in iterator:
            stats = self._process_batch(batch_fact_ids)
            if self.consolidate_every and batch_idx % self.consolidate_every == 0:
                self.consolidate_feature_bank(reason=f"periodic_batch_{batch_idx}")
            if tqdm is not None and hasattr(iterator, "set_postfix"):
                iterator.set_postfix(
                    facts=len(self.memory.facts),
                    edges=len(self.memory.edges),
                    active=self.memory.active_edge_count(),
                    created=stats["created"],
                    reused=stats["reused"],
                )
            elif self.show_progress:
                print(
                    f"[feature-build] batch={batch_idx}/{len(batches)} created={stats['created']} reused={stats['reused']} "
                    f"edges={len(self.memory.edges)} active={self.memory.active_edge_count()}",
                    flush=True,
                )
        self.consolidate_feature_bank(reason="final")
        print(
            f"[feature-build] done elapsed={time.time() - started:.2f}s facts={len(self.memory.facts)} "
            f"edges={len(self.memory.edges)} active={self.memory.active_edge_count()} "
            f"created_edges={self.created_edges} reused_edges={self.reused_edges} "
            f"rejected_assignments={self.rejected_assignments} merges={self.consolidation_merges} splits={self.consolidation_splits}",
            flush=True,
        )

    def _add_facts_without_promotion(self, rows: Sequence[Dict[str, Any]]) -> None:
        iterator = _progress(enumerate(rows), total=len(rows), desc="[feature-build] add raw facts", enabled=self.show_progress)
        for i, row in iterator:
            content = row.get("content") or row.get("text") or row.get("fact") or row.get("summary") or ""
            if not content:
                continue
            fact_id = row.get("fact_id") or row.get("id") or f"fact_{i + 1:06d}"
            raw_keywords = row.get("keywords", [])
            keywords = [str(x) for x in raw_keywords] if isinstance(raw_keywords, list) else []
            self.memory.add_fact(
                content=str(content),
                fact_id=str(fact_id),
                keywords=keywords,
                timestamp=float(row.get("timestamp") or row.get("time_index") or i + 1),
                metadata=dict(row),
                promote=False,
            )

    def _process_batch(self, fact_ids: Sequence[str]) -> Dict[str, int]:
        induced: Dict[str, Tuple[FeatureSpec, List[str]]] = {}
        for fid in fact_ids:
            fact = self.memory.facts[fid]
            specs = self.induce_fact_features(fact)
            for spec in specs:
                if not self._valid_feature_assignment(spec, fact):
                    self.rejected_assignments += 1
                    continue
                if spec.signature not in induced:
                    induced[spec.signature] = (spec, [])
                induced[spec.signature][1].append(fid)

        created = 0
        reused = 0
        touched: Set[str] = set()
        for _, (spec, members) in induced.items():
            members = _dedupe(members)
            if len(members) < self.min_feature_support:
                continue
            edge = self._canonicalize_feature(spec)
            if edge is None:
                edge = self._create_feature_edge(spec, members)
                created += 1
            else:
                self._append_members(edge, members)
                reused += 1
            touched.add(edge.edge_id)
        for edge_id in touched:
            edge = self.memory.edges.get(edge_id)
            if edge and edge.status == "active":
                self._fast_refresh_edge(edge)
        self.created_edges += created
        self.reused_edges += reused
        return {"created": created, "reused": reused, "touched": len(touched)}

    def induce_fact_features(self, fact: ProfileFact) -> List[FeatureSpec]:
        specs: List[FeatureSpec] = []
        text = fact.content
        for rule in FEATURE_RULES:
            positives = _contains_any(text, rule.positive_triggers)
            negatives = _contains_any(text, rule.negative_triggers)
            if not positives or negatives:
                continue
            confidence = clamp(rule.base_confidence + 0.10 * min(4, len(positives)) - 0.08 * len(negatives), lo=0.0, hi=0.95)
            specs.append(
                FeatureSpec(
                    signature=rule.signature,
                    feature_name=rule.feature_name,
                    feature_type=rule.feature_type,
                    edge_type=rule.edge_type,
                    description=rule.description,
                    positive_triggers=list(positives),
                    negative_triggers=list(negatives),
                    confidence=confidence,
                    source="rule",
                )
            )
        if not specs:
            specs.append(_auto_feature_from_fact(fact, self.memory))
        return specs[:5]

    def _valid_feature_assignment(self, spec: FeatureSpec, fact: ProfileFact) -> bool:
        # Boundary control 1: reject overly generic feature names unless they are auto-specific.
        short_name = spec.feature_name.replace("自动画像特征:", "").strip().lower()
        if short_name in GENERIC_FEATURE_NAMES:
            return False
        # Boundary control 2: negative triggers block assignment.
        if _contains_any(fact.content, spec.negative_triggers):
            return False
        # Boundary control 3: non-auto features need a positive trigger in the fact.
        if spec.source != "auto" and not _contains_any(fact.content, spec.positive_triggers):
            return False
        # Boundary control 4: confidence must be meaningful.
        return spec.confidence >= 0.35

    def _canonicalize_feature(self, spec: FeatureSpec) -> Optional[ProfileHyperedgeUnit]:
        # Exact signature first.
        for edge in self.memory.edges.values():
            if edge.status != "active":
                continue
            if edge.metadata.get("feature_signature") == spec.signature:
                return edge

        # Similarity-based canonicalization prevents feature fragmentation.
        best_edge: Optional[ProfileHyperedgeUnit] = None
        best_score = 0.0
        spec_text = spec.text()
        spec_emb = self.memory.embedding_model.encode(spec_text)
        for edge in self.memory.edges.values():
            if edge.status != "active":
                continue
            if edge.metadata.get("feature_type") != spec.feature_type:
                continue
            edge_text = " ".join(
                [
                    str(edge.metadata.get("feature_name", "")),
                    str(edge.metadata.get("feature_description", "")),
                    " ".join(edge.keywords),
                    edge.summary,
                ]
            )
            overlap = keyword_overlap(spec_text, edge_text)
            emb_sim = HashedEmbeddingModel.cosine(spec_emb, edge.embedding)
            score = 0.45 * overlap + 0.55 * emb_sim
            if score > best_score:
                best_score, best_edge = score, edge
        if best_edge is not None and best_score >= self.canonical_threshold:
            best_edge.metadata.setdefault("canonicalized_from", [])
            best_edge.metadata["canonicalized_from"].append({
                "signature": spec.signature,
                "name": spec.feature_name,
                "score": round(best_score, 6),
            })
            best_edge.confidence_score = clamp(max(best_edge.confidence_score, spec.confidence))
            return best_edge
        return None

    def _create_feature_edge(self, spec: FeatureSpec, fact_ids: Sequence[str]) -> ProfileHyperedgeUnit:
        metadata = {
            "construction": "batch_profile_feature_induction",
            "feature_signature": spec.signature,
            "feature_name": spec.feature_name,
            "feature_type": spec.feature_type,
            "feature_description": spec.description,
            "positive_triggers": spec.positive_triggers,
            "negative_triggers": spec.negative_triggers,
            "boundary_control": "positive_negative_trigger_schema",
            "canonicalization": "signature_then_similarity",
        }
        summary = f"{spec.feature_name}: {spec.description}"
        return self.memory.create_edge(
            spec.edge_type,
            list(dict.fromkeys(fact_ids)),
            summary=summary,
            confidence=spec.confidence,
            metadata=metadata,
        )

    def _append_members(self, edge: ProfileHyperedgeUnit, fact_ids: Sequence[str]) -> None:
        existing = set(edge.member_fact_ids)
        for fid in fact_ids:
            if fid not in existing:
                edge.member_fact_ids.append(fid)
                existing.add(fid)

    def _representative_facts(self, facts: Sequence[ProfileFact], max_items: int = 160) -> List[ProfileFact]:
        if len(facts) <= max_items:
            return list(facts)
        half = max_items // 2
        return list(facts[:half]) + list(facts[-half:])

    def _fast_refresh_edge(self, edge: ProfileHyperedgeUnit) -> None:
        facts = [self.memory.facts[fid] for fid in edge.member_fact_ids if fid in self.memory.facts]
        rep_facts = self._representative_facts(facts)
        name = str(edge.metadata.get("feature_name", edge.edge_type.value))
        desc = str(edge.metadata.get("feature_description", ""))
        evidence_preview = "；".join([fact.content.strip() for fact in rep_facts[:3] if fact.content.strip()])
        edge.summary = f"{name}: {desc}" if desc else name
        if evidence_preview:
            edge.summary = f"{edge.summary} | examples: {evidence_preview}"
        edge.keywords = self.memory.extract_keywords(" ".join([edge.summary] + [fact.text() for fact in rep_facts]))
        edge.embedding = self.memory.edge_embedding(edge.summary, rep_facts)
        edge.coherence_score = self._sampled_coherence(rep_facts)
        edge.stability_score = clamp(0.45 + 0.08 * math.log1p(len(facts)))
        if facts:
            max_ts = max([fact.timestamp for fact in self.memory.facts.values()] or [1.0])
            edge.freshness_score = clamp(0.35 + 0.65 * (max(fact.timestamp for fact in facts) / max_ts))
        edge.updated_at = time.time()
        edge.metadata["refresh_mode"] = "sampled_representative_refresh"
        edge.metadata["num_member_facts"] = len(facts)

    def _sampled_coherence(self, facts: Sequence[ProfileFact]) -> float:
        if len(facts) <= 1:
            return 0.55
        sims: List[float] = []
        checked = 0
        for i, left in enumerate(facts):
            step = max(1, len(facts) // 32)
            for j in range(i + 1, len(facts), step):
                right = facts[j]
                sims.append(HashedEmbeddingModel.cosine(left.embedding, right.embedding))
                checked += 1
                if checked >= self.max_coherence_pairs:
                    return clamp(sum(sims) / len(sims)) if sims else 0.55
        return clamp(sum(sims) / len(sims)) if sims else 0.55

    def consolidate_feature_bank(self, reason: str = "periodic") -> None:
        started = time.time()
        active = [edge for edge in self.memory.edges.values() if edge.status == "active"]
        merges = self._merge_duplicate_or_similar_edges(active)
        splits = self._split_oversized_edges()
        self.consolidation_merges += merges
        self.consolidation_splits += splits
        if merges or splits or self.show_progress:
            print(
                f"[consolidate] reason={reason} merges={merges} splits={splits} "
                f"active_edges={self.memory.active_edge_count()} elapsed={time.time() - started:.2f}s",
                flush=True,
            )

    def _merge_duplicate_or_similar_edges(self, active_edges: Sequence[ProfileHyperedgeUnit]) -> int:
        merges = 0
        by_type: Dict[str, List[ProfileHyperedgeUnit]] = {}
        for edge in active_edges:
            by_type.setdefault(str(edge.metadata.get("feature_type", edge.edge_type.value)), []).append(edge)
        for _, edges in by_type.items():
            for i, left in enumerate(list(edges)):
                if left.status != "active":
                    continue
                for right in edges[i + 1 :]:
                    if right.status != "active":
                        continue
                    if self._should_merge_edges(left, right):
                        self._append_members(left, right.member_fact_ids)
                        right.status = "merged"
                        right.metadata["merged_into"] = left.edge_id
                        self._fast_refresh_edge(left)
                        merges += 1
        return merges

    def _should_merge_edges(self, left: ProfileHyperedgeUnit, right: ProfileHyperedgeUnit) -> bool:
        if left.metadata.get("feature_signature") and left.metadata.get("feature_signature") == right.metadata.get("feature_signature"):
            return True
        left_text = " ".join([str(left.metadata.get("feature_name", "")), str(left.metadata.get("feature_description", "")), left.summary])
        right_text = " ".join([str(right.metadata.get("feature_name", "")), str(right.metadata.get("feature_description", "")), right.summary])
        overlap = keyword_overlap(left_text, right_text)
        emb_sim = HashedEmbeddingModel.cosine(left.embedding, right.embedding)
        return (0.45 * overlap + 0.55 * emb_sim) >= min(0.90, self.canonical_threshold + 0.10)

    def _split_oversized_edges(self) -> int:
        splits = 0
        active = [edge for edge in self.memory.edges.values() if edge.status == "active"]
        for edge in active:
            if len(edge.member_fact_ids) <= self.max_edge_facts:
                continue
            chunks = [edge.member_fact_ids[i : i + self.max_edge_facts] for i in range(0, len(edge.member_fact_ids), self.max_edge_facts)]
            if len(chunks) <= 1:
                continue
            base_name = str(edge.metadata.get("feature_name", edge.edge_type.value))
            for part_idx, chunk in enumerate(chunks, start=1):
                metadata = dict(edge.metadata)
                metadata.update({
                    "construction": "batch_profile_feature_induction_split_child",
                    "parent_edge_id": edge.edge_id,
                    "feature_signature": f"{edge.metadata.get('feature_signature', edge.edge_id)}:part:{part_idx}",
                    "feature_name": f"{base_name} #{part_idx}",
                    "split_reason": "max_edge_facts_boundary",
                })
                child = self.memory.create_edge(
                    edge.edge_type,
                    list(chunk),
                    summary=f"{base_name} #{part_idx}: split child of oversized feature edge",
                    confidence=edge.confidence_score,
                    metadata=metadata,
                )
                self._fast_refresh_edge(child)
            edge.status = "merged"
            edge.metadata["split_into_parts"] = len(chunks)
            splits += len(chunks)
        return splits


def build_profile_hypergraph_from_rows(
    memory: ProfileCentricHypergraphMemory,
    rows: Sequence[Dict[str, Any]],
    *,
    batch_size: int = 200,
    canonical_threshold: float = 0.62,
    min_feature_support: int = 1,
    consolidate_every: int = 5,
    max_edge_facts: int = 600,
    show_progress: bool = False,
) -> BatchProfileHypergraphBuilder:
    builder = BatchProfileHypergraphBuilder(
        memory=memory,
        batch_size=batch_size,
        canonical_threshold=canonical_threshold,
        min_feature_support=min_feature_support,
        consolidate_every=consolidate_every,
        max_edge_facts=max_edge_facts,
        show_progress=show_progress,
    )
    builder.build(rows)
    return builder
