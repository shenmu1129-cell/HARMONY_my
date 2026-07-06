"""English LLM profile-feature induction for LoCoMo-style data.

The semantic hyperedge is defined by the LLM-induced feature_name,
feature_type, description, triggers, and member facts. The coarse internal
edge_type is only a backward-compatible implementation field.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Sequence, Set

from hypermem import load_runtime_env
from hypermem.llm_cache import JsonLLMCache
from hypermem.llm_profile_builder import (
    LLMBatchProfileHypergraphBuilder,
    LLMFeature,
    OpenAI,
    _dedupe,
    _safe_json,
)
from hypermem.profile_centric_hypergraph import ProfileCentricHypergraphMemory, ProfileEdgeType, ProfileFact, ProfileHyperedgeUnit, clamp


EXAMPLE_PROFILE_DIMENSIONS = [
    "identity_and_values",
    "preference_pattern",
    "habit_or_routine",
    "relationship_dynamic",
    "family_role",
    "work_or_study_context",
    "health_and_wellbeing_pattern",
    "life_event",
    "goal_or_plan",
    "emotion_or_current_state",
    "communication_style",
    "activity_or_hobby",
    "temporal_update",
    "tool_or_workflow",
    "domain_knowledge",
    "research_focus",
    "writing_or_reporting_style",
]


# Exact compatibility mapping only. Do not infer coarse edge_type from free-form
# feature names/descriptions; the open-set feature itself is the semantic label.
EXACT_FEATURE_TYPE_TO_EDGE_TYPE = {
    "tool_or_workflow": ProfileEdgeType.TOOL_USAGE,
    "domain_knowledge": ProfileEdgeType.DOMAIN_KNOWLEDGE,
    "research_focus": ProfileEdgeType.RESEARCH_FOCUS,
    "writing_or_reporting_style": ProfileEdgeType.WRITING_STYLE,
    "temporal_update": ProfileEdgeType.TEMPORAL_EVOLUTION,
    "life_event": ProfileEdgeType.TEMPORAL_EVOLUTION,
}


def _normalize_feature_type(feature_type: str) -> str:
    return feature_type.strip().lower().replace(" ", "_").replace("-", "_")


def _edge_type_en(feature_type: str, name: str, desc: str) -> ProfileEdgeType:
    del name, desc
    normalized_type = _normalize_feature_type(feature_type)
    return EXACT_FEATURE_TYPE_TO_EDGE_TYPE.get(normalized_type, ProfileEdgeType.AUTO_DISCOVERED)


class EnglishLLMFeatureClient:
    def __init__(
        self,
        api_key_env: str,
        base_url: str,
        model: str,
        temperature: float,
        max_features: int,
        max_features_per_fact: int,
        max_tokens: int,
        cache_dir: str,
        use_cache: bool,
    ) -> None:
        if OpenAI is None:
            raise RuntimeError("openai package is required for English LLM profile construction")
        load_runtime_env()
        key = os.getenv(api_key_env)
        if not key:
            raise RuntimeError(f"missing environment variable: {api_key_env}")
        self.client = OpenAI(api_key=key, base_url=base_url)
        self.model = model
        self.temperature = temperature
        self.max_features = max_features
        self.max_features_per_fact = max_features_per_fact
        self.max_tokens = max_tokens
        self.cache = JsonLLMCache(cache_dir=cache_dir, enabled=use_cache)
        self.prompt_version = "english_open_profile_induction_v5"

    def induce(self, facts: Sequence[ProfileFact], existing_edges: Sequence[ProfileHyperedgeUnit]) -> List[LLMFeature]:
        valid_ids = {fact.fact_id for fact in facts}
        payload = {
            "prompt_version": self.prompt_version,
            "task": "open_set_profile_hyperedge_induction",
            "model": self.model,
            "facts": [
                {"fact_id": fact.fact_id, "content": fact.content[:900], "keywords": fact.keywords[:8]}
                for fact in facts
            ],
            "existing_features": [
                {
                    "edge_id": edge.edge_id,
                    "feature_name": edge.metadata.get("feature_name", edge.summary[:80]),
                    "feature_type": edge.metadata.get("feature_type", edge.edge_type.value),
                    "description": edge.metadata.get("feature_description", edge.summary[:180]),
                    "members": len(edge.member_fact_ids),
                }
                for edge in existing_edges[:80]
                if edge.status == "active"
            ],
            "limits": {
                "max_features": self.max_features,
                "max_features_per_fact": self.max_features_per_fact,
                "allow_multi_membership": True,
                "target_fact_coverage_ratio": 0.80,
                "target_multi_membership_ratio": 0.25,
            },
            "example_profile_dimensions": EXAMPLE_PROFILE_DIMENSIONS,
        }
        cache_key = self.cache.make_key(payload)
        cached = self.cache.get(cache_key)
        if cached is None:
            prompt = self._build_induction_prompt(payload)
            data, raw = self._call_json(prompt, max_tokens=self.max_tokens, temperature=self.temperature)
            features = self._parse_features(data, valid_ids)
            if not features:
                repair_prompt = self._build_repair_prompt(payload, raw)
                data, raw = self._call_json(repair_prompt, max_tokens=self.max_tokens, temperature=0.0)
            cached = {"raw": raw, "parsed": data}
            self.cache.set(cache_key, cached)
        data = cached.get("parsed") or {}
        return self._parse_features(data, valid_ids)

    def suggest_merges(self, edges: Sequence[ProfileHyperedgeUnit]) -> List[List[str]]:
        active = [edge for edge in edges if edge.status == "active"]
        if len(active) <= 1:
            return []
        payload = {
            "prompt_version": self.prompt_version,
            "task": "merge_duplicate_profile_hyperedges",
            "model": self.model,
            "features": [
                {
                    "edge_id": edge.edge_id,
                    "feature_name": edge.metadata.get("feature_name", edge.summary[:80]),
                    "feature_type": edge.metadata.get("feature_type", edge.edge_type.value),
                    "description": edge.metadata.get("feature_description", edge.summary[:180]),
                    "members": len(edge.member_fact_ids),
                }
                for edge in active[:120]
            ],
        }
        cache_key = self.cache.make_key(payload)
        cached = self.cache.get(cache_key)
        if cached is None:
            prompt = (
                "You maintain an open-set profile-centric memory hypergraph. Find features that are semantically duplicate "
                "or whose boundaries strongly overlap. Return strict JSON only: "
                "{\"merge_groups\":[[\"edge_id_a\",\"edge_id_b\"]]}.\n\n"
                f"Input JSON:\n{json.dumps(payload, ensure_ascii=False)}"
            )
            data, raw = self._call_json(prompt, max_tokens=min(self.max_tokens, 2048), temperature=0.0)
            cached = {"raw": raw, "parsed": data}
            self.cache.set(cache_key, cached)
        data = cached.get("parsed") or {}
        return [group for group in data.get("merge_groups", []) if isinstance(group, list) and len(group) >= 2]

    def _call_json(self, prompt: str, max_tokens: int, temperature: float) -> tuple[Dict[str, Any], str]:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        raw = response.choices[0].message.content or ""
        return _safe_json(raw), raw

    def _build_induction_prompt(self, payload: Dict[str, Any]) -> str:
        return (
            "You are building a long-term user-memory hypergraph for English conversational QA. "
            "Given a batch of memory facts, induce shared user-profile features. This is an open-set induction task, "
            "not a closed-set classification task. Do not merely assign facts to predefined categories. "
            "Create concrete, human-readable common features from the facts themselves. "
            "A single fact may belong to multiple features when it expresses multiple user aspects. "
            "Prefer English feature names and English descriptions. "
            "Use feature_type as an open-ended, concise snake_case semantic label created by you. "
            "The example_profile_dimensions are examples only; you may create a more specific feature_type whenever appropriate. "
            "Good feature_type examples include caregiver_identity, community_belonging, emotional_support_pattern, "
            "creative_self_expression, single_parent_adoption_planning, conflict_avoidance_tendency, or health_advocacy_activity. "
            "Avoid overly broad feature names or types such as research, time, emotion, tool, goal, preference, state, misc, or other. "
            "Each feature must have a clear boundary using positive_triggers and negative_triggers. "
            "Coverage requirement: assign at least 80% of current batch facts to one or more features unless a fact is truly unprofiled. "
            "Multi-membership requirement: when a fact expresses multiple user aspects, assign it to 2-3 relevant features. "
            "Reuse requirement: if a current-batch fact matches an existing feature, include that existing edge_id as feature_id and assign the fact to it. "
            "Creation requirement: if a fact does not match existing features, create a specific new feature. "
            "Return between 4 and max_features features when the batch contains enough profile signals. "
            "Prefer features supported by at least two facts, but allow a singleton feature when it is a specific and important profile signal. "
            "Never return an empty features array unless there are no user-profile signals in the entire batch.\n\n"
            "Return strict JSON only in this schema:\n"
            "{\"features\":[{\"feature_id\":\"new_or_existing_id\",\"feature_name\":\"specific English common feature\","
            "\"feature_type\":\"open_ended_snake_case_type\",\"description\":\"English boundary description\","
            "\"positive_triggers\":[\"English trigger\"],\"negative_triggers\":[\"English exclusion\"],"
            "\"assigned_fact_ids\":[\"fact_id\"],\"confidence\":0.75}]}\n\n"
            f"Input JSON:\n{json.dumps(payload, ensure_ascii=False)}"
        )

    def _build_repair_prompt(self, payload: Dict[str, Any], previous_raw: str) -> str:
        return (
            "The previous response produced no valid features, but this batch should be processed. "
            "Regenerate strict JSON only. You must induce open-set user-profile features from the current batch. "
            "Reuse existing edge_id values when the batch facts fit existing features; otherwise create new feature_id values. "
            "Cover at least 80% of facts when possible. Allow multi-membership. Do not return an empty features array.\n\n"
            "Required schema:\n"
            "{\"features\":[{\"feature_id\":\"new_or_existing_id\",\"feature_name\":\"specific English common feature\","
            "\"feature_type\":\"open_ended_snake_case_type\",\"description\":\"English boundary description\","
            "\"positive_triggers\":[\"English trigger\"],\"negative_triggers\":[\"English exclusion\"],"
            "\"assigned_fact_ids\":[\"fact_id\"],\"confidence\":0.75}]}\n\n"
            f"Previous raw response:\n{previous_raw[:2000]}\n\n"
            f"Input JSON:\n{json.dumps(payload, ensure_ascii=False)}"
        )

    def _parse_features(self, data: Dict[str, Any], valid_ids: Set[str]) -> List[LLMFeature]:
        out: List[LLMFeature] = []
        for index, row in enumerate(data.get("features") or []):
            if not isinstance(row, dict):
                continue
            name = str(row.get("feature_name") or "").strip()
            feature_type = str(row.get("feature_type") or "open_profile_feature").strip()
            desc = str(row.get("description") or "").strip()
            assigned = [fact_id for fact_id in _dedupe(row.get("assigned_fact_ids") or []) if fact_id in valid_ids]
            if not name or not desc or not assigned:
                continue
            try:
                confidence = float(row.get("confidence", 0.55))
            except Exception:
                confidence = 0.55
            out.append(
                LLMFeature(
                    feature_id=str(row.get("feature_id") or f"llm_feature_{index+1:03d}"),
                    feature_name=name,
                    feature_type=feature_type,
                    description=desc,
                    assigned_fact_ids=assigned,
                    positive_triggers=_dedupe(row.get("positive_triggers") or []),
                    negative_triggers=_dedupe(row.get("negative_triggers") or []),
                    confidence=clamp(confidence, lo=0.0, hi=0.95),
                    edge_type=_edge_type_en(feature_type, name, desc),
                )
            )
        return out[: self.max_features]


class EnglishLLMBatchProfileHypergraphBuilder(LLMBatchProfileHypergraphBuilder):
    def __init__(self, memory: ProfileCentricHypergraphMemory, **kwargs: Any) -> None:
        load_runtime_env()
        kwargs.setdefault("api_key_env", "DEEPSEEK_API_KEY")
        kwargs.setdefault("base_url", os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"))
        kwargs.setdefault("model", os.getenv("DEEPSEEK_MODEL", "deepseek-chat"))
        kwargs.setdefault("temperature", float(os.getenv("DEEPSEEK_TEMPERATURE", "0.0")))
        max_tokens = int(float(os.getenv("DEEPSEEK_MAX_TOKENS", "8192")))
        cache_dir = kwargs.pop("cache_dir", os.getenv("LLM_PROFILE_CACHE_DIR", "outputs/llm_profile_cache"))
        use_cache = bool(int(os.getenv("LLM_PROFILE_USE_CACHE", "1")))
        super().__init__(memory, **kwargs)
        self.client = EnglishLLMFeatureClient(
            kwargs.get("api_key_env", "DEEPSEEK_API_KEY"),
            kwargs.get("base_url", os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")),
            kwargs.get("model", os.getenv("DEEPSEEK_MODEL", "deepseek-chat")),
            kwargs.get("temperature", float(os.getenv("DEEPSEEK_TEMPERATURE", "0.0"))),
            kwargs.get("max_features_per_batch", 12),
            kwargs.get("max_features_per_fact", 4),
            max_tokens,
            cache_dir,
            use_cache,
        )


def build_english_llm_profile_hypergraph_from_rows(
    memory: ProfileCentricHypergraphMemory,
    rows: Sequence[Dict[str, Any]],
    **kwargs: Any,
) -> EnglishLLMBatchProfileHypergraphBuilder:
    builder = EnglishLLMBatchProfileHypergraphBuilder(memory, **kwargs)
    builder.build(rows)
    return builder
