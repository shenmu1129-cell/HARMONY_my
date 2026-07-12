"""HyperMem-style Topic/Episode/Fact hierarchy extraction.

This adapter keeps the public functions used by the behavioral hybrid pipeline,
but changes the internals from one-shot hierarchy generation to a staged flow:

    raw rows -> episodes -> streaming topics -> topic-based facts

The implementation is intentionally local to this repository. It uses the
existing DeepSeek/OpenAI-compatible runtime configuration and does not depend on
EverMind-AI/HyperMem as a package.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Sequence

from hypermem import load_runtime_env
from hypermem.llm_cache import JsonLLMCache
from hypermem.llm_profile_builder import OpenAI, _safe_json


EPISODE_ROLES = {
    "initiating",
    "developing",
    "climax",
    "concluding",
    "recurring",
    "background",
    "key_moment",
    "transition",
}
FACT_ROLES = {"core", "context", "detail", "temporal", "spatial", "causal"}


def _content(row: Any) -> str:
    if isinstance(row, str):
        return row
    if not isinstance(row, dict):
        return str(row)
    return str(
        row.get("content")
        or row.get("text")
        or row.get("fact")
        or row.get("summary")
        or row.get("message")
        or ""
    )


def _row_id(row: Any, idx: int) -> str:
    if isinstance(row, dict):
        return str(row.get("fact_id") or row.get("id") or row.get("message_id") or f"row_{idx + 1:06d}")
    return f"row_{idx + 1:06d}"


def _timestamp(row: Any, idx: int) -> float:
    if isinstance(row, dict):
        raw = row.get("timestamp") or row.get("time_index") or row.get("turn") or idx + 1
        try:
            return float(raw)
        except Exception:
            return float(idx + 1)
    return float(idx + 1)


def _clean_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x) for x in value if str(x).strip()]
    if isinstance(value, tuple):
        return [str(x) for x in value if str(x).strip()]
    text = str(value).strip()
    return [text] if text else []


def _clip(text: Any, n: int = 1200) -> str:
    s = str(text or "")
    return s[:n]


def normalize_input_rows(rows: Sequence[Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for i, row in enumerate(rows):
        text = _content(row).strip()
        if not text:
            continue
        out.append(
            {
                "row_id": _row_id(row, i),
                "content": text,
                "timestamp": _timestamp(row, i),
                "metadata": row if isinstance(row, dict) else {},
            }
        )
    return out


class HyperMemStyleHierarchyClient:
    """Small DeepSeek/OpenAI-compatible client with JSON cache."""

    def __init__(
        self,
        *,
        api_key_env: str = "DEEPSEEK_API_KEY",
        base_url: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        cache_dir: str = "outputs/llm_hierarchy_cache",
        use_cache: bool = True,
    ) -> None:
        if OpenAI is None:
            raise RuntimeError("openai package is required for LLM hierarchy extraction")
        load_runtime_env()
        key = os.getenv(api_key_env)
        if not key:
            raise RuntimeError(f"missing environment variable: {api_key_env}")
        self.client = OpenAI(api_key=key, base_url=base_url or os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"))
        self.model = model or os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
        self.temperature = float(temperature if temperature is not None else os.getenv("DEEPSEEK_TEMPERATURE", "0"))
        self.max_tokens = int(max_tokens or float(os.getenv("DEEPSEEK_MAX_TOKENS", "8192")))
        self.cache = JsonLLMCache(cache_dir=cache_dir, enabled=use_cache)
        self.prompt_version = "hypermem_style_hierarchy_v1"

    def _call_json(self, task: str, payload: Dict[str, Any], prompt: str) -> Dict[str, Any]:
        cache_payload = {"prompt_version": self.prompt_version, "task": task, "payload": payload}
        key = self.cache.make_key(cache_payload)
        cached = self.cache.get(key)
        if cached is None:
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    response_format={"type": "json_object"},
                )
            except TypeError:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
            raw = response.choices[0].message.content or ""
            parsed = _safe_json(raw)
            cached = {"raw": raw, "parsed": parsed}
            self.cache.set(key, cached)
        parsed = cached.get("parsed")
        return parsed if isinstance(parsed, dict) else {}

    def extract_episodes(self, rows: Sequence[Dict[str, Any]], batch_index: int) -> List[Dict[str, Any]]:
        payload = {
            "batch_index": batch_index,
            "rows": [
                {"row_id": r["row_id"], "timestamp": r["timestamp"], "content": _clip(r["content"], 1100)}
                for r in rows
            ],
        }
        prompt = (
            "You are an episodic memory construction expert. Convert the input memory rows into "
            "semantically complete episodes. An episode should be an independent event or coherent "
            "conversation segment, not just a broad topic. Preserve names, time clues, places, emotions, "
            "plans, decisions, and outcomes. Use the exact row_id values as evidence.\n\n"
            "Return strict JSON only with this schema:\n"
            "{\"episodes\":[{\"episode_id\":\"episode_001\",\"title\":\"specific title\","
            "\"subject\":\"searchable subject\",\"summary\":\"2-4 sentence summary\","
            "\"content\":\"detailed third-person episodic memory\",\"source_row_ids\":[\"row_id\"],"
            "\"timestamp\":1.0,\"keywords\":[\"keyword\"],\"participants\":[\"name\"]}]}\n\n"
            "Guidelines:\n"
            "- Group rows only when they describe the same concrete event/thread.\n"
            "- Avoid over-splitting greetings/supportive short replies from their event.\n"
            "- Avoid over-merging unrelated life aspects.\n"
            "- Every input row should appear in at least one episode.source_row_ids.\n\n"
            f"Input JSON:\n{json.dumps(payload, ensure_ascii=False)}"
        )
        data = self._call_json("episode_generation", payload, prompt)
        return data.get("episodes") if isinstance(data.get("episodes"), list) else []

    def create_topic(self, episodes: Sequence[Dict[str, Any]], topic_index: int) -> Dict[str, Any]:
        payload = {
            "topic_index": topic_index,
            "episodes": [_episode_view(e) for e in episodes],
        }
        prompt = (
            "You are a memory topic aggregation expert. Create ONE specific topic for the given episodes. "
            "A topic is a concrete event thread, activity line, project, relationship development, or ongoing situation. "
            "It must not be a vague category like personal updates, daily life, support, or work discussions.\n\n"
            "Return strict JSON only:\n"
            "{\"title\":\"specific topic title\",\"summary\":\"detailed topic memory\","
            "\"keywords\":[\"keyword\"],\"topic_type\":\"journey/event/project/relationship/etc\"}\n\n"
            f"Input JSON:\n{json.dumps(payload, ensure_ascii=False)}"
        )
        return self._call_json("topic_create", payload, prompt)

    def match_topics(self, episode: Dict[str, Any], topics: Sequence[Dict[str, Any]]) -> List[str]:
        payload = {
            "episode": _episode_view(episode),
            "topics": [
                {
                    "topic_id": t["topic_id"],
                    "title": t.get("title", ""),
                    "summary": _clip(t.get("summary", ""), 900),
                    "episode_count": len(t.get("episode_ids", [])),
                    "keywords": _clean_list(t.get("keywords"))[:20],
                }
                for t in topics
            ],
        }
        prompt = (
            "Determine whether the new episode belongs to any existing memory topics. "
            "Match only when the episode is a direct continuation, natural follow-up, or return to the SAME specific event thread. "
            "Do not match merely because it shares a broad category, the same people, or a general theme. "
            "When unsure, do not match. One episode may match multiple topics only if it genuinely bridges them.\n\n"
            "Return strict JSON only:\n"
            "{\"matched_topic_ids\":[\"topic_id\"],\"reasoning\":\"brief reason\"}\n\n"
            f"Input JSON:\n{json.dumps(payload, ensure_ascii=False)}"
        )
        data = self._call_json("topic_match", payload, prompt)
        allowed = {t["topic_id"] for t in topics}
        return [tid for tid in _clean_list(data.get("matched_topic_ids")) if tid in allowed]

    def update_topic(self, topic: Dict[str, Any], episode: Dict[str, Any]) -> Dict[str, Any]:
        payload = {
            "topic": {
                "topic_id": topic["topic_id"],
                "title": topic.get("title", ""),
                "summary": _clip(topic.get("summary", ""), 1800),
                "keywords": _clean_list(topic.get("keywords"))[:40],
            },
            "new_episode": _episode_view(episode),
        }
        prompt = (
            "Update the existing memory topic by incorporating the new episode. Maintain the topic's specific identity; "
            "do not broaden it into a generic category. Append new developments, preserve prior details, and add useful keywords.\n\n"
            "Return strict JSON only:\n"
            "{\"title\":\"specific topic title\",\"summary\":\"updated detailed topic memory\","
            "\"keywords\":[\"all useful keywords\"],\"update_note\":\"what was added\"}\n\n"
            f"Input JSON:\n{json.dumps(payload, ensure_ascii=False)}"
        )
        return self._call_json("topic_update", payload, prompt)

    def assign_episode_roles(self, topic: Dict[str, Any], episodes: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        payload = {
            "topic": {"topic_id": topic["topic_id"], "title": topic.get("title", ""), "summary": _clip(topic.get("summary", ""), 1200)},
            "episodes": [_episode_view(e) for e in episodes],
            "valid_roles": sorted(EPISODE_ROLES),
        }
        prompt = (
            "Assign each episode a role and importance weight within the topic. Roles must be one of the valid_roles. "
            "Weights are floats in [0,1], where higher means more essential to the topic. Include every episode_id exactly once.\n\n"
            "Return strict JSON only:\n"
            "{\"episode_roles\":[{\"episode_id\":\"...\",\"role\":\"initiating\",\"weight\":0.9}],"
            "\"coherence_score\":0.8,\"reasoning\":\"brief reason\"}\n\n"
            f"Input JSON:\n{json.dumps(payload, ensure_ascii=False)}"
        )
        return self._call_json("episode_role_weight", payload, prompt)

    def extract_facts(self, topic: Dict[str, Any], episodes: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        payload = {
            "topic": {"topic_id": topic["topic_id"], "title": topic.get("title", ""), "summary": _clip(topic.get("summary", ""), 1500)},
            "episodes": [_episode_view(e, content_chars=1600) for e in episodes],
        }
        prompt = (
            "Extract all self-contained, queryable facts from the topic and its associated episodes. "
            "Each fact must be independently understandable and faithful to the input. Preserve exact names, titles, places, numbers, "
            "time expressions, emotions, reasons, plans, and outcomes. Prefer several atomic facts over one vague summary. "
            "Use exact episode_id values as evidence.\n\n"
            "Return strict JSON only:\n"
            "{\"facts\":[{\"fact_id\":\"fact_1\",\"content\":\"complete atomic fact\","
            "\"episode_ids\":[\"episode_id\"],\"source_row_ids\":[\"row_id\"],\"confidence\":0.9,"
            "\"temporal\":null,\"spatial\":null,\"keywords\":[\"keyword\"],"
            "\"query_patterns\":[\"question this fact can answer\"]}],\"reasoning\":\"brief strategy\"}\n\n"
            f"Input JSON:\n{json.dumps(payload, ensure_ascii=False)}"
        )
        data = self._call_json("fact_extraction", payload, prompt)
        return data.get("facts") if isinstance(data.get("facts"), list) else []

    def assign_fact_roles(self, topic: Dict[str, Any], facts: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        payload = {
            "topic": {"topic_id": topic["topic_id"], "title": topic.get("title", ""), "summary": _clip(topic.get("summary", ""), 1000)},
            "facts": [
                {"fact_id": f["fact_id"], "content": _clip(f.get("content", ""), 900), "episode_ids": f.get("episode_ids", [])}
                for f in facts
            ],
            "valid_roles": sorted(FACT_ROLES),
        }
        prompt = (
            "Assign each fact a role and importance weight within the topic. Roles must be one of valid_roles. "
            "Weights are floats in [0,1]. Include every fact_id exactly once.\n\n"
            "Return strict JSON only:\n"
            "{\"fact_roles\":[{\"fact_id\":\"...\",\"role\":\"core\",\"weight\":0.9}],"
            "\"extraction_confidence\":0.8,\"reasoning\":\"brief reason\"}\n\n"
            f"Input JSON:\n{json.dumps(payload, ensure_ascii=False)}"
        )
        return self._call_json("fact_role_weight", payload, prompt)


def _episode_view(episode: Dict[str, Any], content_chars: int = 1000) -> Dict[str, Any]:
    return {
        "episode_id": episode.get("episode_id", ""),
        "title": episode.get("title", ""),
        "subject": episode.get("subject", episode.get("title", "")),
        "summary": episode.get("summary", ""),
        "content": _clip(episode.get("content") or episode.get("episode_description") or episode.get("summary") or "", content_chars),
        "source_row_ids": _clean_list(episode.get("source_row_ids")),
        "timestamp": episode.get("timestamp"),
        "keywords": _clean_list(episode.get("keywords"))[:20],
    }


def _fallback_hierarchy(rows: Sequence[Dict[str, Any]], batch_size: int = 40) -> Dict[str, Any]:
    episodes: List[Dict[str, Any]] = []
    topics: List[Dict[str, Any]] = []
    facts: List[Dict[str, Any]] = []
    episode_hyperedges: Dict[str, Dict[str, Any]] = {}
    fact_hyperedges: Dict[str, Dict[str, Any]] = {}

    for batch_index in range((len(rows) + batch_size - 1) // max(1, batch_size)):
        batch = rows[batch_index * batch_size : (batch_index + 1) * batch_size]
        if not batch:
            continue
        episode_id = f"episode_{batch_index + 1:04d}"
        topic_id = f"topic_{batch_index + 1:04d}"
        episode = {
            "episode_id": episode_id,
            "title": f"Fallback episode {batch_index + 1}",
            "subject": f"Fallback episode {batch_index + 1}",
            "summary": "Fallback episode wrapping already extracted memory rows.",
            "content": " ".join(r["content"] for r in batch),
            "source_row_ids": [r["row_id"] for r in batch],
            "timestamp": batch[0]["timestamp"],
            "keywords": [],
            "participants": [],
        }
        episodes.append(episode)
        topic_facts = []
        for i, row in enumerate(batch):
            fact = {
                "fact_id": f"fact_{batch_index + 1:04d}_{i + 1:04d}",
                "content": row["content"],
                "episode_ids": [episode_id],
                "topic_id": topic_id,
                "confidence": 0.8,
                "temporal": None,
                "spatial": None,
                "keywords": [],
                "query_patterns": [],
                "source_row_ids": [row["row_id"]],
                "role": "detail",
                "weight": 0.5,
            }
            facts.append(fact)
            topic_facts.append(fact)
        topic = {
            "topic_id": topic_id,
            "title": f"Fallback topic {batch_index + 1}",
            "summary": "Fallback topic wrapping already extracted memory rows.",
            "keywords": [],
            "episode_ids": [episode_id],
            "episodes": [episode],
            "facts": topic_facts,
        }
        topics.append(topic)
        episode_hyperedges[f"episode_hyperedge_{topic_id}"] = {
            "id": f"episode_hyperedge_{topic_id}",
            "relation": {episode_id: "initiating"},
            "weights": {episode_id: 1.0},
            "topic_node_id": topic_id,
            "coherence_score": 0.8,
        }
        fact_hyperedges[f"fact_hyperedge_{episode_id}"] = {
            "id": f"fact_hyperedge_{episode_id}",
            "relation": {f["fact_id"]: "detail" for f in topic_facts},
            "weights": {f["fact_id"]: 0.5 for f in topic_facts},
            "episode_node_id": episode_id,
            "extraction_confidence": 0.8,
        }

    return _make_hierarchy(rows, episodes, topics, facts, episode_hyperedges, fact_hyperedges, fallback=True)


def _make_hierarchy(
    rows: Sequence[Dict[str, Any]],
    episodes: List[Dict[str, Any]],
    topics: List[Dict[str, Any]],
    facts: List[Dict[str, Any]],
    episode_hyperedges: Dict[str, Dict[str, Any]],
    fact_hyperedges: Dict[str, Dict[str, Any]],
    *,
    fallback: bool = False,
) -> Dict[str, Any]:
    return {
        "schema": "hypermem_style_topic_episode_fact_tree_v2",
        "builder": "fallback" if fallback else "deepseek_hypermem_style_adapter",
        "num_source_rows": len(rows),
        "num_episodes": len(episodes),
        "num_topics": len(topics),
        "num_facts": len(facts),
        "source_rows": {row["row_id"]: row for row in rows},
        "episodes": episodes,
        "topics": topics,
        "facts": facts,
        "episode_hyperedges": episode_hyperedges,
        "fact_hyperedges": fact_hyperedges,
    }


def _normalize_episode(item: Dict[str, Any], rows: Sequence[Dict[str, Any]], index: int) -> Dict[str, Any]:
    valid_ids = {r["row_id"] for r in rows}
    source_ids = [rid for rid in _clean_list(item.get("source_row_ids")) if rid in valid_ids]
    if not source_ids:
        source_ids = [rows[min(index, len(rows) - 1)]["row_id"]]
    row_map = {r["row_id"]: r for r in rows}
    first_row = row_map[source_ids[0]]
    title = str(item.get("title") or item.get("subject") or f"Episode {index + 1}")
    summary = str(item.get("summary") or item.get("content") or title)
    content = str(item.get("content") or item.get("episode_description") or summary)
    return {
        "episode_id": str(item.get("episode_id") or f"episode_{index + 1:05d}"),
        "title": title,
        "subject": str(item.get("subject") or title),
        "summary": summary,
        "content": content,
        "episode_description": content,
        "source_row_ids": source_ids,
        "timestamp": item.get("timestamp", first_row.get("timestamp")),
        "keywords": _clean_list(item.get("keywords")),
        "participants": _clean_list(item.get("participants")),
    }


def _new_topic_from_response(data: Dict[str, Any], topic_id: str, episode: Dict[str, Any]) -> Dict[str, Any]:
    title = str(data.get("title") or episode.get("title") or topic_id)
    summary = str(data.get("summary") or episode.get("summary") or title)
    return {
        "topic_id": topic_id,
        "title": title,
        "summary": summary,
        "keywords": _clean_list(data.get("keywords")),
        "topic_type": data.get("topic_type", "auto"),
        "episode_ids": [episode["episode_id"]],
        "episodes": [episode],
        "facts": [],
    }


def _update_topic_in_place(topic: Dict[str, Any], data: Dict[str, Any], episode: Dict[str, Any]) -> None:
    topic["title"] = str(data.get("title") or topic.get("title") or topic["topic_id"])
    topic["summary"] = str(data.get("summary") or topic.get("summary") or "")
    merged_keywords = list(dict.fromkeys(_clean_list(topic.get("keywords")) + _clean_list(data.get("keywords"))))
    topic["keywords"] = merged_keywords
    if episode["episode_id"] not in topic["episode_ids"]:
        topic["episode_ids"].append(episode["episode_id"])
    if episode["episode_id"] not in {e["episode_id"] for e in topic.get("episodes", [])}:
        topic.setdefault("episodes", []).append(episode)


def _parse_episode_roles(data: Dict[str, Any], episodes: Sequence[Dict[str, Any]]) -> tuple[Dict[str, str], Dict[str, float], float]:
    episode_ids = [e["episode_id"] for e in episodes]
    relation: Dict[str, str] = {}
    weights: Dict[str, float] = {}
    for item in data.get("episode_roles", []) if isinstance(data.get("episode_roles"), list) else []:
        eid = str(item.get("episode_id") or "")
        if eid not in episode_ids:
            continue
        role = str(item.get("role") or "developing")
        if role not in EPISODE_ROLES:
            role = "developing"
        try:
            weight = float(item.get("weight", 0.5))
        except Exception:
            weight = 0.5
        relation[eid] = role
        weights[eid] = max(0.0, min(1.0, weight))
    for i, eid in enumerate(episode_ids):
        relation.setdefault(eid, "initiating" if i == 0 else "developing")
        weights.setdefault(eid, 0.85 if i == 0 else 0.65)
    try:
        coherence = float(data.get("coherence_score", 0.8))
    except Exception:
        coherence = 0.8
    return relation, weights, max(0.0, min(1.0, coherence))


def _normalize_fact(item: Dict[str, Any], topic: Dict[str, Any], episode_map: Dict[str, Dict[str, Any]], fact_index: int) -> Dict[str, Any] | None:
    content = str(item.get("content") or "").strip()
    if not content:
        return None
    valid_episode_ids = set(episode_map.keys())
    episode_ids = [eid for eid in _clean_list(item.get("episode_ids")) if eid in valid_episode_ids]
    if not episode_ids:
        # Best-effort fallback: attach to every episode in the topic.
        episode_ids = list(topic.get("episode_ids", []))[:1]
    source_row_ids: List[str] = []
    for eid in episode_ids:
        source_row_ids.extend(_clean_list(episode_map[eid].get("source_row_ids")))
    explicit_source_ids = _clean_list(item.get("source_row_ids"))
    if explicit_source_ids:
        source_row_ids = explicit_source_ids
    try:
        confidence = float(item.get("confidence", 0.8))
    except Exception:
        confidence = 0.8
    return {
        "fact_id": str(item.get("fact_id") or f"fact_{fact_index:06d}"),
        "content": content,
        "episode_ids": episode_ids,
        "topic_id": topic["topic_id"],
        "confidence": max(0.0, min(1.0, confidence)),
        "temporal": item.get("temporal"),
        "spatial": item.get("spatial"),
        "keywords": _clean_list(item.get("keywords")),
        "query_patterns": _clean_list(item.get("query_patterns")),
        "source_row_ids": list(dict.fromkeys(source_row_ids)),
    }


def _parse_fact_roles(data: Dict[str, Any], facts: Sequence[Dict[str, Any]]) -> tuple[Dict[str, str], Dict[str, float], float]:
    fact_ids = {f["fact_id"] for f in facts}
    relation: Dict[str, str] = {}
    weights: Dict[str, float] = {}
    for item in data.get("fact_roles", []) if isinstance(data.get("fact_roles"), list) else []:
        fid = str(item.get("fact_id") or "")
        if fid not in fact_ids:
            continue
        role = str(item.get("role") or "detail")
        if role not in FACT_ROLES:
            role = "detail"
        try:
            weight = float(item.get("weight", 0.5))
        except Exception:
            weight = 0.5
        relation[fid] = role
        weights[fid] = max(0.0, min(1.0, weight))
    for fact in facts:
        fid = fact["fact_id"]
        relation.setdefault(fid, "core" if not relation else "detail")
        weights.setdefault(fid, 0.75 if relation[fid] == "core" else 0.55)
    try:
        confidence = float(data.get("extraction_confidence", 0.8))
    except Exception:
        confidence = 0.8
    return relation, weights, max(0.0, min(1.0, confidence))


def extract_topic_episode_fact_hierarchy(
    rows: Sequence[Any],
    *,
    batch_size: int = 40,
    use_llm: bool = True,
    show_progress: bool = True,
) -> Dict[str, Any]:
    normalized = normalize_input_rows(rows)
    if not normalized:
        return _make_hierarchy([], [], [], [], {}, {})
    if not use_llm:
        return _fallback_hierarchy(normalized, batch_size=batch_size)

    client = HyperMemStyleHierarchyClient()

    # Stage 1: raw rows -> episodes.
    episodes: List[Dict[str, Any]] = []
    total_batches = (len(normalized) + batch_size - 1) // max(1, batch_size)
    for batch_index in range(total_batches):
        batch = normalized[batch_index * batch_size : (batch_index + 1) * batch_size]
        if show_progress:
            print(f"[hierarchy] stage=episode batch={batch_index + 1}/{total_batches} rows={len(batch)}", flush=True)
        raw_episodes = client.extract_episodes(batch, batch_index)
        if not raw_episodes:
            raw_episodes = [_fallback_hierarchy(batch, batch_size=len(batch))["episodes"][0]]
        for item in raw_episodes:
            ep = _normalize_episode(item, batch, len(episodes))
            # Ensure globally unique episode_id.
            ep["episode_id"] = f"episode_{len(episodes) + 1:05d}"
            episodes.append(ep)

    # Stage 2: streaming topic matching/create/update.
    topics: List[Dict[str, Any]] = []
    topic_map: Dict[str, Dict[str, Any]] = {}
    for i, episode in enumerate(episodes):
        if show_progress:
            print(f"[hierarchy] stage=topic episode={i + 1}/{len(episodes)} active_topics={len(topics)}", flush=True)
        if not topics:
            topic_id = f"topic_{len(topics) + 1:05d}"
            data = client.create_topic([episode], len(topics))
            topic = _new_topic_from_response(data, topic_id, episode)
            topics.append(topic)
            topic_map[topic_id] = topic
            continue

        matched_ids = client.match_topics(episode, topics)
        if not matched_ids:
            topic_id = f"topic_{len(topics) + 1:05d}"
            data = client.create_topic([episode], len(topics))
            topic = _new_topic_from_response(data, topic_id, episode)
            topics.append(topic)
            topic_map[topic_id] = topic
        else:
            for tid in matched_ids:
                topic = topic_map[tid]
                data = client.update_topic(topic, episode)
                _update_topic_in_place(topic, data, episode)

    # Build episode hyperedges after topic aggregation.
    episode_hyperedges: Dict[str, Dict[str, Any]] = {}
    for topic in topics:
        topic_episodes = topic.get("episodes", [])
        data = client.assign_episode_roles(topic, topic_episodes)
        relation, weights, coherence = _parse_episode_roles(data, topic_episodes)
        edge_id = f"episode_hyperedge_{topic['topic_id']}"
        episode_hyperedges[edge_id] = {
            "id": edge_id,
            "relation": relation,
            "weights": weights,
            "topic_node_id": topic["topic_id"],
            "coherence_score": coherence,
        }

    # Stage 3: topic-based fact extraction and fact hyperedges.
    all_facts: List[Dict[str, Any]] = []
    fact_hyperedges: Dict[str, Dict[str, Any]] = {}
    episode_map = {e["episode_id"]: e for e in episodes}
    seen_fact_ids: set[str] = set()
    for ti, topic in enumerate(topics):
        if show_progress:
            print(f"[hierarchy] stage=fact topic={ti + 1}/{len(topics)} episodes={len(topic.get('episodes', []))}", flush=True)
        raw_facts = client.extract_facts(topic, topic.get("episodes", []))
        topic_facts: List[Dict[str, Any]] = []
        for item in raw_facts:
            fact = _normalize_fact(item, topic, episode_map, len(all_facts) + 1)
            if fact is None:
                continue
            if fact["fact_id"] in seen_fact_ids:
                fact["fact_id"] = f"fact_{len(all_facts) + 1:06d}"
            seen_fact_ids.add(fact["fact_id"])
            topic_facts.append(fact)
            all_facts.append(fact)
        if not topic_facts:
            # Do not leave a topic without evidence.
            for episode in topic.get("episodes", []):
                for rid in episode.get("source_row_ids", [])[:2]:
                    row = next((r for r in normalized if r["row_id"] == rid), None)
                    if not row:
                        continue
                    fact = {
                        "fact_id": f"fact_{len(all_facts) + 1:06d}",
                        "content": row["content"],
                        "episode_ids": [episode["episode_id"]],
                        "topic_id": topic["topic_id"],
                        "confidence": 0.7,
                        "temporal": None,
                        "spatial": None,
                        "keywords": [],
                        "query_patterns": [],
                        "source_row_ids": [rid],
                    }
                    topic_facts.append(fact)
                    all_facts.append(fact)
        role_data = client.assign_fact_roles(topic, topic_facts)
        fact_roles, fact_weights, extraction_confidence = _parse_fact_roles(role_data, topic_facts)
        for fact in topic_facts:
            fact["role"] = fact_roles.get(fact["fact_id"], "detail")
            fact["weight"] = fact_weights.get(fact["fact_id"], 0.5)
        topic["facts"] = topic_facts
        for episode_id in topic.get("episode_ids", []):
            related = [f for f in topic_facts if episode_id in f.get("episode_ids", [])]
            if not related:
                continue
            edge_id = f"fact_hyperedge_{episode_id}"
            existing = fact_hyperedges.get(edge_id)
            relation = {f["fact_id"]: fact_roles.get(f["fact_id"], "detail") for f in related}
            weights = {f["fact_id"]: fact_weights.get(f["fact_id"], 0.5) for f in related}
            if existing:
                existing["relation"].update(relation)
                existing["weights"].update(weights)
            else:
                fact_hyperedges[edge_id] = {
                    "id": edge_id,
                    "relation": relation,
                    "weights": weights,
                    "episode_node_id": episode_id,
                    "extraction_confidence": extraction_confidence,
                }

    return _make_hierarchy(normalized, episodes, topics, all_facts, episode_hyperedges, fact_hyperedges)


def flatten_hierarchy_facts(hierarchy: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    idx = 0
    episode_lookup = {e.get("episode_id"): e for e in hierarchy.get("episodes", [])}
    for topic in hierarchy.get("topics", []):
        topic_id = str(topic.get("topic_id") or f"topic_{len(rows) + 1:05d}")
        topic_facts = topic.get("facts") if isinstance(topic.get("facts"), list) else []
        for fact in topic_facts:
            content = str(fact.get("content") or "").strip()
            if not content:
                continue
            idx += 1
            episode_ids = _clean_list(fact.get("episode_ids"))
            primary_episode = episode_lookup.get(episode_ids[0]) if episode_ids else None
            rows.append(
                {
                    "fact_id": str(fact.get("fact_id") or f"fact_{idx:06d}"),
                    "content": content,
                    "keywords": _clean_list(fact.get("keywords")),
                    "timestamp": float(idx),
                    "topic_id": topic_id,
                    "topic_title": topic.get("title", ""),
                    "topic_summary": topic.get("summary", ""),
                    "episode_id": episode_ids[0] if episode_ids else "",
                    "episode_title": (primary_episode or {}).get("title", ""),
                    "episode_summary": (primary_episode or {}).get("summary", ""),
                    "source_row_ids": _clean_list(fact.get("source_row_ids")),
                    "metadata": {
                        "topic": topic,
                        "episode": primary_episode or {},
                        "fact": fact,
                        "fact_role": fact.get("role", "detail"),
                        "fact_weight": fact.get("weight", 0.5),
                        "all_episode_ids": episode_ids,
                    },
                }
            )
    return rows


def save_hierarchy_outputs(hierarchy: Dict[str, Any], output_dir: str | Path) -> List[Dict[str, Any]]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    facts = flatten_hierarchy_facts(hierarchy)
    (out_dir / "topic_episode_fact_tree.json").write_text(json.dumps(hierarchy, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "episodes.json").write_text(json.dumps(hierarchy.get("episodes", []), ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "topics.json").write_text(json.dumps(hierarchy.get("topics", []), ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "facts.json").write_text(json.dumps(hierarchy.get("facts", []), ensure_ascii=False, indent=2), encoding="utf-8")
    with (out_dir / "hierarchical_facts.jsonl").open("w", encoding="utf-8") as f:
        for row in facts:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return facts
