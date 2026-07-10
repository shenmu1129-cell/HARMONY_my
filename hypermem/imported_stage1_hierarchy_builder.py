"""Reuse official HyperMem Stage-1 episode outputs when available.

If an official Stage-1 episodes directory is available, this wrapper skips local
episode extraction and starts from topic aggregation + fact extraction. For smoke
tests it limits imported episodes to the number of requested input rows unless
HYPERMEM_MAX_IMPORTED_EPISODES is explicitly set.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Sequence

import hypermem.hypermem_style_hierarchy_builder as base

DEFAULT_EXPERIMENT = "base_sum_alpha0.5_RRF_fix-100-10-10-30_r011_wo-rerank_cot"

normalize_input_rows = base.normalize_input_rows
flatten_hierarchy_facts = base.flatten_hierarchy_facts
save_hierarchy_outputs = base.save_hierarchy_outputs
HyperMemStyleHierarchyClient = base.HyperMemStyleHierarchyClient


def _as_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x) for x in value if str(x).strip()]
    if isinstance(value, tuple):
        return [str(x) for x in value if str(x).strip()]
    text = str(value).strip()
    return [text] if text else []


def _candidate_episode_dirs() -> List[Path]:
    exp = os.getenv("HYPERMEM_EXPERIMENT_NAME", DEFAULT_EXPERIMENT)
    cwd = Path.cwd()
    candidates: List[Path] = []
    for key in ["HYPERMEM_STAGE1_EPISODES_DIR", "HYPERMEM_IMPORTED_EPISODES_DIR"]:
        value = os.getenv(key)
        if value:
            candidates.append(Path(value).expanduser())
    candidates.extend(
        [
            cwd / "results" / exp / "episodes",
            cwd.parent / "hyperMem_my" / "results" / exp / "episodes",
            cwd.parent / "HyperMem_official" / "results" / exp / "episodes",
            Path("/home/sutongtong/wwt/code/hyperMem_my/results") / exp / "episodes",
            Path("/home/sutongtong/wwt/code/HyperMem_official/results") / exp / "episodes",
        ]
    )
    unique: List[Path] = []
    seen = set()
    for p in candidates:
        k = str(p)
        if k not in seen:
            unique.append(p)
            seen.add(k)
    return unique


def find_imported_episode_dir() -> Path | None:
    for d in _candidate_episode_dirs():
        if d.exists() and d.is_dir() and ((d / "episode_list_all.json").exists() or list(d.glob("episode_list_conv_*.json"))):
            return d
    return None


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _looks_like_episode(obj: Any) -> bool:
    if not isinstance(obj, dict):
        return False
    keys = set(obj.keys())
    return bool(keys & {"episode_id", "id", "subject", "title", "summary", "episode", "episode_description", "content"}) and not (
        "episodes" in obj and isinstance(obj.get("episodes"), list)
    )


def _collect_episode_dicts(obj: Any) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if isinstance(obj, list):
        for item in obj:
            out.extend(_collect_episode_dicts(item))
    elif isinstance(obj, dict):
        if _looks_like_episode(obj):
            out.append(obj)
        else:
            for key in ["episodes", "episode_list", "data", "items", "results", "memories"]:
                if key in obj:
                    out.extend(_collect_episode_dicts(obj[key]))
    return out


def _normalize_imported_episode(item: Dict[str, Any], index: int) -> Dict[str, Any]:
    title = str(item.get("title") or item.get("subject") or item.get("topic") or f"Imported episode {index + 1}")
    summary = str(item.get("summary") or item.get("episode_summary") or item.get("content") or item.get("episode") or title)
    content = str(item.get("content") or item.get("episode") or item.get("episode_description") or item.get("text") or summary)
    source_ids = (
        _as_list(item.get("source_row_ids"))
        or _as_list(item.get("memory_ids"))
        or _as_list(item.get("message_ids"))
        or _as_list(item.get("row_ids"))
        or [str(item.get("id") or item.get("episode_id") or f"imported_{index + 1:06d}")]
    )
    timestamp = item.get("timestamp") or item.get("time") or item.get("start_time") or index + 1
    try:
        timestamp = float(timestamp)
    except Exception:
        timestamp = float(index + 1)
    return {
        "episode_id": str(item.get("episode_id") or item.get("id") or f"episode_{index + 1:05d}"),
        "title": title,
        "subject": str(item.get("subject") or title),
        "summary": summary,
        "content": content,
        "episode_description": content,
        "source_row_ids": source_ids,
        "timestamp": timestamp,
        "keywords": _as_list(item.get("keywords") or item.get("tags")),
        "participants": _as_list(item.get("participants") or item.get("entities")),
        "imported_stage1": True,
    }


def _load_imported_episodes(directory: Path, max_episodes: int = 0) -> List[Dict[str, Any]]:
    raw: List[Dict[str, Any]] = []
    all_file = directory / "episode_list_all.json"
    if all_file.exists():
        raw.extend(_collect_episode_dicts(_read_json(all_file)))
    if not raw:
        for path in sorted(directory.glob("episode_list_conv_*.json")):
            raw.extend(_collect_episode_dicts(_read_json(path)))
    episodes: List[Dict[str, Any]] = []
    seen = set()
    for i, item in enumerate(raw):
        ep = _normalize_imported_episode(item, i)
        key = (ep.get("title"), ep.get("summary"), tuple(ep.get("source_row_ids", [])))
        if key in seen:
            continue
        seen.add(key)
        ep["episode_id"] = f"episode_{len(episodes) + 1:05d}"
        episodes.append(ep)
        if max_episodes and len(episodes) >= max_episodes:
            break
    return episodes


def _resolve_import_limit(num_input_rows: int) -> int:
    raw = os.getenv("HYPERMEM_MAX_IMPORTED_EPISODES")
    if raw:
        try:
            value = int(raw)
            return max(0, value)
        except Exception:
            pass
    # Smoke-test default: respect --max-memory indirectly, because the public
    # API receives already-sliced rows from build_behavioral_hybrid_memory.py.
    return max(0, num_input_rows)


def _build_from_imported_episodes(rows: Sequence[Any], episodes: List[Dict[str, Any]], *, show_progress: bool = True) -> Dict[str, Any]:
    normalized = normalize_input_rows(rows)
    client = HyperMemStyleHierarchyClient()
    topics: List[Dict[str, Any]] = []
    topic_map: Dict[str, Dict[str, Any]] = {}

    for i, episode in enumerate(episodes):
        if show_progress:
            print(f"[hierarchy] imported-stage1 topic episode={i + 1}/{len(episodes)} active_topics={len(topics)}", flush=True)
        if not topics:
            topic_id = f"topic_{len(topics) + 1:05d}"
            topic = base._new_topic_from_response(client.create_topic([episode], len(topics)), topic_id, episode)
            topics.append(topic)
            topic_map[topic_id] = topic
            continue
        matched_ids = client.match_topics(episode, topics)
        if not matched_ids:
            topic_id = f"topic_{len(topics) + 1:05d}"
            topic = base._new_topic_from_response(client.create_topic([episode], len(topics)), topic_id, episode)
            topics.append(topic)
            topic_map[topic_id] = topic
        else:
            for tid in matched_ids:
                topic = topic_map[tid]
                base._update_topic_in_place(topic, client.update_topic(topic, episode), episode)

    episode_hyperedges: Dict[str, Dict[str, Any]] = {}
    for topic in topics:
        topic_episodes = topic.get("episodes", [])
        relation, weights, coherence = base._parse_episode_roles(client.assign_episode_roles(topic, topic_episodes), topic_episodes)
        edge_id = f"episode_hyperedge_{topic['topic_id']}"
        episode_hyperedges[edge_id] = {
            "id": edge_id,
            "relation": relation,
            "weights": weights,
            "topic_node_id": topic["topic_id"],
            "coherence_score": coherence,
        }

    all_facts: List[Dict[str, Any]] = []
    fact_hyperedges: Dict[str, Dict[str, Any]] = {}
    episode_map = {e["episode_id"]: e for e in episodes}
    seen_fact_ids: set[str] = set()
    for ti, topic in enumerate(topics):
        if show_progress:
            print(f"[hierarchy] imported-stage1 fact topic={ti + 1}/{len(topics)} episodes={len(topic.get('episodes', []))}", flush=True)
        raw_facts = client.extract_facts(topic, topic.get("episodes", []))
        topic_facts: List[Dict[str, Any]] = []
        for item in raw_facts:
            fact = base._normalize_fact(item, topic, episode_map, len(all_facts) + 1)
            if fact is None:
                continue
            if fact["fact_id"] in seen_fact_ids:
                fact["fact_id"] = f"fact_{len(all_facts) + 1:06d}"
            seen_fact_ids.add(fact["fact_id"])
            topic_facts.append(fact)
            all_facts.append(fact)
        if not topic_facts:
            for episode in topic.get("episodes", []):
                fact = {
                    "fact_id": f"fact_{len(all_facts) + 1:06d}",
                    "content": episode.get("summary") or episode.get("content") or episode.get("title"),
                    "episode_ids": [episode["episode_id"]],
                    "topic_id": topic["topic_id"],
                    "confidence": 0.7,
                    "temporal": None,
                    "spatial": None,
                    "keywords": episode.get("keywords", []),
                    "query_patterns": [],
                    "source_row_ids": episode.get("source_row_ids", []),
                }
                topic_facts.append(fact)
                all_facts.append(fact)
        fact_roles, fact_weights, extraction_confidence = base._parse_fact_roles(client.assign_fact_roles(topic, topic_facts), topic_facts)
        for fact in topic_facts:
            fact["role"] = fact_roles.get(fact["fact_id"], "detail")
            fact["weight"] = fact_weights.get(fact["fact_id"], 0.5)
        topic["facts"] = topic_facts
        for episode_id in topic.get("episode_ids", []):
            related = [f for f in topic_facts if episode_id in f.get("episode_ids", [])]
            if not related:
                continue
            edge_id = f"fact_hyperedge_{episode_id}"
            fact_hyperedges[edge_id] = {
                "id": edge_id,
                "relation": {f["fact_id"]: fact_roles.get(f["fact_id"], "detail") for f in related},
                "weights": {f["fact_id"]: fact_weights.get(f["fact_id"], 0.5) for f in related},
                "episode_node_id": episode_id,
                "extraction_confidence": extraction_confidence,
            }

    hierarchy = base._make_hierarchy(normalized, episodes, topics, all_facts, episode_hyperedges, fact_hyperedges)
    hierarchy["builder"] = "imported_official_stage1_plus_deepseek_stage2"
    hierarchy["imported_stage1_episode_count"] = len(episodes)
    return hierarchy


def extract_topic_episode_fact_hierarchy(
    rows: Sequence[Any],
    *,
    batch_size: int = 40,
    use_llm: bool = True,
    show_progress: bool = True,
) -> Dict[str, Any]:
    normalized = normalize_input_rows(rows)
    if use_llm:
        episode_dir = find_imported_episode_dir()
        if episode_dir is not None:
            max_episodes = _resolve_import_limit(len(normalized))
            episodes = _load_imported_episodes(episode_dir, max_episodes=max_episodes)
            if episodes:
                if show_progress:
                    print(
                        f"[hierarchy] using imported official Stage-1 episodes: {episode_dir} "
                        f"count={len(episodes)} limit={max_episodes or 'all'}",
                        flush=True,
                    )
                return _build_from_imported_episodes(normalized, episodes, show_progress=show_progress)
    return base.extract_topic_episode_fact_hierarchy(normalized, batch_size=batch_size, use_llm=use_llm, show_progress=show_progress)
