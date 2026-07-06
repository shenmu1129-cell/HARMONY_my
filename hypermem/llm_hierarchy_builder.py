"""LLM Topic-Episode-Fact hierarchy extraction.

This module makes the hybrid pipeline explicit:
    raw memory rows -> LLM topics -> episodes -> facts.

If the input is already a fact list, the builder still wraps it into a minimal
Topic/Episode/Fact tree so downstream behavioral-profile induction can run after
hierarchical extraction.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Sequence

from hypermem import load_runtime_env
from hypermem.llm_cache import JsonLLMCache
from hypermem.llm_profile_builder import OpenAI, _safe_json


def _content(row: Any) -> str:
    if isinstance(row, str):
        return row
    if not isinstance(row, dict):
        return str(row)
    return str(row.get("content") or row.get("text") or row.get("fact") or row.get("summary") or row.get("message") or "")


def _row_id(row: Any, idx: int) -> str:
    if isinstance(row, dict):
        return str(row.get("fact_id") or row.get("id") or row.get("message_id") or f"row_{idx+1:06d}")
    return f"row_{idx+1:06d}"


def _timestamp(row: Any, idx: int) -> float:
    if isinstance(row, dict):
        raw = row.get("timestamp") or row.get("time_index") or row.get("turn") or idx + 1
        try:
            return float(raw)
        except Exception:
            return float(idx + 1)
    return float(idx + 1)


def normalize_input_rows(rows: Sequence[Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for i, row in enumerate(rows):
        text = _content(row).strip()
        if not text:
            continue
        out.append({"row_id": _row_id(row, i), "content": text, "timestamp": _timestamp(row, i), "metadata": row if isinstance(row, dict) else {}})
    return out


class LLMHierarchyClient:
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
        self.prompt_version = "llm_topic_episode_fact_v1"

    def extract_batch(self, rows: Sequence[Dict[str, Any]], batch_index: int) -> Dict[str, Any]:
        payload = {
            "prompt_version": self.prompt_version,
            "task": "topic_episode_fact_hierarchy_extraction",
            "batch_index": batch_index,
            "rows": [{"row_id": r["row_id"], "content": r["content"][:900], "timestamp": r["timestamp"]} for r in rows],
        }
        key = self.cache.make_key(payload)
        cached = self.cache.get(key)
        if cached is None:
            prompt = self._prompt(payload)
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
        return cached.get("parsed") or {}

    def _prompt(self, payload: Dict[str, Any]) -> str:
        return (
            "You are constructing a long-term memory hierarchy for conversational QA. "
            "Given memory rows, extract a Topic-Episode-Fact tree. Topics are broad themes, episodes are coherent event segments, "
            "and facts are atomic retrievable statements. Keep facts faithful to the input and preserve row_id evidence. "
            "Return strict JSON only.\n\n"
            "Schema:\n"
            "{\"topics\":[{\"topic_id\":\"topic_001\",\"title\":\"...\",\"summary\":\"...\","
            "\"episodes\":[{\"episode_id\":\"episode_001\",\"title\":\"...\",\"summary\":\"...\","
            "\"source_row_ids\":[\"row_id\"],\"facts\":[{\"fact_id\":\"fact_001\",\"content\":\"atomic fact\","
            "\"source_row_ids\":[\"row_id\"],\"keywords\":[\"keyword\"]}]}]}]}\n\n"
            f"Input JSON:\n{json.dumps(payload, ensure_ascii=False)}"
        )


def _fallback_hierarchy(rows: Sequence[Dict[str, Any]], batch_index: int) -> Dict[str, Any]:
    topic_id = f"topic_{batch_index+1:03d}"
    episode_id = f"episode_{batch_index+1:03d}"
    facts = []
    for i, row in enumerate(rows):
        facts.append({
            "fact_id": f"fact_{batch_index+1:03d}_{i+1:04d}",
            "content": row["content"],
            "source_row_ids": [row["row_id"]],
            "keywords": [],
        })
    return {
        "topics": [{
            "topic_id": topic_id,
            "title": f"Batch {batch_index+1} memory topic",
            "summary": "Fallback topic wrapping already extracted memory facts.",
            "episodes": [{
                "episode_id": episode_id,
                "title": f"Batch {batch_index+1} memory episode",
                "summary": "Fallback episode wrapping already extracted memory facts.",
                "source_row_ids": [row["row_id"] for row in rows],
                "facts": facts,
            }],
        }]
    }


def extract_topic_episode_fact_hierarchy(
    rows: Sequence[Any],
    *,
    batch_size: int = 40,
    use_llm: bool = True,
    show_progress: bool = True,
) -> Dict[str, Any]:
    normalized = normalize_input_rows(rows)
    client = LLMHierarchyClient() if use_llm else None
    topics: List[Dict[str, Any]] = []
    source_rows = {row["row_id"]: row for row in normalized}
    total_batches = (len(normalized) + batch_size - 1) // max(1, batch_size)
    for batch_index in range(total_batches):
        batch = normalized[batch_index * batch_size : (batch_index + 1) * batch_size]
        if show_progress:
            print(f"[hierarchy] batch {batch_index+1}/{total_batches} rows={len(batch)}", flush=True)
        data = client.extract_batch(batch, batch_index) if client is not None else _fallback_hierarchy(batch, batch_index)
        batch_topics = data.get("topics") if isinstance(data, dict) else None
        if not batch_topics:
            data = _fallback_hierarchy(batch, batch_index)
            batch_topics = data["topics"]
        for topic in batch_topics:
            topics.append(topic)
    return {
        "schema": "topic_episode_fact_tree_v1",
        "num_source_rows": len(normalized),
        "source_rows": source_rows,
        "topics": topics,
    }


def flatten_hierarchy_facts(hierarchy: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    idx = 0
    for topic in hierarchy.get("topics", []):
        topic_id = str(topic.get("topic_id") or f"topic_{len(rows)+1:03d}")
        for episode in topic.get("episodes", []) or []:
            episode_id = str(episode.get("episode_id") or f"episode_{len(rows)+1:03d}")
            for fact in episode.get("facts", []) or []:
                content = str(fact.get("content") or "").strip()
                if not content:
                    continue
                idx += 1
                rows.append({
                    "fact_id": str(fact.get("fact_id") or f"fact_{idx:06d}"),
                    "content": content,
                    "keywords": fact.get("keywords") or [],
                    "timestamp": float(idx),
                    "topic_id": topic_id,
                    "topic_title": topic.get("title", ""),
                    "topic_summary": topic.get("summary", ""),
                    "episode_id": episode_id,
                    "episode_title": episode.get("title", ""),
                    "episode_summary": episode.get("summary", ""),
                    "source_row_ids": fact.get("source_row_ids") or episode.get("source_row_ids") or [],
                    "metadata": {"topic": topic, "episode": episode, "fact": fact},
                })
    return rows


def save_hierarchy_outputs(hierarchy: Dict[str, Any], output_dir: str | Path) -> List[Dict[str, Any]]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    facts = flatten_hierarchy_facts(hierarchy)
    (out_dir / "topic_episode_fact_tree.json").write_text(json.dumps(hierarchy, ensure_ascii=False, indent=2), encoding="utf-8")
    with (out_dir / "hierarchical_facts.jsonl").open("w", encoding="utf-8") as f:
        for row in facts:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return facts
