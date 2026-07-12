"""Env-aware wrapper for LLM profile hypergraph construction."""

from __future__ import annotations

import os
from typing import Any, Dict, Sequence

from hypermem import load_runtime_env
from hypermem.llm_profile_builder import LLMBatchProfileHypergraphBuilder
from hypermem.profile_centric_hypergraph import ProfileCentricHypergraphMemory


def _float_env(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


def _int_env(name: str, default: int) -> int:
    try:
        return int(float(os.getenv(name, str(default))))
    except ValueError:
        return default


def build_llm_profile_hypergraph_from_rows_auto(
    memory: ProfileCentricHypergraphMemory,
    rows: Sequence[Dict[str, Any]],
    **kwargs: Any,
) -> LLMBatchProfileHypergraphBuilder:
    """Build with config read from deepseek.env by the package itself.

    The config file path defaults to:
        /home/sutongtong/wwt/code/hyperMem_my/configs/deepseek.env

    Override it by setting DEEPSEEK_ENV_FILE before Python starts.
    Explicit kwargs still take precedence over env values.
    """
    load_runtime_env()
    params = dict(kwargs)
    params.setdefault("api_key_env", "DEEPSEEK_API_KEY")
    params.setdefault("base_url", os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"))
    params.setdefault("model", os.getenv("DEEPSEEK_MODEL", "deepseek-chat"))
    params.setdefault("temperature", _float_env("DEEPSEEK_TEMPERATURE", 0.2))
    params.setdefault("max_features_per_batch", _int_env("LLM_MAX_FEATURES_PER_BATCH", 12))
    params.setdefault("max_features_per_fact", _int_env("LLM_MAX_FEATURES_PER_FACT", 4))
    builder = LLMBatchProfileHypergraphBuilder(memory, **params)
    builder.build(rows)
    return builder
