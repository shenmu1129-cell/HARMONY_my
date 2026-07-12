"""
HyperMem: Hypergraph-based Memory System for Long-term Conversational QA
"""

from __future__ import annotations

import os
from pathlib import Path


def _strip_env_quotes(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def load_runtime_env(path: str | None = None, *, override: bool = False) -> None:
    """Load simple KEY=VALUE runtime config for local experiments.

    The project-local OpenAI config is preferred.  DeepSeek's legacy server
    config remains a fallback for existing experiments.  This runs when the
    package is imported, so scripts do not need to manually source either file.
    """
    project_root = Path(__file__).resolve().parents[1]
    default_path = "/home/sutongtong/wwt/code/hyperMem_my/configs/deepseek.env"
    explicit_path = path or os.getenv("OPENAI_ENV_FILE") or os.getenv("DEEPSEEK_ENV_FILE")
    env_paths = [Path(explicit_path).expanduser()] if explicit_path else [
        project_root / "configs" / "openai.env",
        Path(default_path),
    ]
    for env_path in env_paths:
        if not env_path.exists() or not env_path.is_file():
            continue
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[len("export ") :].strip()
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = _strip_env_quotes(value)
            if key and value and (override or key not in os.environ):
                os.environ[key] = value


load_runtime_env()

from hypermem.types import Episode, Fact, Topic, RawDataType
from hypermem.structure import (
    Hypergraph,
    FactNode,
    EpisodeNode,
    TopicNode,
    FactHyperedge,
    EpisodeHyperedge,
    FactRole,
    EpisodeRole,
)

__version__ = "0.1.0"
__all__ = [
    # Runtime config
    "load_runtime_env",
    # Types
    "Episode",
    "Fact",
    "Topic",
    "RawDataType",
    # Structure
    "Hypergraph",
    "FactNode",
    "EpisodeNode",
    "TopicNode",
    "FactHyperedge",
    "EpisodeHyperedge",
    "FactRole",
    "EpisodeRole",
]
