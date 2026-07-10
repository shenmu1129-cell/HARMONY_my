"""Auto-load local runtime configuration when Python starts.

This file is imported automatically by Python when the repository root is on
sys.path. It loads a simple KEY=VALUE env file so experiments can read the
DeepSeek configuration without manually sourcing it before every run.
"""

from __future__ import annotations

import os
from pathlib import Path

DEFAULT_ENV_FILE = "/home/sutongtong/wwt/code/hyperMem_my/configs/deepseek.env"


def _strip_quotes(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def _load_env_file(path: str) -> None:
    env_path = Path(path).expanduser()
    if not env_path.exists() or not env_path.is_file():
        return
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
        value = _strip_quotes(value)
        if key and key not in os.environ:
            os.environ[key] = value


_load_env_file(os.getenv("DEEPSEEK_ENV_FILE", DEFAULT_ENV_FILE))
