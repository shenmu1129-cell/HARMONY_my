"""Minimal OpenAI API smoke test.

Usage:
    OPENAI_API_KEY=sk-... conda run -n wwt310 python examples/openai_api_smoke_test.py
"""

from __future__ import annotations

import argparse
import os
import sys

from dotenv import load_dotenv
from openai import OpenAI, OpenAIError


DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_PROMPT = "Use one Chinese sentence to confirm that the OpenAI API test succeeded."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a minimal OpenAI API smoke test.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="OpenAI model to call.")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="Prompt to send.")
    parser.add_argument("--max-output-tokens", type=int, default=80)
    return parser.parse_args()


def build_client() -> OpenAI:
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        print("Set OPENAI_API_KEY before running this script.", file=sys.stderr)
        raise SystemExit(1)
    return OpenAI(api_key=api_key)


def main() -> None:
    args = parse_args()
    try:
        response = build_client().responses.create(
            model=args.model,
            input=args.prompt,
            max_output_tokens=args.max_output_tokens,
            store=False,
        )
    except OpenAIError as exc:
        print(f"OpenAI API request failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    print(f"Model: {args.model}")
    print(response.output_text)


if __name__ == "__main__":
    main()
