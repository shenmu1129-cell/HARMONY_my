"""Prepare JSONL inputs for UP-HyperPool formal evaluation.

This script is intentionally conservative:
- If --demo is set, it writes a small built-in profile-memory dataset.
- If --source-dir is set, it scans JSON/JSONL files and tries to extract
  memory facts and questions using common field names.

Examples:
    python examples/prepare_profile_eval_data.py --demo --out-dir data

    python examples/prepare_profile_eval_data.py \
      --source-dir outputs/hypermem \
      --out-dir data \
      --max-memory 2000 \
      --max-questions 500

Outputs:
    <out-dir>/locomo_memory_facts.jsonl
    <out-dir>/locomo_questions.jsonl
    <out-dir>/profile_eval_data_report.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


DEMO_MEMORY = [
    "用户正在研究 LLM memory，核心对象包括 HyperMem、A-MEM、MeMo、MemEvolve 和 LoCoMo。",
    "用户目标是把 memory 方向做成 AAAI 级别的创新，而不是简单工程拼接。",
    "用户喜欢审稿人视角：先判断创新性，再判断实验是否能支撑，最后给 Codex prompt。",
    "用户不希望被空泛鼓励，更希望指出当前方案的风险、弱点和可救方向。",
    "fixed_400 是强低成本 baseline，global_fact_only_800 命中率高但 token 成本大。",
    "override_logreg 能接近 global_fact_only_800 的 hit，同时显著减少 token，因此说明 query-dependent routing 有价值。",
    "adaptive_controller_v1 已经跑通多步闭环，但规则版 verifier/action policy 引入噪声，reward 不如 override_logreg。",
    "当前新方向是用户画像引导的动态超边池，在 Topic/Episode/Fact 底座之上维护个性化快速通道。",
    "用户画像超边池存 preference、goal、habit、domain knowledge、current state、temporal evolution 等高价值超边。",
    "当 profile fast channel 证据不足时，系统应 fallback 到原始 HyperMem path 或 global fact retrieval。",
    "奖励更新用于维护 profile hyperedge utility：命中和帮助回答则升权，错误、过期或无贡献则降权。",
    "时间处理可以成为创新点，因为长期记忆需要区分过去想法、最新状态和想法演化链。",
    "用户经常讨论代码实现、服务器运行、GitHub main 分支、README 和 Codex 生成代码。",
    "用户也会讨论论文投稿、审稿人意见、AAAI 竞争力、实验是否足够支撑创新。",
]

DEMO_QUESTIONS = [
    {"question": "我现在这个 memory 方案的核心是什么？", "gold": ["用户画像", "动态超边池"], "category": "current_state"},
    {"question": "我通常希望你怎么分析论文？", "gold": ["审稿人视角", "Codex prompt"], "category": "preference"},
    {"question": "为什么 adaptive_controller_v1 还不行？", "gold": ["规则版", "引入噪声"], "category": "experiment"},
    {"question": "如果快速通道证据不足怎么办？", "gold": ["fallback", "原始 HyperMem"], "category": "method"},
    {"question": "强化学习奖励在超边池里干什么？", "gold": ["utility", "升权", "降权"], "category": "rl"},
    {"question": "我经常让你帮我做哪些工程操作？", "gold": ["服务器运行", "GitHub", "Codex"], "category": "auto_discovery"},
]

MEMORY_FIELDS = ["content", "text", "fact", "summary", "memory", "value", "utterance"]
QUESTION_FIELDS = ["question", "query", "q"]
ANSWER_FIELDS = ["gold", "answer", "answers", "evidence", "gold_evidence", "target"]


def load_json_like(path: Path) -> Iterable[Any]:
    try:
        text = path.read_text(encoding="utf-8").strip()
    except UnicodeDecodeError:
        return []
    if not text:
        return []
    try:
        if text.startswith("[") or text.startswith("{"):
            obj = json.loads(text)
            if isinstance(obj, list):
                return obj
            return [obj]
    except json.JSONDecodeError:
        pass

    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def flatten_records(obj: Any) -> Iterable[Dict[str, Any]]:
    if isinstance(obj, dict):
        yield obj
        for value in obj.values():
            if isinstance(value, (dict, list)):
                yield from flatten_records(value)
    elif isinstance(obj, list):
        for item in obj:
            yield from flatten_records(item)


def first_present(row: Dict[str, Any], fields: List[str]) -> Any:
    for field in fields:
        if field in row and row[field] not in (None, ""):
            return row[field]
    return None


def normalize_gold(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x) for x in value if str(x).strip()]
    if isinstance(value, dict):
        return [str(x) for x in value.values() if str(x).strip()]
    return [str(value)] if str(value).strip() else []


def scan_source_dir(source_dir: Path, max_memory: int, max_questions: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    memory_rows: List[Dict[str, Any]] = []
    question_rows: List[Dict[str, Any]] = []
    seen_memory = set()
    seen_questions = set()
    scanned_files = []

    candidates = sorted(
        p for p in source_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in {".json", ".jsonl"}
    )

    for path in candidates:
        scanned_files.append(path.as_posix())
        for top in load_json_like(path):
            for row in flatten_records(top):
                q = first_present(row, QUESTION_FIELDS)
                if q is not None and len(question_rows) < max_questions:
                    gold = normalize_gold(first_present(row, ANSWER_FIELDS))
                    key = str(q).strip()
                    if key and key not in seen_questions:
                        seen_questions.add(key)
                        question_rows.append(
                            {
                                "qid": row.get("qid") or row.get("id") or f"q_{len(question_rows)+1:05d}",
                                "question": key,
                                "gold": gold,
                                "category": row.get("category") or row.get("type") or path.stem,
                            }
                        )

                content = first_present(row, MEMORY_FIELDS)
                if content is not None and len(memory_rows) < max_memory:
                    content = str(content).strip()
                    if content and len(content) >= 5 and content not in seen_memory:
                        # Avoid adding questions as memory facts when both fields exist.
                        if q is not None and content == str(q).strip():
                            continue
                        seen_memory.add(content)
                        memory_rows.append(
                            {
                                "fact_id": row.get("fact_id") or row.get("id") or f"fact_{len(memory_rows)+1:05d}",
                                "content": content,
                                "keywords": row.get("keywords") or [],
                                "timestamp": row.get("timestamp") or row.get("time_index") or len(memory_rows) + 1,
                                "topic_id": row.get("topic_id") or row.get("conversation_id") or row.get("session_id") or path.stem,
                                "episode_ids": row.get("episode_ids") or ([row["episode_id"]] if row.get("episode_id") else []),
                            }
                        )

        if len(memory_rows) >= max_memory and len(question_rows) >= max_questions:
            break

    report = {
        "source_dir": source_dir.as_posix(),
        "scanned_files": scanned_files,
        "num_scanned_files": len(scanned_files),
        "num_memory_rows": len(memory_rows),
        "num_question_rows": len(question_rows),
    }
    return memory_rows, question_rows, report


def write_jsonl(rows: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", action="store_true", help="Write built-in demo data.")
    parser.add_argument("--source-dir", type=str, default="", help="Scan a directory for JSON/JSONL records.")
    parser.add_argument("--out-dir", type=str, default="data")
    parser.add_argument("--max-memory", type=int, default=2000)
    parser.add_argument("--max-questions", type=int, default=500)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    if args.demo:
        memory_rows = [
            {"fact_id": f"fact_{i+1:05d}", "content": text, "timestamp": i + 1, "topic_id": "demo"}
            for i, text in enumerate(DEMO_MEMORY)
        ]
        question_rows = [
            {"qid": f"q_{i+1:05d}", **row}
            for i, row in enumerate(DEMO_QUESTIONS)
        ]
        report = {"mode": "demo", "num_memory_rows": len(memory_rows), "num_question_rows": len(question_rows)}
    elif args.source_dir:
        memory_rows, question_rows, report = scan_source_dir(Path(args.source_dir), args.max_memory, args.max_questions)
        report["mode"] = "scan"
    else:
        raise SystemExit("ERROR: pass either --demo or --source-dir")

    memory_path = out_dir / "locomo_memory_facts.jsonl"
    questions_path = out_dir / "locomo_questions.jsonl"
    report_path = out_dir / "profile_eval_data_report.json"
    write_jsonl(memory_rows, memory_path)
    write_jsonl(question_rows, questions_path)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("wrote", memory_path, "rows=", len(memory_rows))
    print("wrote", questions_path, "rows=", len(question_rows))
    print("wrote", report_path)
    if not memory_rows or not question_rows:
        print("WARNING: generated empty memory or question file; inspect profile_eval_data_report.json")


if __name__ == "__main__":
    main()
