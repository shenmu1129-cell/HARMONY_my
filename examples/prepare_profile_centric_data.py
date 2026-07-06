"""Prepare JSONL inputs for profile-centric hypergraph memory.

Outputs:
    <out-dir>/memory_facts.jsonl
    <out-dir>/questions.jsonl
    <out-dir>/data_report.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None  # type: ignore[assignment]


DEMO_MEMORY = [
    "用户正在研究 LLM memory，重点关注 HyperMem、A-MEM、MeMo、MemEvolve 和 LoCoMo。",
    "用户希望把 memory 方向做成 AAAI 级别创新，而不是简单工程拼接。",
    "用户喜欢审稿人视角：先判断创新性，再判断实验是否能支撑，最后给 Codex prompt。",
    "用户不喜欢空泛鼓励，更希望指出当前方案风险、弱点和可救方向。",
    "用户经常在服务器上运行实验，使用 conda 环境、bash 脚本和 GitHub main 分支。",
    "用户当前主线是用用户画像超边替代 HyperMem 的 Topic-Episode-Fact 主检索路径。",
    "用户认为 embedding 仍然需要，但它负责向量召回和相似度匹配，profile utility 负责个性化价值排序。",
    "用户倾向使用轻量 bandit-style reward update，而不是训练 PPO 或大模型。",
]

DEMO_QUESTIONS = [
    {"question": "我现在 memory 方案的核心主线是什么？", "gold": ["用户画像超边", "Topic-Episode-Fact"], "category": "method"},
    {"question": "我希望你怎么评价论文创新？", "gold": ["审稿人视角", "创新性", "实验"], "category": "preference"},
    {"question": "embedding 在我的方法里还有用吗？", "gold": ["向量", "profile utility"], "category": "method"},
    {"question": "我经常让你帮我做哪些工程操作？", "gold": ["服务器", "conda", "GitHub"], "category": "habit"},
    {"question": "强化学习版本最好先做哪种？", "gold": ["bandit", "reward update"], "category": "rl"},
]

MEMORY_FIELDS = ["content", "text", "fact", "summary", "memory", "value", "utterance"]
QUESTION_FIELDS = ["question", "query", "q"]
ANSWER_FIELDS = ["gold", "answer", "answers", "evidence", "gold_evidence", "target"]


def maybe_progress(iterable, *, total: int | None = None, desc: str = "", enabled: bool = False):
    if enabled and tqdm is not None:
        return tqdm(iterable, total=total, desc=desc, dynamic_ncols=True)
    return iterable


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


def scan_source_dir(source_dir: Path, max_memory: int, max_questions: int, show_progress: bool = False) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    started = time.time()
    print(f"[stage1] indexing json/jsonl files under {source_dir}", flush=True)
    candidates = sorted(path for path in source_dir.rglob("*") if path.is_file() and path.suffix.lower() in {".json", ".jsonl"})
    print(f"[stage1] found candidate_files={len(candidates)}", flush=True)

    memory_rows: List[Dict[str, Any]] = []
    question_rows: List[Dict[str, Any]] = []
    seen_memory = set()
    seen_questions = set()
    scanned_files = []

    iterator = maybe_progress(candidates, total=len(candidates), desc="[stage1] scan files", enabled=show_progress)
    for idx, path in enumerate(iterator, start=1):
        scanned_files.append(path.as_posix())
        for top in load_json_like(path):
            for row in flatten_records(top):
                q = first_present(row, QUESTION_FIELDS)
                if q is not None and len(question_rows) < max_questions:
                    gold = normalize_gold(first_present(row, ANSWER_FIELDS))
                    key = str(q).strip()
                    if key and key not in seen_questions:
                        seen_questions.add(key)
                        question_rows.append({
                            "qid": row.get("qid") or row.get("id") or f"q_{len(question_rows)+1:05d}",
                            "question": key,
                            "gold": gold,
                            "category": row.get("category") or row.get("type") or path.stem,
                        })

                content = first_present(row, MEMORY_FIELDS)
                if content is not None and len(memory_rows) < max_memory:
                    content = str(content).strip()
                    if content and len(content) >= 5 and content not in seen_memory:
                        if q is not None and content == str(q).strip():
                            continue
                        seen_memory.add(content)
                        memory_rows.append({
                            "fact_id": row.get("fact_id") or row.get("id") or f"fact_{len(memory_rows)+1:05d}",
                            "content": content,
                            "keywords": row.get("keywords") or [],
                            "timestamp": row.get("timestamp") or row.get("time_index") or len(memory_rows) + 1,
                            "source": path.stem,
                        })
        if tqdm is not None and hasattr(iterator, "set_postfix"):
            iterator.set_postfix(memory=len(memory_rows), questions=len(question_rows))
        elif show_progress and idx % 100 == 0:
            print(f"[stage1] scanned_files={idx}/{len(candidates)} memory={len(memory_rows)} questions={len(question_rows)}", flush=True)
        if len(memory_rows) >= max_memory and len(question_rows) >= max_questions:
            break

    report = {
        "source_dir": source_dir.as_posix(),
        "num_scanned_files": len(scanned_files),
        "num_candidate_files": len(candidates),
        "num_memory_rows": len(memory_rows),
        "num_question_rows": len(question_rows),
        "elapsed_sec": round(time.time() - started, 3),
        "scanned_files": scanned_files,
    }
    return memory_rows, question_rows, report


def write_jsonl(rows: List[Dict[str, Any]], path: Path, show_progress: bool = False, desc: str = "write jsonl") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    iterator = maybe_progress(rows, total=len(rows), desc=desc, enabled=show_progress)
    with path.open("w", encoding="utf-8") as f:
        for row in iterator:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", action="store_true", help="Write built-in demo data.")
    parser.add_argument("--source-dir", type=str, default="", help="Scan a directory for JSON/JSONL records.")
    parser.add_argument("--out-dir", type=str, default="data")
    parser.add_argument("--max-memory", type=int, default=2000)
    parser.add_argument("--max-questions", type=int, default=500)
    parser.add_argument("--show-progress", action="store_true", help="Show per-file/per-row progress bars.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    if args.demo:
        memory_rows = [{"fact_id": f"fact_{i+1:05d}", "content": text, "timestamp": i + 1, "source": "demo"} for i, text in enumerate(DEMO_MEMORY)]
        question_rows = [{"qid": f"q_{i+1:05d}", **row} for i, row in enumerate(DEMO_QUESTIONS)]
        report = {"mode": "demo", "num_memory_rows": len(memory_rows), "num_question_rows": len(question_rows)}
    elif args.source_dir:
        memory_rows, question_rows, report = scan_source_dir(Path(args.source_dir), args.max_memory, args.max_questions, show_progress=args.show_progress)
        report["mode"] = "scan"
    else:
        raise SystemExit("ERROR: pass either --demo or --source-dir")

    memory_path = out_dir / "memory_facts.jsonl"
    questions_path = out_dir / "questions.jsonl"
    report_path = out_dir / "data_report.json"
    write_jsonl(memory_rows, memory_path, show_progress=args.show_progress, desc="[stage1] write memory")
    write_jsonl(question_rows, questions_path, show_progress=args.show_progress, desc="[stage1] write questions")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("wrote", memory_path, "rows=", len(memory_rows))
    print("wrote", questions_path, "rows=", len(question_rows))
    print("wrote", report_path)
    if not memory_rows or not question_rows:
        print("WARNING: generated empty memory or question file; inspect data_report.json")


if __name__ == "__main__":
    main()
