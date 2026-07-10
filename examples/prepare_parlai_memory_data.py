"""Prepare Persona-Chat / ConvAI2 / MSC ParlAI data for memory evaluation.

The script converts ParlAI-style dialogue files into the repository's common
JSONL format:

    memory_facts.jsonl
    questions.jsonl
    data_report.json

It is intentionally local and dependency-free. It supports the common ParlAI
numbered text format used by Persona-Chat and ConvAI2, and also scans JSON/JSONL
records used by some MSC builds.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None  # type: ignore[assignment]

_LINE_RE = re.compile(r"^(?P<num>\d+)\s+(?P<body>.*)$")
_PERSONA_PREFIXES = (
    "your persona:",
    "partner's persona:",
    "partner persona:",
    "persona:",
    "personasummary:",
)
_TEXT_EXTS = {".txt", ".train", ".valid", ".test"}
_JSON_EXTS = {".json", ".jsonl"}


def progress(iterable, *, total: int | None = None, desc: str = "", enabled: bool = False):
    if enabled and tqdm is not None:
        return tqdm(iterable, total=total, desc=desc, dynamic_ncols=True)
    return iterable


def clean_text(text: str) -> str:
    text = str(text or "").replace("\\n", " ").replace("\t", " ")
    return " ".join(text.split()).strip()


def strip_line_number(line: str) -> Tuple[int | None, str]:
    m = _LINE_RE.match(line.strip())
    if not m:
        return None, line.strip()
    return int(m.group("num")), m.group("body").strip()


def is_persona(text: str) -> bool:
    lower = text.lower().strip()
    return any(lower.startswith(p) for p in _PERSONA_PREFIXES)


def strip_persona(text: str) -> str:
    out = clean_text(text)
    lower = out.lower()
    for prefix in _PERSONA_PREFIXES:
        if lower.startswith(prefix):
            return clean_text(out[len(prefix):])
    return out


def split_parlai_turn(body: str) -> Tuple[str, str, List[str]]:
    parts = body.split("\t")
    context = clean_text(parts[0]) if parts else ""
    label = clean_text(parts[1]) if len(parts) > 1 else ""
    candidates: List[str] = []
    if len(parts) > 3:
        candidates = [clean_text(x) for x in parts[3].split("|") if clean_text(x)]
    return context, label, candidates


def add_memory(
    rows: List[Dict[str, Any]],
    seen: set[str],
    *,
    dataset: str,
    split: str,
    dialogue_id: str,
    content: str,
    source_type: str,
    timestamp: int,
    extra: Dict[str, Any] | None = None,
    max_memory: int,
) -> None:
    text = clean_text(content)
    if not text or len(text) < 3 or len(rows) >= max_memory:
        return
    key = f"{dataset}|{split}|{dialogue_id}|{source_type}|{text}"
    if key in seen:
        return
    seen.add(key)
    row = {
        "fact_id": f"{dataset}_{len(rows)+1:08d}",
        "content": text,
        "keywords": [],
        "timestamp": timestamp,
        "dataset": dataset,
        "split": split,
        "dialogue_id": dialogue_id,
        "source_type": source_type,
    }
    if extra:
        row.update(extra)
    rows.append(row)


def add_question(
    rows: List[Dict[str, Any]],
    seen: set[str],
    *,
    dataset: str,
    split: str,
    dialogue_id: str,
    question: str,
    gold: Sequence[str],
    category: str,
    max_questions: int,
) -> None:
    q = clean_text(question)
    golds = [clean_text(x) for x in gold if clean_text(x)]
    if not q or not golds or len(rows) >= max_questions:
        return
    key = f"{dataset}|{split}|{dialogue_id}|{category}|{q}|{'|'.join(golds[:2])}"
    if key in seen:
        return
    seen.add(key)
    rows.append({
        "qid": f"{dataset}_q_{len(rows)+1:08d}",
        "question": q,
        "gold": golds,
        "category": category,
        "dataset": dataset,
        "split": split,
        "dialogue_id": dialogue_id,
    })


def infer_split(path: Path) -> str:
    name = path.as_posix().lower()
    if "valid" in name or "dev" in name:
        return "valid"
    if "test" in name:
        return "test"
    if "train" in name:
        return "train"
    return "unknown"


def dataset_name(path: Path) -> str:
    return path.name.replace("-", "_").lower()


def parse_parlai_text_file(
    path: Path,
    *,
    dataset: str,
    max_memory: int,
    max_questions: int,
    memory_rows: List[Dict[str, Any]],
    question_rows: List[Dict[str, Any]],
    seen_memory: set[str],
    seen_questions: set[str],
) -> Dict[str, int]:
    split = infer_split(path)
    dialogue_idx = 0
    turn_idx = 0
    personas: List[str] = []
    history: List[str] = []
    stats = {"dialogues": 0, "turns": 0, "personas": 0, "questions": 0, "memory": 0}
    before_mem = len(memory_rows)
    before_q = len(question_rows)

    try:
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except OSError:
        return stats

    for raw in lines:
        raw = raw.strip()
        if not raw:
            continue
        num, body = strip_line_number(raw)
        if num == 1 or (num is None and not history and not personas):
            if history or personas:
                stats["dialogues"] += 1
            dialogue_idx += 1
            turn_idx = 0
            personas = []
            history = []

        context, label, _ = split_parlai_turn(body)
        if not context and not label:
            continue
        dialogue_id = f"{path.stem}_{dialogue_idx:06d}"
        timestamp = len(memory_rows) + 1

        if is_persona(context):
            persona = strip_persona(context)
            personas.append(persona)
            stats["personas"] += 1
            add_memory(memory_rows, seen_memory, dataset=dataset, split=split, dialogue_id=dialogue_id,
                       content=f"Persona statement: {persona}", source_type="persona", timestamp=timestamp,
                       extra={"file": path.as_posix()}, max_memory=max_memory)
            add_question(question_rows, seen_questions, dataset=dataset, split=split, dialogue_id=dialogue_id,
                         question="What persona/profile information is stated in this dialogue?", gold=[persona],
                         category="persona_profile", max_questions=max_questions)
            continue

        if label:
            turn_idx += 1
            stats["turns"] += 1
            context_text = context
            if personas:
                context_text = " ".join(["Persona: " + " | ".join(personas[-4:]), "Dialogue context: " + context])
            add_memory(memory_rows, seen_memory, dataset=dataset, split=split, dialogue_id=dialogue_id,
                       content=f"Dialogue context: {context}", source_type="dialogue_context", timestamp=timestamp,
                       extra={"file": path.as_posix(), "turn_idx": turn_idx}, max_memory=max_memory)
            add_memory(memory_rows, seen_memory, dataset=dataset, split=split, dialogue_id=dialogue_id,
                       content=f"Expected response: {label}", source_type="response", timestamp=timestamp + 1,
                       extra={"file": path.as_posix(), "turn_idx": turn_idx}, max_memory=max_memory)
            add_question(question_rows, seen_questions, dataset=dataset, split=split, dialogue_id=dialogue_id,
                         question=f"Given the persona and dialogue context, what is the next response? {context_text}",
                         gold=[label], category="next_response_retrieval", max_questions=max_questions)
            if context:
                history.append(context)
            history.append(label)
        elif context:
            add_memory(memory_rows, seen_memory, dataset=dataset, split=split, dialogue_id=dialogue_id,
                       content=context, source_type="utterance", timestamp=timestamp,
                       extra={"file": path.as_posix()}, max_memory=max_memory)
        if len(memory_rows) >= max_memory and len(question_rows) >= max_questions:
            break

    if history or personas:
        stats["dialogues"] += 1
    stats["memory"] = len(memory_rows) - before_mem
    stats["questions"] = len(question_rows) - before_q
    return stats


def load_json_like(path: Path) -> Iterable[Any]:
    try:
        text = path.read_text(encoding="utf-8", errors="ignore").strip()
    except OSError:
        return []
    if not text:
        return []
    if path.suffix.lower() == ".jsonl":
        rows = []
        for line in text.splitlines():
            try:
                rows.append(json.loads(line.strip()))
            except Exception:
                pass
        return rows
    try:
        obj = json.loads(text)
    except Exception:
        return []
    return obj if isinstance(obj, list) else [obj]


def flatten_json(obj: Any) -> Iterable[Dict[str, Any]]:
    if isinstance(obj, dict):
        yield obj
        for value in obj.values():
            if isinstance(value, (dict, list)):
                yield from flatten_json(value)
    elif isinstance(obj, list):
        for item in obj:
            yield from flatten_json(item)


def parse_json_file(path: Path, *, dataset: str, max_memory: int, max_questions: int,
                    memory_rows: List[Dict[str, Any]], question_rows: List[Dict[str, Any]],
                    seen_memory: set[str], seen_questions: set[str]) -> Dict[str, int]:
    split = infer_split(path)
    before_mem = len(memory_rows)
    before_q = len(question_rows)
    stats = {"memory": 0, "questions": 0, "records": 0}
    text_fields = ["text", "utterance", "response", "label", "answer", "persona", "summary", "dialog", "dialogue"]
    q_fields = ["question", "query", "prompt", "context"]
    a_fields = ["answer", "response", "label", "target"]
    for top in load_json_like(path):
        for row in flatten_json(top):
            stats["records"] += 1
            dialogue_id = str(row.get("dialogue_id") or row.get("episode_id") or row.get("id") or path.stem)
            for field in text_fields:
                val = row.get(field)
                if isinstance(val, str) and clean_text(val):
                    add_memory(memory_rows, seen_memory, dataset=dataset, split=split, dialogue_id=dialogue_id,
                               content=val, source_type=field, timestamp=len(memory_rows) + 1,
                               extra={"file": path.as_posix()}, max_memory=max_memory)
            q = next((row.get(f) for f in q_fields if isinstance(row.get(f), str) and clean_text(row.get(f))), "")
            a = next((row.get(f) for f in a_fields if isinstance(row.get(f), str) and clean_text(row.get(f))), "")
            if q and a:
                add_question(question_rows, seen_questions, dataset=dataset, split=split, dialogue_id=dialogue_id,
                             question=clean_text(q), gold=[clean_text(a)], category="json_qa", max_questions=max_questions)
            if len(memory_rows) >= max_memory and len(question_rows) >= max_questions:
                break
    stats["memory"] = len(memory_rows) - before_mem
    stats["questions"] = len(question_rows) - before_q
    return stats


def scan_dataset(root: Path, *, max_memory: int, max_questions: int, show_progress: bool) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    started = time.time()
    memory_rows: List[Dict[str, Any]] = []
    question_rows: List[Dict[str, Any]] = []
    seen_memory: set[str] = set()
    seen_questions: set[str] = set()
    files = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in (_TEXT_EXTS | _JSON_EXTS))
    per_file = []
    dataset = dataset_name(root)
    for path in progress(files, total=len(files), desc=f"[prepare] {root.name}", enabled=show_progress):
        if path.suffix.lower() in _JSON_EXTS:
            stats = parse_json_file(path, dataset=dataset, max_memory=max_memory, max_questions=max_questions,
                                    memory_rows=memory_rows, question_rows=question_rows,
                                    seen_memory=seen_memory, seen_questions=seen_questions)
        else:
            stats = parse_parlai_text_file(path, dataset=dataset, max_memory=max_memory, max_questions=max_questions,
                                           memory_rows=memory_rows, question_rows=question_rows,
                                           seen_memory=seen_memory, seen_questions=seen_questions)
        per_file.append({"file": path.as_posix(), **stats})
        if len(memory_rows) >= max_memory and len(question_rows) >= max_questions:
            break
    report = {"dataset_root": root.as_posix(), "dataset": dataset, "candidate_files": len(files),
              "num_memory_rows": len(memory_rows), "num_question_rows": len(question_rows),
              "elapsed_sec": round(time.time() - started, 3), "per_file": per_file}
    return memory_rows, question_rows, report


def write_jsonl(rows: Sequence[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--max-memory", type=int, default=2000)
    parser.add_argument("--max-questions", type=int, default=1000)
    parser.add_argument("--show-progress", action="store_true")
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    memory, questions, report = scan_dataset(Path(args.dataset_root), max_memory=args.max_memory,
                                             max_questions=args.max_questions, show_progress=args.show_progress)
    write_jsonl(memory, out_dir / "memory_facts.jsonl")
    write_jsonl(questions, out_dir / "questions.jsonl")
    (out_dir / "data_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "per_file"}, ensure_ascii=False, indent=2))
    print("wrote", out_dir / "memory_facts.jsonl", "rows=", len(memory))
    print("wrote", out_dir / "questions.jsonl", "rows=", len(questions))
    if not memory or not questions:
        raise SystemExit("ERROR: no memory/questions generated; inspect data_report.json")


if __name__ == "__main__":
    main()
