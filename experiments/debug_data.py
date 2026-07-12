"""Debug LoCoMo data loading for PCH-Mem."""
import json

with open("data/locomo10.json") as f:
    data = json.load(f)

conv = data[0]["conversation"]
qa = data[0]["qa"]

# Check dia_ids in sessions
for k in sorted(conv.keys()):
    if k.startswith("session_") and not k.endswith("_date_time"):
        turns = conv[k]
        if isinstance(turns, list):
            print(f"Session: {k}, turns: {len(turns)}")
            for t in turns[:2]:
                did = t.get("dia_id", "N/A")
                print(f"  dia_id={did}, speaker={t.get('speaker','?')}")
            break

# Check QA evidence format
print()
for q in qa[:5]:
    ev = q.get("evidence", [])
    print(f"QA: evidence={ev}, Q={q.get('question','')[:80]}")
