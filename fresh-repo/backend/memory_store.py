import json
from pathlib import Path
from typing import List, Dict

OUT = Path(__file__).resolve().parent / "memory.jsonl"


def add_memory(record: Dict):
    with OUT.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def get_memories(limit: int = 100) -> List[Dict]:
    if not OUT.exists():
        return []
    res = []
    with OUT.open("r", encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if i >= limit:
                break
            try:
                res.append(json.loads(line))
            except Exception:
                continue
    return res
