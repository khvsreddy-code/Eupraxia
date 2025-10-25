import json
from pathlib import Path
from typing import Dict

OUT = Path(__file__).resolve().parent / "feedback.jsonl"


def save_feedback(record: Dict):
    """Append a JSON record to feedback.jsonl (atomic append)."""
    with OUT.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_feedback(limit: int = 100):
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
