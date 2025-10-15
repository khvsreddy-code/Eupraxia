"""
Lightweight self-evolution prototype for local/dev use.

This script is intentionally conservative: it does NOT perform heavy
fine-tuning or require bitsandbytes/peft/torch. Instead it generates a
small synthetic dataset of (query, good, bad) examples using the server-side
proxy endpoint `/api/v1/proxy/generate` (so provider secrets remain server-side).

Usage (local dev):
  1. Ensure the backend API is running locally on port 8000 and `PROXY_API_KEY`
     is set in the environment used to run the script (the script reads it).
  2. Run: .venv\Scripts\python.exe backend/scripts/evolution_cycle.py

The script will write JSONL files to `backend/evolution_data/` with small
batches suitable for iterative testing on low-memory machines.

Security: do NOT paste API keys into this file. Use environment variables or a
secret manager. Revoke tokens if they were exposed in chat.
"""
import os
import time
import json
import random
from pathlib import Path
from typing import List

import requests

BASE = os.getenv("BASE_URL", "http://127.0.0.1:8000")
PROXY_KEY = os.getenv("PROXY_API_KEY")

OUT_DIR = Path(__file__).resolve().parents[1] / "evolution_data"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def gen_prompt_from_task(task: str) -> str:
    return f"Generate a correct Python implementation for the following task and a short (1-2 sentence) explanation. Task: {task}\nReturn only the implementation and explanation."


def call_proxy(prompt: str, provider: str = None) -> str:
    if not PROXY_KEY:
        raise RuntimeError("PROXY_API_KEY not set in environment. Set it before running this script.")
    url = f"{BASE}/api/v1/proxy/generate"
    headers = {"Authorization": f"Bearer {PROXY_KEY}"}
    payload = {"prompt": prompt}
    if provider:
        payload["provider"] = provider
    resp = requests.post(url, json=payload, headers=headers, timeout=30)
    resp.raise_for_status()
    j = resp.json()
    return j.get("text") or j.get("raw") or ""


def make_synthetic_dataset(num_examples: int = 200, seed: int = None) -> List[dict]:
    random.seed(seed or int(time.time()))

    # Simple small task list — expand with your domain-specific tasks
    tasks = [
        "Reverse a list in Python",
        "Implement factorial recursively and iteratively",
        "Check if a string is a palindrome",
        "Merge two sorted lists",
        "Find the nth Fibonacci number (iterative)",
        "Compute the GCD of two integers",
        "Flatten a nested list of lists",
        "Count word frequencies in a string",
        "Serialize a dict to JSON and pretty-print it",
        "Read a CSV file and compute column averages",
    ]

    ds = []
    for i in range(num_examples):
        task = random.choice(tasks)
        prompt_good = gen_prompt_from_task(task)
        # Good example (low temperature via proxy if available)
        try:
            good = call_proxy(prompt_good)
        except Exception as e:
            good = f"[error generating good example: {e}]"

        # Bad example: ask for the same but with high temperature or ask for a broken implementation
        prompt_bad = prompt_good + "\nProvide a purposely flawed implementation that contains at least one bug and a short note describing the bug."
        try:
            bad = call_proxy(prompt_bad)
        except Exception as e:
            bad = f"[error generating bad example: {e}]"

        ds.append({"id": i, "task": task, "good": good, "bad": bad})
        # small delay to avoid rate spikes
        time.sleep(0.15)

    return ds


def save_dataset(ds: List[dict], prefix: str = "synth") -> Path:
    ts = int(time.time())
    out = OUT_DIR / f"{prefix}_{ts}.jsonl"
    with out.open("w", encoding="utf-8") as fh:
        for row in ds:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    return out


def main():
    n = int(os.getenv("SYNTH_EXAMPLES", "200"))
    print(f"Generating {n} synthetic examples via proxy at {BASE}")
    ds = make_synthetic_dataset(num_examples=n)
    saved = save_dataset(ds)
    print(f"Saved {len(ds)} examples to {saved}")


if __name__ == "__main__":
    main()
