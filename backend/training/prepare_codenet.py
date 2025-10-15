"""Prepare Project_CodeNet dataset for instruction tuning / code modeling.

This script clones (or updates) the IBM Project_CodeNet repository, extracts code files and metadata,
and converts them into a JSONL of {"prompt":..., "completion":...} suitable for instruction tuning.

WARNING: Project_CodeNet is large. Run this on a machine with sufficient disk space.
"""
import os
import json
import argparse
import subprocess
from pathlib import Path


def git_clone_or_pull(repo_url: str, target_dir: Path):
    if target_dir.exists():
        print(f"Updating {target_dir} via git pull")
        subprocess.check_call(["git", "-C", str(target_dir), "pull"])
    else:
        print(f"Cloning {repo_url} into {target_dir}")
        subprocess.check_call(["git", "clone", repo_url, str(target_dir)])


def gather_code_files(code_root: Path, dest_jsonl: Path, max_examples: int | None = None):
    # Project_CodeNet stores many problems; we will iterate and pick code files
    out_count = 0
    with dest_jsonl.open("w", encoding="utf-8") as fout:
        # Walk tree for source code files with common extensions
        exts = {".py", ".java", ".cpp", ".c", ".cs", ".js"}
        for root, dirs, files in os.walk(code_root):
            for f in files:
                if Path(f).suffix.lower() in exts:
                    p = Path(root) / f
                    try:
                        text = p.read_text(encoding="utf-8")
                    except Exception:
                        # skip binary or problematic encodings
                        continue

                    # Build a simple instruction - treat file contents as completion and create a request prompt
                    prompt = (
                        "Write a function or program that satisfies the following specification.\n"
                        "Provide only the final code in the requested language without additional explanation.\n"
                        "---\n"
                        f"File: {p.name}\n"
                        "---\n"
                        "Requirements: Implement the described functionality.\n"
                        "Return only the code.\n"
                    )

                    # For Program synthesis, the 'completion' will be the code file content
                    item = {"prompt": prompt, "completion": text}
                    fout.write(json.dumps(item, ensure_ascii=False) + "\n")
                    out_count += 1
                    if max_examples and out_count >= max_examples:
                        print(f"Reached max_examples={max_examples}")
                        return out_count
    return out_count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default="https://github.com/IBM/Project_CodeNet.git")
    parser.add_argument("--target", default="./Project_CodeNet")
    parser.add_argument("--out", default="./codenet_dataset.jsonl")
    parser.add_argument("--max-examples", type=int, default=None)
    args = parser.parse_args()

    target = Path(args.target).resolve()
    out = Path(args.out).resolve()

    git_clone_or_pull(args.repo, target)
    print("Gathering code files (this may take a while)...")
    n = gather_code_files(target, out, max_examples=args.max_examples)
    print(f"Wrote {n} examples to {out}")


if __name__ == "__main__":
    main()
