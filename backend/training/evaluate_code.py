"""Evaluate Python code for syntax correctness and optional linting.

Usage: python evaluate_code.py path/to/code.py
"""
import sys
from pathlib import Path


def run_syntax_check(code_text: str):
    import ast
    try:
        ast.parse(code_text)
        return True, None
    except Exception as e:
        return False, str(e)


def run_ruff(code_text: str):
    try:
        import subprocess, tempfile
        with tempfile.NamedTemporaryFile('w', suffix='.py', delete=False, encoding='utf-8') as tf:
            tf.write(code_text)
            temp_path = tf.name
        # Run ruff if available
        res = subprocess.run(["ruff", temp_path, "--quiet"], capture_output=True, text=True)
        out = res.stdout + res.stderr
        return res.returncode == 0, out
    except Exception as e:
        return False, f"ruff not available or failed: {e}"


def main():
    if len(sys.argv) < 2:
        print("Usage: python evaluate_code.py <file_or_code>\nIf path exists, file is read; otherwise the arg is treated as code.")
        sys.exit(1)

    arg = sys.argv[1]
    p = Path(arg)
    if p.exists():
        code = p.read_text(encoding='utf-8')
    else:
        code = arg

    ok, msg = run_syntax_check(code)
    if ok:
        print("Syntax: OK")
    else:
        print("Syntax: ERROR", msg)

    lint_ok, lint_out = run_ruff(code)
    print("Ruff OK:", lint_ok)
    if lint_out:
        print(lint_out)


if __name__ == '__main__':
    main()
