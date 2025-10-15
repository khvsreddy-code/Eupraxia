"""
Scheduler runner for evolution cycles.

This script runs `evolution_cycle.py` at the configured cadence and optionally
runs the fine-tune prototype when `ENABLE_FINE_TUNE=1` in the environment.

Use Task Scheduler / cron to start this script on boot or run it in a container.
"""
import os
import time
import subprocess
from datetime import datetime

BASE_DIR = Path = __import__("pathlib").Path(__file__).resolve().parents[1]
EVOLVE_SCRIPT = BASE_DIR / "scripts" / "evolution_cycle.py"
FINE_TUNE_SCRIPT = BASE_DIR / "scripts" / "fine_tune_prototype.py"

INTERVAL_SECONDS = int(os.getenv("EVOLVE_INTERVAL_SEC", str(60 * 60 * 24)))  # default daily


def run_cmd(cmd_list):
    print(f"[{datetime.utcnow().isoformat()}] Running: {' '.join(cmd_list)}")
    try:
        res = subprocess.run(cmd_list, check=True, capture_output=True, text=True)
        print(res.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print("Error:", e.stderr)
        return False


def main():
    print("Scheduler runner started. Press Ctrl-C to stop.")
    while True:
        # Run evolution cycle
        ok = run_cmd(["python", str(EVOLVE_SCRIPT)])
        # Optionally run fine-tune prototype
        if ok and os.getenv("ENABLE_FINE_TUNE", "0") == "1":
            print("ENABLE_FINE_TUNE=1; running fine-tune prototype (no heavy deps are installed by default)")
            run_cmd(["python", str(FINE_TUNE_SCRIPT)])

        time.sleep(INTERVAL_SECONDS)


if __name__ == '__main__':
    main()
