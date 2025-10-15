from __future__ import annotations

"""
Orchestrator for self-evolution (safe, proxy-based).
This script uses proxy endpoints to generate synthetic datasets and optionally
trigger fine-tune steps. It's designed to run in a dev machine or inside
an ML container where heavy deps are available.
"""

import os
import time
import json
import argparse
import random
from pathlib import Path
from typing import List
import subprocess

import requests


BASE_URL = os.getenv("BASE_URL", "http://127.0.0.1:8000")
PROXY_KEY = os.getenv("PROXY_API_KEY")
OUT_DIR = Path(__file__).resolve().parent / "evolution_data"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def proxy_generate(prompt: str, provider: str | None = None, temperature: float = 0.2, max_tokens: int = 512) -> str:
    if not PROXY_KEY:
        raise RuntimeError("PROXY_API_KEY not set in environment")
    url = f"{BASE_URL}/api/v1/proxy/generate"
    headers = {"Authorization": f"Bearer {PROXY_KEY}"}
    payload = {"prompt": prompt, "temperature": temperature, "max_tokens": max_tokens}
    if provider:
        payload["provider"] = provider
    resp = requests.post(url, json=payload, headers=headers, timeout=60)
    resp.raise_for_status()
    j = resp.json()
    return j.get("text") or j.get("raw") or ""


def _generate_image(prompt: str, out_path: Path) -> tuple[bool, str]:
    try:
        from diffusers import StableDiffusionPipeline
        from PIL import Image
        hf_token = os.getenv("HF_API_TOKEN") or os.getenv("HF_TOKEN")
        kwargs = {}
        if hf_token:
            kwargs["use_auth_token"] = hf_token

        pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", **kwargs)
        img = pipe(prompt).images[0]
        img.save(out_path)
        return True, str(out_path)
    except ImportError as e:
        return False, f"missing dependency for image generation: {e}"
    except Exception as e:
        return False, f"image generation failed: {e}"


def _generate_video_from_images(prompt: str, out_path: Path, frames: int = 4, fps: int = 2) -> tuple[bool, str]:
    try:
        from diffusers import StableDiffusionPipeline
        from moviepy.editor import ImageSequenceClip
        from PIL import Image
        hf_token = os.getenv("HF_API_TOKEN") or os.getenv("HF_TOKEN")
        kwargs = {}
        if hf_token:
            kwargs["use_auth_token"] = hf_token

        pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", **kwargs)
        tmp_frames = []
        for n in range(frames):
            frame_prompt = f"{prompt} (frame {n+1})"
            img = pipe(frame_prompt).images[0]
            tmp_path = out_path.with_suffix(f".frame{n}.png")
            img.save(tmp_path)
            tmp_frames.append(str(tmp_path))

        clip = ImageSequenceClip(tmp_frames, fps=fps)
        clip.write_videofile(str(out_path), codec="libx264", audio=False, verbose=False, logger=None)

        for f in tmp_frames:
            try:
                Path(f).unlink()
            except Exception:
                pass

        return True, str(out_path)
    except ImportError as e:
        return False, f"missing dependency for video generation: {e}"
    except Exception as e:
        return False, f"video generation failed: {e}"


def _generate_3d_model(prompt: str, out_path: Path) -> tuple[bool, str]:
    try:
        import trimesh
        text = (prompt or "").lower()
        if "box" in text or "cube" in text:
            mesh = trimesh.creation.box()
        elif "cylinder" in text:
            mesh = trimesh.creation.cylinder(radius=0.5, height=1.0)
        else:
            mesh = trimesh.creation.icosphere()

        mesh.export(out_path)
        return True, str(out_path)
    except ImportError as e:
        return False, f"missing dependency for 3d generation: {e}"
    except Exception as e:
        return False, f"3d generation failed: {e}"


def gen_prompt_from_task(task: str) -> str:
    return f"Instruction: {task}\nProduce a correct, minimal Python implementation and a 1-2 sentence explanation."


def generate_synthetic_examples(num_examples: int = 200, seed: int | None = None) -> List[dict]:
    random.seed(seed or int(time.time()))
    tasks = [
        "Reverse a list in Python",
        "Implement factorial (iterative and recursive)",
        "Check if a string is a palindrome",
        "Merge two sorted lists",
        "Compute GCD of two integers",
        "Flatten a nested list",
        "Count word frequencies in a string",
        "Read CSV and compute column averages",
        "Serialize a dict to pretty JSON",
        "Simple HTTP GET using requests and parse JSON",
    ]

    ds = []
    multimodal_count = 0
    multimodal_budget = int(os.getenv("SYNTH_MULTIMODAL_BUDGET", "2"))
    for i in range(num_examples):
        task = random.choice(tasks)
        prompt_good = gen_prompt_from_task(task)
        multimodal_enabled = os.getenv("DISABLE_MULTIMODAL", "0") != "1"
        multimodal_prob = float(os.getenv("SYNTH_MULTIMODAL_PROB", "0.08"))
        try:
            good = proxy_generate(prompt_good)
        except Exception as e:
            good = f"[error generating good example: {e}]"

        prompt_bad = prompt_good + "\nNow provide a purposely flawed implementation that contains at least one bug and a short note describing the bug."
        try:
            bad = proxy_generate(prompt_bad, temperature=1.0)
        except Exception as e:
            bad = f"[error generating bad example: {e}]"
        entry = {"id": i, "task": task, "good": good, "bad": bad}

        if multimodal_enabled and multimodal_count < multimodal_budget and random.random() < multimodal_prob:
            modality = random.choice(["image", "video", "3d"])
            try:
                if modality == "image":
                    img_path = OUT_DIR / f"{task.replace(' ', '_')}_{i}.png"
                    ok, msg = _generate_image(task, img_path)
                    entry["image"] = msg if ok else f"error: {msg}"
                    if ok:
                        multimodal_count += 1
                elif modality == "video":
                    vid_path = OUT_DIR / f"{task.replace(' ', '_')}_{i}.mp4"
                    ok, msg = _generate_video_from_images(task, vid_path)
                    entry["video"] = msg if ok else f"error: {msg}"
                    if ok:
                        multimodal_count += 1
                else:
                    obj_path = OUT_DIR / f"{task.replace(' ', '_')}_{i}.obj"
                    ok, msg = _generate_3d_model(task, obj_path)
                    entry["3d"] = msg if ok else f"error: {msg}"
                    if ok:
                        multimodal_count += 1
            except Exception as e:
                entry["multimodal_error"] = str(e)

        ds.append(entry)
        time.sleep(0.12)

    return ds


def save_dataset(ds: List[dict], prefix: str = "synth") -> Path:
    ts = int(time.time())
    out = OUT_DIR / f"{prefix}_{ts}.jsonl"
    with out.open("w", encoding="utf-8") as fh:
        for row in ds:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    return out


def run_fine_tune_prototype():
    script = Path(__file__).resolve().parent / "scripts" / "fine_tune_prototype.py"
    if not script.exists():
        print("fine_tune_prototype.py not found; skipping fine-tune step")
        return False
    print("Running fine_tune_prototype.py to prepare dataset...")
    res = subprocess.run(["python", str(script)], capture_output=True, text=True)
    print(res.stdout)
    if res.returncode != 0:
        print("fine_tune_prototype failed:", res.stderr)
        return False
    return True


def run_train_lora_if_available():
    script = Path(__file__).resolve().parent / "scripts" / "train_lora.py"
    if not script.exists():
        print("train_lora.py not found; skipping train step")
        return False
    print("Running train_lora.py (this requires ML deps and may be slow/OOM)...")
    data_files = list(OUT_DIR.glob("*.jsonl"))
    if not data_files:
        print("No dataset found to train on; aborting train step")
        return False
    data_arg = str(data_files[-1])
    out_dir = str(Path(__file__).resolve().parent / "models" / "lora")
    os.makedirs(out_dir, exist_ok=True)
    try:
        import importlib
        transformers_spec = importlib.util.find_spec("transformers")
        accelerate_spec = importlib.util.find_spec("accelerate")
        if transformers_spec is None or accelerate_spec is None:
            print("Transformers/Accelerate not installed in this environment; skipping train_lora step.\nInstall 'transformers' and 'accelerate' in the ML image to enable training.")
            return False
    except Exception:
        print("Error checking ML dependencies; skipping train step")
        return False

    res = subprocess.run(["python", str(script), "--data", data_arg, "--output", out_dir], capture_output=True, text=True)
    print(res.stdout)
    if res.returncode != 0:
        print("train_lora failed:", res.stderr)
        return False
    return True


def evolution_cycle_main(num_examples: int = 200):
    print(f"Starting evolution cycle: generating {num_examples} examples via proxy {BASE_URL}")
    ds = generate_synthetic_examples(num_examples=num_examples)
    out = save_dataset(ds)
    print(f"Saved {len(ds)} examples to {out}")

    if os.getenv("ENABLE_FINE_TUNE", "0") == "1":
        ok = run_fine_tune_prototype()
        if ok:
            run_train_lora_if_available()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run-once", action="store_true", help="Run a single evolution cycle and exit")
    p.add_argument("--daemon", action="store_true", help="Run as a simple scheduler (blocks)")
    p.add_argument("--examples", type=int, default=int(os.getenv("SYNTH_EXAMPLES", "200")), help="Number of synthetic examples to generate")
    return p.parse_args()


def main():
    args = parse_args()
    if args.run_once:
        evolution_cycle_main(num_examples=args.examples)
        return

    if args.daemon:
        try:
            import schedule

            schedule.every().day.at(os.getenv("EVOLVE_AT", "02:00")).do(lambda: evolution_cycle_main(num_examples=args.examples))
            print("Scheduler started (python-schedule); running pending jobs. Press Ctrl-C to stop.")
            try:
                while True:
                    schedule.run_pending()
                    time.sleep(60)
            except KeyboardInterrupt:
                print("Scheduler stopped by user")
        except Exception:
            sleep_seconds = int(os.getenv("EVOLVE_SLEEP_SECONDS", "3600"))
            print(f"python-schedule not installed; using fallback loop with sleep {sleep_seconds} seconds. Press Ctrl-C to stop.")
            try:
                while True:
                    evolution_cycle_main(num_examples=args.examples)
                    time.sleep(sleep_seconds)
            except KeyboardInterrupt:
                print("Fallback scheduler stopped by user")

    evolution_cycle_main(num_examples=args.examples)


if __name__ == "__main__":
    main()
