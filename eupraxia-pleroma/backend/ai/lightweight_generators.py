"""Lightweight placeholder generators to run on CPU-only machines.
These provide fast, deterministic outputs so the evolution loop can run
without downloading large models.
"""
import io
import math
import random
from typing import Dict, Any, Optional, List
from PIL import Image, ImageDraw, ImageFont
import numpy as np


class LightweightAICoder:
    def __init__(self):
        self.name = "lightweight-coder"

    def generate_code(self, prompt: str, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        # Generate a simple, plausible code template based on keywords
        p = prompt.lower()
        if "fibonacci" in p:
            code = (
                "def fibonacci(n):\n"
                "    a, b = 0, 1\n"
                "    for _ in range(n):\n"
                "        a, b = b, a + b\n"
                "    return a\n"
            )
        elif "sort" in p or "quicksort" in p:
            code = (
                "def quicksort(arr):\n"
                "    if len(arr) <= 1:\n"
                "        return arr\n"
                "    pivot = arr[len(arr)//2]\n"
                "    left = [x for x in arr if x < pivot]\n"
                "    middle = [x for x in arr if x == pivot]\n"
                "    right = [x for x in arr if x > pivot]\n"
                "    return quicksort(left) + middle + quicksort(right)\n"
            )
        else:
            # Generic function template
            code = (
                "def solution(*args, **kwargs):\n"
                "    \"\"\"Auto-generated solution placeholder.\"\"\"\n"
                "    # Prompt: " + prompt.replace('\n', ' ')[:200] + "\n"
                "    return None\n"
            )

        return {"code": code}

    def self_evaluate_code(self, code: str, language: str = "python") -> float:
        score = 0.0
        if "def " in code or "function " in code:
            score += 0.4
        if "#" in code or "//" in code:
            score += 0.2
        if "test" in code.lower():
            score += 0.2
        if "return" in code:
            score += 0.2
        return min(score, 1.0)


class LightweightImageGenerator:
    def __init__(self):
        self.name = "lightweight-image"

    def generate_image(self, prompt: str, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if options is None:
            options = {}
        w = int(options.get("width", 512))
        h = int(options.get("height", 512))
        # Create a simple gradient background
        img = Image.new("RGB", (w, h), color=(0, 0, 0))
        draw = ImageDraw.Draw(img)
        for i in range(h):
            color = int(255 * (i / max(1, h - 1)))
            draw.line([(0, i), (w, i)], fill=(color // 2, color, 255 - color // 3))
        # Overlay the prompt text
        try:
            font = ImageFont.load_default()
            draw.text((8, 8), prompt[:200], fill=(255, 255, 255), font=font)
        except Exception:
            draw.text((8, 8), prompt[:200], fill=(255, 255, 255))

        return {"image": img, "metadata": {"width": w, "height": h}}


class LightweightVideoGenerator:
    def __init__(self):
        self.name = "lightweight-video"

    def generate_video(self, prompt: str, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if options is None:
            options = {}
        fps = max(1, int(options.get("fps", 4)))
        duration = max(1, int(options.get("duration", 2)))
        frames = []
        img_gen = LightweightImageGenerator()
        for i in range(fps * duration):
            frame_prompt = f"{prompt} (frame {i+1})"
            frames.append(np.array(img_gen.generate_image(frame_prompt, {"width": 256, "height": 256})["image"]))
        return {"frames": frames, "fps": fps, "duration": duration}


class Lightweight3DGenerator:
    def __init__(self):
        self.name = "lightweight-3d"

    def generate_3d_model(self, prompt: str, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        # Return a simple cube-like mesh descriptor
        mesh = {
            "vertices": 8,
            "faces": 12,
            "format": "simple",
            "description": f"Placeholder 3D mesh for: {prompt[:80]}"
        }
        return {"model": mesh}


class LightweightMusicGenerator:
    def __init__(self):
        self.name = "lightweight-music"

    def generate_music(self, prompt: str, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        sr = 22050
        duration = max(1, int(options.get("duration", 3))) if options else 3
        t = np.linspace(0, duration, int(sr * duration), False)
        freq = 440.0 + (hash(prompt) % 200)  # deterministic-ish
        audio = 0.1 * np.sin(2 * math.pi * freq * t)
        return {"audio": audio, "sample_rate": sr}


class LightweightWebsiteGenerator:
    def __init__(self):
        self.name = "lightweight-website"
        self.image_gen = LightweightImageGenerator()

    def generate_website(self, prompt: str, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        # Simple single-page HTML with an auto-generated logo image
        logo = self.image_gen.generate_image(f"Logo for: {prompt}", {"width": 128, "height": 64})["image"]
        buffer = io.BytesIO()
        logo.save(buffer, format="PNG")
        logo_b64 = buffer.getvalue()
        html = f"<html><head><title>{prompt[:60]}</title></head><body><h1>{prompt}</h1><p>Generated site placeholder.</p></body></html>"
        return {"html": html, "logo_bytes": logo_b64}


class LightweightGameGenerator:
    def __init__(self):
        self.name = "lightweight-game"
        self.img = LightweightImageGenerator()

    def generate_game(self, prompt: str, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        # Produce a small game spec and a few placeholder assets
        spec = {
            "title": prompt[:40],
            "genre": options.get("genre") if options else "Action RPG",
            "features": ["Placeholder combat", "Placeholder world", "Procedural map"]
        }
        assets = {
            "screenshot": self.img.generate_image(f"Game screenshot for {prompt}", {"width": 320, "height": 180})["image"]
        }
        return {"spec": spec, "assets": assets}


class LightweightAssistantGenerator:
    def __init__(self):
        self.name = "lightweight-assistant"

    def generate_assistant(self, prompt: str, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        persona = {
            "name": "LightAssist",
            "description": f"A lightweight assistant trained to help with: {prompt[:80]}"
        }
        return {"assistant": persona}


class LightweightMetaEvolution:
    def __init__(self):
        self.history = []

    def evolve_ai(self, prompt: str, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        entry = {"prompt": prompt, "result": "evolved_placeholder", "options": options}
        self.history.append(entry)
        return {"status": "evolved", "entry": entry}


__all__ = [
    "LightweightAICoder",
    "LightweightImageGenerator",
    "LightweightVideoGenerator",
    "Lightweight3DGenerator",
    "LightweightMusicGenerator",
    "LightweightWebsiteGenerator",
    "LightweightGameGenerator",
    "LightweightAssistantGenerator",
    "LightweightMetaEvolution",
]
