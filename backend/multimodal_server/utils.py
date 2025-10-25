"""Helper utilities for multimodal server: remote provider wrappers and small local helpers."""
import os
import base64
import uuid
from typing import Optional

import requests

HF_TOKEN = os.getenv('HF_API_TOKEN')
OPENAI_KEY = os.getenv('OPENAI_API_KEY')


def save_bytes_as_file(b: bytes, out_dir: str, suffix: str = '.png') -> str:
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    filename = f"{uuid.uuid4().hex}{suffix}"
    path = os.path.join(out_dir, filename)
    with open(path, 'wb') as f:
        f.write(b)
    return path


def call_hf_image(prompt: str, model: str = 'runwayml/stable-diffusion-v1-5') -> bytes:
    if not HF_TOKEN:
        raise RuntimeError('HF_API_TOKEN not set')
    url = f'https://api-inference.huggingface.co/models/{model}'
    headers = {
        'Authorization': f'Bearer {HF_TOKEN}'
    }
    payload = {'inputs': prompt}
    resp = requests.post(url, headers=headers, json=payload, timeout=120)
    if resp.status_code != 200:
        raise RuntimeError(f"HF inference failed: {resp.status_code} {resp.text}")
    return resp.content


def call_openai_image(prompt: str, size: str = '512x512') -> bytes:
    try:
        import openai
    except Exception as e:
        raise RuntimeError('openai package not available')
    if not OPENAI_KEY:
        raise RuntimeError('OPENAI_API_KEY not set')
    openai.api_key = OPENAI_KEY
    resp = openai.Image.create(prompt=prompt, n=1, size=size)
    b64 = resp['data'][0].get('b64_json')
    if not b64:
        raise RuntimeError('OpenAI image API did not return base64 content')
    return base64.b64decode(b64)


def assemble_video_with_ffmpeg(frames_dir: str, output_path: str, fps: int = 8) -> str:
    """Uses ffmpeg (must be installed) to assemble frames into a video.
    frames_dir should contain sequential frame PNGs. Output path will be written.
    """
    import subprocess
    # ffmpeg command: ffmpeg -y -framerate {fps} -i frame_%04d.png -c:v libx264 -pix_fmt yuv420p out.mp4
    cmd = [
        'ffmpeg', '-y', '-framerate', str(fps),
        '-i', os.path.join(frames_dir, 'frame_%04d.png'),
        '-c:v', 'libx264', '-pix_fmt', 'yuv420p', output_path
    ]
    res = subprocess.run(cmd, capture_output=True)
    if res.returncode != 0:
        raise RuntimeError(f'ffmpeg failed: {res.stderr.decode()}')
    return output_path


def create_cube_obj(out_path: str) -> str:
    """Create a simple cube OBJ as a 3D placeholder."""
    vertices = [
        (-1, -1, -1),
        (1, -1, -1),
        (1, 1, -1),
        (-1, 1, -1),
        (-1, -1, 1),
        (1, -1, 1),
        (1, 1, 1),
        (-1, 1, 1),
    ]
    faces = [
        (1,2,3,4),
        (5,8,7,6),
        (1,5,6,2),
        (2,6,7,3),
        (3,7,8,4),
        (5,1,4,8),
    ]
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    with open(out_path, 'w') as f:
        for v in vertices:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")
        for face in faces:
            # OBJ is 1-indexed
            f.write('f ' + ' '.join(str(i) for i in face) + '\n')
    return out_path
