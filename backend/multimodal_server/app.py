"""Minimal multimodal generation server.
Endpoints:
- POST /generate/image
- POST /generate/video
- POST /generate/3d

Design:
- Prefer remote providers (OpenAI, Hugging Face) when keys available or when `use_remote` true.
- Fall back to a tiny local Stable Diffusion POC if diffusers/torch are installed and a model is cached.
- Save outputs under `OUTPUT_DIR` and expose them at /multimodal/outputs/{filename}
"""
import os
import io
import uuid
import base64
import shutil
import subprocess
from typing import Optional, List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse

load_dotenv()

OPENAI_KEY = os.getenv('OPENAI_API_KEY')
HF_TOKEN = os.getenv('HF_API_TOKEN')
OUTPUT_DIR = os.getenv('MULTIMODAL_OUTPUT_DIR', os.path.join(os.getcwd(), 'backend', 'multimodal_server', 'outputs'))
os.makedirs(OUTPUT_DIR, exist_ok=True)

# utils (local helper functions)
from .utils import save_bytes_as_file, call_hf_image, call_openai_image, assemble_video_with_ffmpeg, create_cube_obj

app = FastAPI(title='Eupraxia Multimodal Server')

# Expose output files
app.mount('/multimodal/outputs', StaticFiles(directory=OUTPUT_DIR), name='multimodal_outputs')

# Request models
class ImageGenerateRequest(BaseModel):
    prompt: str
    width: Optional[int] = 512
    height: Optional[int] = 512
    num_inference_steps: Optional[int] = 20
    guidance_scale: Optional[float] = 7.5
    use_remote: Optional[bool] = True
    provider: Optional[str] = 'auto'  # openai | hf | local | auto

class VideoGenerateRequest(BaseModel):
    prompt: str
    frames: Optional[int] = 8
    fps: Optional[int] = 8
    width: Optional[int] = 512
    height: Optional[int] = 512
    use_remote: Optional[bool] = True
    provider: Optional[str] = 'auto'

class ThreeDGenerateRequest(BaseModel):
    prompt: str
    type: Optional[str] = 'primitive'  # only primitive supported in POC

# Helpers

def _choose_provider(req_provider: str, use_remote: bool) -> str:
    """Pick provider: openai|hf|local. 'auto' prefers OpenAI (if key), else HF (if token), else local."""
    if req_provider and req_provider != 'auto':
        return req_provider
    if use_remote:
        if OPENAI_KEY:
            return 'openai'
        if HF_TOKEN:
            return 'hf'
    return 'local'

async def _generate_image_local(prompt: str, width: int, height: int, steps: int, guidance: float) -> str:
    """Attempt to generate an image locally using diffusers. Lazy-imports to avoid hard dependency.
    Saves PNG and returns path.
    """
    try:
        import torch
        from diffusers import StableDiffusionPipeline
    except Exception as e:
        raise HTTPException(status_code=503, detail='Local pipeline not available (missing diffusers/torch)')

    model_id = os.getenv('LOCAL_SD_MODEL', 'runwayml/stable-diffusion-v1-5')

    # Use CPU by default; on a small machine use low memory settings
    device = 'cpu'

    try:
        pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=None)
        pipe = pipe.to(device)
        result = pipe(prompt, guidance_scale=guidance, num_inference_steps=steps, height=height, width=width)
        img = result.images[0]
        out_path = save_bytes_as_file(img.tobytes() if hasattr(img, 'tobytes') else img.convert('RGBA').tobytes(), OUTPUT_DIR, suffix='.png')
        # Above uses raw bytes; to be safer, save via PIL
        try:
            from PIL import Image
            out_file = os.path.join(OUTPUT_DIR, f"{uuid.uuid4().hex}.png")
            img.save(out_file)
            return out_file
        except Exception:
            return out_path
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Local generation failed: {e}')


@app.post('/generate/image')
async def generate_image(req: ImageGenerateRequest):
    provider = _choose_provider(req.provider, bool(req.use_remote))

    # Remote: OpenAI
    if provider == 'openai':
        try:
            b = call_openai_image(req.prompt, size=f"{req.width}x{req.height}")
            path = save_bytes_as_file(b, OUTPUT_DIR, suffix='.png')
            url = f"/multimodal/outputs/{os.path.basename(path)}"
            return { 'source': 'openai', 'path': path, 'url': url }
        except Exception as e:
            # fallback to HF or local
            print('OpenAI provider failed:', e)
            if HF_TOKEN:
                provider = 'hf'
            else:
                provider = 'local'

    if provider == 'hf':
        try:
            b = call_hf_image(req.prompt)
            path = save_bytes_as_file(b, OUTPUT_DIR, suffix='.png')
            url = f"/multimodal/outputs/{os.path.basename(path)}"
            return { 'source': 'hf', 'path': path, 'url': url }
        except Exception as e:
            print('HF provider failed:', e)
            provider = 'local'

    # Local fallback
    if provider == 'local':
        # Local generation should be low-res on constrained hardware
        if req.width > 512 or req.height > 512:
            req.width = min(req.width, 512)
            req.height = min(req.height, 512)
        try:
            path = await _generate_image_local(req.prompt, req.width, req.height, req.num_inference_steps, req.guidance_scale)
            url = f"/multimodal/outputs/{os.path.basename(path)}"
            return { 'source': 'local', 'path': path, 'url': url }
        except HTTPException as e:
            raise e
        except Exception as e:
            raise HTTPException(status_code=500, detail=f'All providers failed: {e}')

    raise HTTPException(status_code=500, detail='No provider available')

@app.post('/generate/video')
async def generate_video(req: VideoGenerateRequest):
    # Simple POC: generate N frames and stitch with ffmpeg
    frames = max(2, min(64, req.frames or 8))
    fps = max(1, min(30, req.fps or 8))
    tmp_dir = os.path.join(OUTPUT_DIR, f"frames_{uuid.uuid4().hex}")
    os.makedirs(tmp_dir, exist_ok=True)

    provider = _choose_provider(req.provider, bool(req.use_remote))

    # Generate frames
    for i in range(frames):
        prompt = f"{req.prompt} -- frame {i+1} of {frames}"
        try:
            # call generate_image endpoint internally via function to avoid HTTP roundtrip
            img_req = ImageGenerateRequest(prompt=prompt, width=req.width, height=req.height, num_inference_steps=12, guidance_scale=7.0, use_remote=(provider!='local'))
            res = await generate_image(img_req)
            # res['path'] is an absolute path
            # copy to tmp_dir with frame index
            src = res.get('path')
            if not src or not os.path.exists(src):
                raise RuntimeError('Frame generation failed')
            dst = os.path.join(tmp_dir, f"frame_{i:04d}.png")
            shutil.copy(src, dst)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f'Failed to generate frame {i}: {e}')

    out_video = os.path.join(OUTPUT_DIR, f"video_{uuid.uuid4().hex}.mp4")
    try:
        assemble_video_with_ffmpeg(tmp_dir, out_video, fps=fps)
        url = f"/multimodal/outputs/{os.path.basename(out_video)}"
        # Cleanup frames
        try:
            shutil.rmtree(tmp_dir)
        except Exception:
            pass
        return { 'source': provider, 'path': out_video, 'url': url }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'FFMPEG failed: {e}')

@app.post('/generate/3d')
async def generate_3d(req: ThreeDGenerateRequest):
    # POC: produce a simple cube OBJ file and return path
    out_obj = os.path.join(OUTPUT_DIR, f"model_{uuid.uuid4().hex}.obj")
    try:
        create_cube_obj(out_obj)
        url = f"/multimodal/outputs/{os.path.basename(out_obj)}"
        return { 'source': 'poc', 'path': out_obj, 'url': url }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'3D generation failed: {e}')

@app.get('/health')
async def health():
    return { 'status': 'ok', 'openai': bool(OPENAI_KEY), 'hf': bool(HF_TOKEN) }
