Eupraxia Multimodal Server (scaffold)

Purpose
- Provide endpoints for image, video, and simple 3D placeholder generation.
- Prefer remote providers (OpenAI, Hugging Face, Replicate) for heavy tasks and fall back to a local Stable Diffusion POC when available.

Quick setup (recommended: use a venv)

```powershell
cd backend\multimodal_server
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
# If you want local Stable Diffusion, install heavy deps:
# pip install -r requirements-full.txt
```

Environment
- Copy your keys into a `.env` file in this folder (DO NOT COMMIT `.env`). Example keys:

```
OPENAI_API_KEY=sk-...
HF_API_TOKEN=hf_...
```

Run server

```powershell
uvicorn app:app --host 127.0.0.1 --port 9000 --reload
```

Endpoints
- POST /generate/image -> { prompt, width, height, num_inference_steps, guidance_scale, use_remote, provider }
- POST /generate/video -> { prompt, frames, fps, width, height, use_remote }
- POST /generate/3d -> { prompt }
- GET  /multimodal/outputs/{filename} -> serve generated assets

Notes for your hardware (AMD Radeon 660, 8GB RAM)
- Local Stable Diffusion on 8GB is memory constrained and will be slow or fail. Prefer remote providers for high-quality images/video.
- Use low-res outputs (256–512) and few steps (12–25) for local generation to reduce memory/time.
- For 3D/animation pipelines, use Blender on a machine with more RAM/GPU or run headless on cloud.

Design choices
- The server attempts remote provider calls when `use_remote` is true or when a provider key is present.
- If remote providers are unavailable, it attempts a local diffusers Pipeline if dependencies are installed and a model is cached.
- Video generation is implemented by creating frames (from image generation) and stitching them with ffmpeg (ffmpeg must be installed on the host).

Security
- Keep keys in `.env` and use OS-level permissions to restrict access. The server reads keys at startup.

