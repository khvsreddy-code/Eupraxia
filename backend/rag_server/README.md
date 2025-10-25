Eupraxia RAG server (minimal scaffold)

Purpose
- Local RAG server: embed documents, persist vector DB, retrieve and generate.
- Uses a small embedding model (SentenceTransformers) and Chroma for persistence.
- Generation: tries local Llama via `llama-cpp-python` if `LOCAL_LLM_PATH` set; otherwise falls back to OpenAI (if `OPENAI_API_KEY` is provided).

Quick setup (Windows PowerShell)
1. Create a venv and activate

```powershell
cd backend\rag_server
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Copy your API keys into `.env` (DO NOT COMMIT `.env`)

```powershell
# COPY the file you provided into .env in this folder (this avoids printing keys)
Get-Content "C:\Users\harin\Desktop\vscode repos\eupraxiapl vsc\apikeysforeupraxia.txt" -Raw | Out-File -FilePath .\.env -Encoding ascii
```

3. Run server

```powershell
uvicorn app:app --host 127.0.0.1 --port 8000 --reload
```

Endpoints
- POST /ingest { id?, text, metadata? }  -> chunk & store
- POST /search { query, top_k } -> returns documents
- POST /generate { prompt, top_k, use_local } -> retrieval + local or OpenAI generation
- GET /health

Notes & resource guidance
- On your AMD 660 + 8GB RAM: use small embedding model (`all-MiniLM-L6-v2`) and keep chunk sizes ~300 words and top_k small (2–4).
- For heavy multimodal tasks (high-res images, video, large 3D generation), use cloud GPU / remote inference. Local image/video generation on 8GB will be slow or impossible at high quality.
- If you want GPU acceleration on Windows for local LLMs, consider WSL2 + Linux builds and `llama.cpp` with Vulkan support — this is advanced and hardware dependent.
