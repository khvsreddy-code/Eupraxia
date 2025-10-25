"""Minimal FastAPI RAG server for Eupraxia
- Ingest text -> chunk -> embed -> store in Chroma
- Search -> return top-k docs
- Generate -> retrieve + call local Llama (llama-cpp-python) if available, else OpenAI fallback

Designed for low-RAM dev machines: keep chunk sizes small and top_k low.
"""
import os
import uuid
from typing import Optional, List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

EMB_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
CHROMA_DIR = os.getenv("CHROMA_DIR", "./chroma_db")
LOCAL_LLM_PATH = os.getenv("LOCAL_LLM_PATH", "")

app = FastAPI(title="Eupraxia RAG server")

# Lazy imports to keep startup fast when dependencies are missing
try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

try:
    import chromadb
    from chromadb.config import Settings
except Exception:
    chromadb = None

try:
    from llama_cpp import Llama
except Exception:
    Llama = None

try:
    import openai
except Exception:
    openai = None

# Initialize components if available
embedding_model = None
if SentenceTransformer:
    embedding_model = SentenceTransformer(EMB_MODEL)

chroma_client = None
collection = None
if chromadb and embedding_model:
    chroma_client = chromadb.Client(Settings(chroma_db_impl="duckdb+parquet", persist_directory=CHROMA_DIR))
    collection = chroma_client.get_or_create_collection("documents")

llama = None
if Llama and LOCAL_LLM_PATH:
    try:
        llama = Llama(model_path=LOCAL_LLM_PATH)
    except Exception as e:
        print("Warning: failed to initialize local Llama:", e)

OPENAI_KEY = os.getenv("OPENAI_API_KEY")
if openai and OPENAI_KEY:
    openai.api_key = OPENAI_KEY

# Utilities

def chunk_text(text: str, chunk_size_words: int = 300, overlap: int = 50):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i + chunk_size_words])
        chunks.append(chunk)
        i += chunk_size_words - overlap
    return chunks

# Request models
class IngestRequest(BaseModel):
    id: Optional[str] = None
    text: str
    metadata: Optional[dict] = {}

class SearchRequest(BaseModel):
    query: str
    top_k: Optional[int] = 5

class GenerateRequest(BaseModel):
    prompt: str
    top_k: Optional[int] = 3
    use_local: Optional[bool] = False
    temperature: Optional[float] = 0.2
    max_tokens: Optional[int] = 256

@app.post("/ingest")
async def ingest(req: IngestRequest):
    if not collection or not embedding_model:
        raise HTTPException(status_code=500, detail="Embedding model or vector DB not ready. Install requirements and start again.")
    prefix = req.id or str(uuid.uuid4())
    chunks = chunk_text(req.text, chunk_size_words=300, overlap=50)
    ids = [f"{prefix}_{i}" for i in range(len(chunks))]
    embs = embedding_model.encode(chunks, show_progress_bar=False)
    embs_list = [e.tolist() for e in embs]
    metas = [req.metadata or {} for _ in chunks]
    collection.add(ids=ids, documents=chunks, metadatas=metas, embeddings=embs_list)
    chroma_client.persist()
    return {"ingested": len(chunks), "ids": ids}

@app.post("/search")
async def search(req: SearchRequest):
    if not collection or not embedding_model:
        raise HTTPException(status_code=500, detail="Embedding model or vector DB not ready.")
    emb = embedding_model.encode([req.query], show_progress_bar=False)[0].tolist()
    res = collection.query(query_embeddings=[emb], n_results=req.top_k, include=["documents", "metadatas", "distances"])
    return res

@app.post("/generate")
async def generate(req: GenerateRequest):
    if not collection or not embedding_model:
        raise HTTPException(status_code=500, detail="Embedding model or vector DB not ready.")
    emb = embedding_model.encode([req.prompt], show_progress_bar=False)[0].tolist()
    res = collection.query(query_embeddings=[emb], n_results=req.top_k, include=["documents"])
    docs = []
    if res and "documents" in res and len(res["documents"])>0:
        docs = res["documents"][0]
    context = "\n\n".join(docs)
    combined = f"Context:\n{context}\n\nUser:\n{req.prompt}\n\nAssistant:"

    # Try local Llama first (if requested)
    if llama and req.use_local:
        try:
            out = llama.create(prompt=combined, max_tokens=req.max_tokens, temperature=req.temperature)
            text = out.get("choices", [{"text":""}])[0].get("text", "")
            return {"source":"local", "text": text}
        except Exception as e:
            print("Local Llama error:", e)

    # Fallback to OpenAI if key available
    if openai and OPENAI_KEY:
        try:
            resp = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role":"system","content":"You are Eupraxia assistant. Use the context to answer precisely."},
                    {"role":"user","content": combined}
                ],
                max_tokens=req.max_tokens,
                temperature=req.temperature
            )
            text = resp["choices"][0]["message"]["content"]
            return {"source":"openai", "text": text}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"OpenAI error: {e}")

    # No generator available - return retrieved context
    return {"source":"retrieval_only", "text":"No LLM available (local or OpenAI). Returning top retrieved passages.", "context": docs}

@app.get("/health")
async def health():
    return {"status":"ok", "has_local_llm": bool(llama), "has_openai_key": bool(OPENAI_KEY)}
