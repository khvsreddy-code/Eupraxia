from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import Optional
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Eupraxia API", version="1.0.0")

# CORS - Allow frontend to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your Vercel domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

security = HTTPBearer()

import logging
from datetime import datetime
from typing import List

# Lazy import pattern for RAGManager: avoid importing heavy ML libs at module import time
RAGManager = None
rag_manager = None

def get_rag_manager():
    """Lazily import and instantiate RAGManager. Returns instance or raises ImportError."""
    global RAGManager, rag_manager
    if rag_manager is not None:
        return rag_manager
    try:
        if RAGManager is None:
            from .rag_manager import RAGManager as _RAG
            RAGManager = _RAG
        rag_manager = RAGManager()
        return rag_manager
    except Exception as e:
        logging.exception("Failed to initialize RAGManager: %s", e)
        raise
from .feedback_store import save_feedback, load_feedback
from .memory_store import add_memory, get_memories

# Setup logging for AI/model errors
LOG_PATH = os.path.join(os.path.dirname(__file__), "ai.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler(LOG_PATH), logging.StreamHandler()],
)


# Simple ModelManager: uses transformers when explicitly enabled via env var
class ModelManager:
    def __init__(self):
        self.use_real = os.getenv("USE_REAL_MODEL", "0") == "1"
        self.available = False
        self._model = None
        if self.use_real:
            try:
                from transformers import pipeline
                model_name = os.getenv("MODEL_NAME", "gpt2")
                # load a text-generation pipeline (may be slow)
                logging.info(f"Attempting to load model '{model_name}'")
                self._model = pipeline("text-generation", model=model_name)
                self.available = True
                logging.info("Model loaded successfully")
            except Exception as e:
                logging.exception("Failed to load real model, falling back to mock.")
                self.available = False
        else:
            logging.info("USE_REAL_MODEL not set; using mock AI responses")

    def generate_text(self, prompt: str, max_tokens: int = 500, temperature: float = 0.7):
        if self.available and self._model is not None:
            try:
                out = self._model(prompt, max_length=max_tokens, do_sample=True, temperature=temperature)
                text = out[0]["generated_text"]
                tokens = len(text.split())
                return text, tokens
            except Exception as e:
                logging.exception("Error while generating text with real model")
                # fall through to mock

        # Mock behavior: safe echo + helpful hint
        text = f"[Mock AI] Response to: {prompt}\n\n(This is a mock response. Set environment variable USE_REAL_MODEL=1 and install transformers to enable a real model.)"
        return text, len(text.split())


# Instantiate global model manager
model_manager = ModelManager()

# RAG manager will be lazily initialized by get_rag_manager() to avoid importing
# heavy ML libraries at module import time on startup.
rag_manager = None

# Providers-based code generation
try:
    from .providers import try_providers
except Exception:
    # relative import may fail when running script directly; try absolute
    try:
        from providers import try_providers
    except Exception:
        try_providers = None


# Simple proxy endpoints for third-party provider APIs so secrets remain server-side
class ProxyRequest(BaseModel):
    prompt: str
    provider: Optional[str] = None
    max_tokens: Optional[int] = 2048
    temperature: Optional[float] = 0.7


@app.post("/api/v1/proxy/generate")
async def proxy_generate(req: ProxyRequest, credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Proxy generate requests to configured providers (deepseek, blackbox, openai, etc.).

    This keeps API keys on the server. The frontend should call this endpoint instead of
    calling provider APIs directly.
    """
    try:
        # Verify API key for proxy usage
        proxy_key = os.getenv("PROXY_API_KEY")
        if not proxy_key:
            logging.warning("PROXY_API_KEY not configured; rejecting proxy request")
            raise HTTPException(status_code=401, detail="Proxy API key not configured")

        incoming = credentials.credentials if credentials else None
        if incoming != proxy_key:
            logging.warning("Invalid proxy auth attempt")
            raise HTTPException(status_code=403, detail="Forbidden")

        provider = (req.provider or os.getenv("DEFAULT_PROVIDER", "deepseek")).lower()

        # If providers helper is available, use it for provider chain or single provider
        if try_providers is not None:
            # try_providers expects a list; pass single provider
            try:
                res = await try_providers(req.prompt, [provider])
                return {"provider": res.get("provider"), "text": res.get("text"), "raw": res.get("raw")}
            except Exception as e:
                logging.exception("Provider call failed: %s", e)
                # fall through to local mock

        # Fallback: simple local model manager
        text, tokens = model_manager.generate_text(req.prompt, max_tokens=req.max_tokens or 512, temperature=req.temperature or 0.7)
        return {"provider": "mock", "text": text, "tokens_used": tokens}
    except Exception as e:
        logging.exception("Proxy generate failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))



@app.get("/admin/health")
async def admin_health():
    """Basic health endpoint for orchestration checks."""
    return {"status": "ok", "time": datetime.utcnow().isoformat()}


@app.get("/admin/smokecheck")
async def admin_smokecheck():
    """Return quick diagnostics: files and env var presence."""
    base = os.path.dirname(__file__)
    files = {}
    paths = [
        os.path.join(base, "evolution_data"),
        os.path.join(base, "evolution_data", ""),
        os.path.join(base, "feedback.jsonl"),
        os.path.join(base, "memory.jsonl"),
    ]
    for p in paths:
        try:
            files[p] = os.path.exists(p)
        except Exception:
            files[p] = False

    env_ok = {
        "PROXY_API_KEY": bool(os.getenv("PROXY_API_KEY")),
        "HF_API_TOKEN": bool(os.getenv("HF_API_TOKEN")),
    }

    return {"status": "ok", "files": files, "env": env_ok}


class RAGIndexRequest(BaseModel):
    docs: List[str]


@app.post("/api/v1/rag/index")
async def rag_index(req: RAGIndexRequest, credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Index a list of document strings for RAG retrieval. Requires proxy auth."""
    proxy_key = os.getenv("PROXY_API_KEY")
    incoming = credentials.credentials if credentials else None
    if incoming != proxy_key:
        raise HTTPException(status_code=403, detail="Forbidden")

    try:
        docs = req.docs
        try:
            rm = get_rag_manager()
        except Exception as e:
            logging.exception("RAG manager unavailable: %s", e)
            raise HTTPException(status_code=503, detail="RAG manager unavailable; install sentence-transformers/faiss or use container")

        rm.build_index(docs)
        return {"status": "ok", "indexed_docs": len(docs)}
    except HTTPException:
        raise
    except Exception as e:
        logging.exception("RAG index failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


class RAGQueryRequest(BaseModel):
    query: str
    k: int = 3
    provider: Optional[str] = None


@app.post("/api/v1/rag/query")
async def rag_query(req: RAGQueryRequest, credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Perform a RAG query: retrieve top-k docs and call provider for answer."""
    proxy_key = os.getenv("PROXY_API_KEY")
    incoming = credentials.credentials if credentials else None
    if incoming != proxy_key:
        raise HTTPException(status_code=403, detail="Forbidden")

    try:
        try:
            rm = get_rag_manager()
        except Exception as e:
            logging.exception("RAG manager unavailable: %s", e)
            raise HTTPException(status_code=503, detail="RAG manager unavailable; install sentence-transformers/faiss or use container")

        results = rm.retrieve(req.query, k=req.k)
        # Build context string
        context = "\n---\n".join([f"(score={score:.3f})\n{doc}" for _, score, doc in results])
        provider = (req.provider or os.getenv("DEFAULT_PROVIDER", "deepseek")).lower()

        # Augment prompt
        prompt = f"Use the following retrieved documents as context when answering the question.\n\nContext:\n{context}\n\nQuestion: {req.query}\nAnswer:" 

        if try_providers is not None:
            try:
                res = await try_providers(prompt, [provider])
                return {"provider": res.get("provider"), "text": res.get("text"), "raw": res.get("raw"), "retrieved": results}
            except Exception as e:
                logging.exception("RAG provider call failed: %s", e)
                # fall through to local mock

        # Fallback to local mock model manager
        text, tokens = model_manager.generate_text(prompt)
        return {"provider": "mock", "text": text, "retrieved": results}
    except Exception as e:
        logging.exception("RAG query failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/langchain/qa")
async def langchain_qa(req: RAGQueryRequest, credentials: HTTPAuthorizationCredentials = Depends(security)):
    """LangChain-style RetrievalQA endpoint (server-side). Uses RAGManager + provider chain.

    This endpoint mimics a RetrievalQA chain: it retrieves top-k documents and then asks the
    configured provider to produce an answer using the retrieved context.
    """
    proxy_key = os.getenv("PROXY_API_KEY")
    incoming = credentials.credentials if credentials else None
    if incoming != proxy_key:
        raise HTTPException(status_code=403, detail="Forbidden")

    try:
        try:
            rm = get_rag_manager()
        except Exception as e:
            logging.exception("RAG manager unavailable: %s", e)
            raise HTTPException(status_code=503, detail="RAG manager unavailable; install sentence-transformers/faiss or use container")

        results = rm.retrieve(req.query, k=req.k)
        context = "\n---\n".join([f"(score={score:.3f})\n{doc}" for _, score, doc in results])
        provider = (req.provider or os.getenv("DEFAULT_PROVIDER", "deepseek")).lower()
        prompt = f"You are a helpful assistant. Use the retrieved documents below to answer the question.\n\nContext:\n{context}\n\nQuestion: {req.query}\nAnswer:"

        if try_providers is not None:
            try:
                res = await try_providers(prompt, [provider])
                return {"provider": res.get("provider"), "text": res.get("text"), "retrieved": results}
            except Exception as e:
                logging.exception("LangChain QA provider call failed: %s", e)

        text, tokens = model_manager.generate_text(prompt)
        return {"provider": "mock", "text": text, "retrieved": results}
    except Exception as e:
        logging.exception("LangChain QA failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


class CodeGenRequest(BaseModel):
    prompt: str
    language: Optional[str] = "python"
    max_tokens: Optional[int] = 512
    post_process: Optional[bool] = True


class FeedbackRequest(BaseModel):
    request_id: Optional[str] = None
    rating: int  # 1-5
    notes: Optional[str] = None
    user_id: Optional[str] = None


@app.post("/api/v1/feedback")
async def submit_feedback(req: FeedbackRequest):
    """Persist user feedback and append it to lightweight memory for evolution seeds."""
    try:
        rec = {
            "timestamp": datetime.utcnow().isoformat(),
            "request_id": req.request_id,
            "rating": req.rating,
            "notes": req.notes,
            "user_id": req.user_id,
        }
        save_feedback(rec)
        # Add to memory for future evolution cycles
        add_memory({"type": "feedback", "data": rec})
        return {"status": "ok"}
    except Exception as e:
        logging.exception("Failed to save feedback: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/feedback/recent")
async def recent_feedback(limit: int = 50):
    try:
        items = load_feedback(limit)
        return {"count": len(items), "items": items}
    except Exception as e:
        logging.exception("Failed to load feedback: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/generate/code")
async def generate_code(req: CodeGenRequest):
    """Generate code using configured providers (OpenAI, Anthropic, Google Palm) in priority order."""
    try:
        providers_env = os.getenv("PROVIDER_PRIORITY", "openai,anthropic,google")
        providers = [p.strip() for p in providers_env.split(",") if p.strip()]
        prompt = f"Generate {req.language} code for the following request:\n{req.prompt}"

        # If no external providers are configured, use the local ModelManager mock
        if try_providers is None:
            text, tokens = model_manager.generate_text(prompt, max_tokens=req.max_tokens)
            return {"provider": "mock", "code": text, "tokens_used": tokens}

        # Helper: extract first fenced code block if present
        def extract_code(text: str) -> str:
            import re
            # match ```lang\n...``` or ```\n...```
            m = re.search(r"```(?:[a-zA-Z0-9_+-]*)\n([\s\S]*?)```", text)
            if m:
                return m.group(1).strip()
            return text

        # Request code from provider chain
        try:
            res = await try_providers(prompt, providers)
            provider = res.get("provider")
            raw_text = res.get("text") or ""
            code_text = extract_code(raw_text)

            diagnostics = []

            # Optional post-processing: syntax check and auto-fix loop
            if req.post_process:
                # Syntax check using ast for Python
                if req.language.lower() == "python":
                    import ast
                    try:
                        ast.parse(code_text)
                        diagnostics.append({"type": "syntax", "status": "ok"})
                    except SyntaxError as se:
                        diagnostics.append({"type": "syntax", "status": "error", "message": str(se)})
                        # Ask provider to fix the code by sending the code and error
                        fix_prompt = (
                            f"The following Python code has a syntax error: {se}\n\n"
                            f"Code:\n```python\n{code_text}\n```\n\n"
                            "Please return only the corrected full code block in a fenced ```python``` block and nothing else."
                        )
                        try:
                            fix_res = await try_providers(fix_prompt, providers)
                            fixed_raw = fix_res.get("text") or ""
                            fixed_code = extract_code(fixed_raw)
                            # Re-parse
                            try:
                                ast.parse(fixed_code)
                                code_text = fixed_code
                                diagnostics.append({"type": "auto_fix", "status": "fixed"})
                            except SyntaxError as se2:
                                diagnostics.append({"type": "auto_fix", "status": "failed", "message": str(se2)})
                        except Exception:
                            diagnostics.append({"type": "auto_fix", "status": "error", "message": "provider_fix_failed"})

                # Run Black to format Python code if available
                if req.language.lower() == "python":
                    try:
                        import black
                        # Use mode default - wrap_line_length from black >=22
                        try:
                            mode = black.Mode()
                        except Exception:
                            mode = None
                        formatted = black.format_str(code_text, mode=mode) if mode is not None else black.format_str(code_text)
                        code_text = formatted
                        diagnostics.append({"type": "format", "status": "ok"})
                    except Exception:
                        diagnostics.append({"type": "format", "status": "unavailable"})

            return {"provider": provider, "code": code_text, "raw": res.get("raw"), "diagnostics": diagnostics}
        except Exception as e:
            # Log and fallback to local mock model manager
            logging.exception("Provider chain failed, falling back to mock: %s", e)
            text, tokens = model_manager.generate_text(prompt, max_tokens=req.max_tokens)
            return {"provider": "mock", "code": text, "tokens_used": tokens}
    except Exception as e:
        logging.exception("Code generation failed")
        raise HTTPException(status_code=500, detail=str(e))

# Request/Response Models
class TextGenerationRequest(BaseModel):
    prompt: str
    max_tokens: int = 500
    temperature: float = 0.7

class TextGenerationResponse(BaseModel):
    text: str
    tokens_used: int

class ImageGenerationRequest(BaseModel):
    prompt: str
    width: int = 512
    height: int = 512

# Simple in-memory storage (replace with DB later)
conversations = {}

@app.get("/")
async def root():
    return {
        "message": "Eupraxia-Pleroma API",
        "status": "running",
        "version": "1.0.0"
    }

@app.post("/api/v1/generate/text", response_model=TextGenerationResponse)
async def generate_text(request: TextGenerationRequest):
    """Generate text using AI model"""
    try:
        # TODO: Replace with actual AI model
        # For now, simple echo response
        response_text = f"AI Response to: {request.prompt}\n\nThis is a placeholder. Integrate your AI model here."
        
        return TextGenerationResponse(
            text=response_text,
            tokens_used=len(response_text.split())
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/chat")
async def chat(request: TextGenerationRequest):
    """Chat endpoint with conversation history"""
    try:
        # Simple conversation memory
        conversation_id = "default"
        
        if conversation_id not in conversations:
            conversations[conversation_id] = []
        
        # Add user message
        conversations[conversation_id].append({
            "role": "user",
            "content": request.prompt
        })
        
        # Generate response (placeholder)
        ai_response = f"Greg says: I heard you say '{request.prompt}'. I'm still learning! *wags tail*"
        
        conversations[conversation_id].append({
            "role": "assistant",
            "content": ai_response
        })
        
        return {
            "response": ai_response,
            "conversation_history": conversations[conversation_id]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/analyze/webpage")
async def analyze_webpage(url: str):
    """Analyze a webpage"""
    try:
        # Placeholder for webpage analysis
        return {
            "url": url,
            "title": "Webpage Title (Placeholder)",
            "summary": "This will analyze the webpage when implemented.",
            "main_content": "Content extraction coming soon!"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/memory")
async def get_memory():
    """Retrieve conversation memory"""
    return {"conversations": conversations}

@app.delete("/api/v1/memory")
async def clear_memory():
    """Clear all conversation memory"""
    conversations.clear()
    return {"message": "Memory cleared"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)