import os
import logging
import asyncio
from typing import Optional

try:
    import openai
    _HAS_OPENAI_PKG = True
except Exception:
    openai = None
    _HAS_OPENAI_PKG = False

try:
    import httpx
except Exception:
    httpx = None

logger = logging.getLogger(__name__)

OPENAI_URL = "https://api.openai.com/v1/chat/completions"
ANTHROPIC_URL = "https://api.anthropic.com/v1/complete"
GOOGLE_PALM_URL = "https://generativelanguage.googleapis.com/v1beta2/models/{model}:generate"


async def call_openai(prompt: str, model: str = "gpt-4o-mini", temperature: float = 0.2, max_tokens: int = 512):
    key = os.getenv("OPENAI_API_KEY")
    if key:
        key = key.strip().strip('"')
    if not key:
        raise RuntimeError("OPENAI_API_KEY not configured")

    # Prefer official openai package when available
    if _HAS_OPENAI_PKG:
        try:
            openai.api_key = key
            # Use ChatCompletions if available
            if hasattr(openai, "ChatCompletion"):
                resp = openai.ChatCompletion.create(model=model, messages=[{"role": "user", "content": prompt}], temperature=temperature, max_tokens=max_tokens)
                # normalize
                text = ""
                if resp and getattr(resp, 'choices', None):
                    ch = resp.choices[0]
                    if hasattr(ch, 'message') and getattr(ch.message, 'content', None):
                        text = ch.message.content
                    elif getattr(ch, 'text', None):
                        text = ch.text
                return {"provider": "openai", "raw": resp, "text": text}
            else:
                # fallback to completions endpoint
                resp = openai.Completion.create(model=model, prompt=prompt, temperature=temperature, max_tokens=max_tokens)
                text = ""
                if resp and getattr(resp, 'choices', None):
                    text = resp.choices[0].text
                return {"provider": "openai", "raw": resp, "text": text}
        except Exception as e:
            logging.exception("OpenAI SDK call failed: %s", e)
            raise

    if httpx is None:
        raise RuntimeError("httpx not available to call OpenAI REST API and openai package not installed")

    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.post(OPENAI_URL, json=body, headers=headers)
        r.raise_for_status()
        data = r.json()
        # For chat-completions, extract text
        text = ""
        if "choices" in data and len(data["choices"]) > 0:
            # Chat-style
            ch = data["choices"][0]
            if "message" in ch and "content" in ch["message"]:
                text = ch["message"]["content"]
            elif "text" in ch:
                text = ch["text"]
        return {"provider": "openai", "raw": data, "text": text}


async def call_anthropic(prompt: str, model: str = "claude-2.1", max_tokens: int = 512, temperature: float = 0.2):
    key = os.getenv("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError("ANTHROPIC_API_KEY not configured")

    headers = {"x-api-key": key, "Content-Type": "application/json"}
    # Anthropic expects `prompt` shaped with system messages; we'll use a simple wrapper
    body = {
        "model": model,
        "prompt": f"\n\nHuman: {prompt}\n\nAssistant:",
        "max_tokens_to_sample": max_tokens,
        "temperature": temperature,
    }

    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.post(ANTHROPIC_URL, json=body, headers=headers)
        r.raise_for_status()
        data = r.json()
        text = data.get("completion") or data.get("text") or ""
        return {"provider": "anthropic", "raw": data, "text": text}


async def call_google_palm(prompt: str, model: str = "text-bison-001", api_key: str | None = None):
    api_key = api_key or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY not configured")

    url = GOOGLE_PALM_URL.format(model=model) + f"?key={api_key}"
    body = {"prompt": {"text": prompt}, "temperature": 0.2, "maxOutputTokens": 512}

    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.post(url, json=body)
        r.raise_for_status()
        data = r.json()
        # response structure: candidates -> output
        text = ""
        if "candidates" in data and len(data["candidates"]) > 0:
            text = data["candidates"][0].get("output", "")
        return {"provider": "google_palm", "raw": data, "text": text}


async def call_deepseek(prompt: str, model: str = "deepseek-coder-33b", temperature: float = 0.7, max_tokens: int = 2048):
    """Call DeepSeek API using server-side env var DEEPSEEK_API_KEY"""
    key = os.getenv("DEEPSEEK_API_KEY")
    if key:
        key = key.strip().strip('"')
    if not key:
        # Fallback: use a lightweight local model (distilgpt2) to keep generation working
        logger.warning("DEEPSEEK_API_KEY not configured — falling back to local distilgpt2 (may be lower quality)")

        async def _fallback():
            try:
                # run blocking transformers pipeline in thread
                def _sync_generate(p):
                    from transformers import pipeline
                    gen = pipeline("text-generation", model="distilgpt2")
                    out = gen(p, max_length=200, do_sample=True, top_k=50)
                    if out and isinstance(out, list) and len(out) > 0 and isinstance(out[0], dict):
                        return out[0].get("generated_text", "")
                    return str(out)

                text = await asyncio.to_thread(_sync_generate, prompt)
                return {"provider": "deepseek-fallback", "raw": None, "text": text}
            except Exception as e:
                logger.exception("DeepSeek fallback generation failed: %s", e)
                raise RuntimeError("DeepSeek fallback failed")

        return await _fallback()

    if httpx is None:
        raise RuntimeError("httpx not available to call DeepSeek API")

    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    body = {
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "model": model,
    }

    async with httpx.AsyncClient(timeout=30) as client:
        try:
            r = await client.post("https://api.deepseek.com/v1/chat/completions", json=body, headers=headers)
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            logger.exception("DeepSeek API request failed: %s — falling back to local generator", e)
            # Run a small local generator synchronously in a thread to avoid blocking the event loop
            def _sync_generate(p):
                from transformers import pipeline
                gen = pipeline("text-generation", model="distilgpt2")
                out = gen(p, max_length=200, do_sample=True, top_k=50)
                if out and isinstance(out, list) and len(out) > 0 and isinstance(out[0], dict):
                    return out[0].get("generated_text", "")
                return str(out)

            text = await asyncio.to_thread(_sync_generate, prompt)
            return {"provider": "deepseek-fallback", "raw": None, "text": text}
        # Extract chat-style content
        text = ""
        try:
            if "choices" in data and len(data["choices"]) > 0:
                ch = data["choices"][0]
                if "message" in ch and "content" in ch["message"]:
                    text = ch["message"]["content"]
                elif "text" in ch:
                    text = ch["text"]
        except Exception:
            text = data.get("response") or data.get("output") or ""

        return {"provider": "deepseek", "raw": data, "text": text}


async def call_blackbox(prompt: str, temperature: float = 0.7, max_tokens: int = 2048):
    """Call Blackbox API using server-side env var BLACKBOX_API_KEY"""
    key = os.getenv("BLACKBOX_API_KEY")
    if key:
        key = key.strip().strip('"')
    if not key:
        raise RuntimeError("BLACKBOX_API_KEY not configured")

    if httpx is None:
        raise RuntimeError("httpx not available to call Blackbox API")

    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    body = {
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
    }

    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.post("https://www.useblackbox.io/chat-request-v4", json=body, headers=headers)
        r.raise_for_status()
        data = r.json()
        # Blackbox may return `response` or similar
        text = data.get("response") or data.get("output") or data.get("text") or ""
        return {"provider": "blackbox", "raw": data, "text": text}


async def call_huggingface(prompt: str, model: str = "gpt2", temperature: float = 0.7, max_tokens: int = 512):
    """Call Hugging Face Inference API using HF_API_TOKEN from env.

    This uses the public Inference API: POST https://api-inference.huggingface.co/models/{model}
    The response format varies by model; normalize to text when possible.
    """
    hf_token = os.getenv("HF_API_TOKEN")
    if hf_token:
        hf_token = hf_token.strip().strip('"')
    if not hf_token:
        raise RuntimeError("HF_API_TOKEN not configured")

    if httpx is None:
        raise RuntimeError("httpx not available to call Hugging Face Inference API")

    url = f"https://api-inference.huggingface.co/models/{model}"
    headers = {"Authorization": f"Bearer {hf_token}", "Content-Type": "application/json"}
    body = {"inputs": prompt, "parameters": {"temperature": temperature, "max_new_tokens": max_tokens}}

    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(url, json=body, headers=headers)
        r.raise_for_status()
        data = r.json()

        # Normalize response
        text = ""
        try:
            if isinstance(data, list) and len(data) > 0:
                # e.g., [{'generated_text': '...'}]
                first = data[0]
                if isinstance(first, dict) and 'generated_text' in first:
                    text = first['generated_text']
                elif isinstance(first, dict) and 'generated_text' not in first:
                    # Try join of values
                    text = ' '.join(str(v) for v in first.values())
            elif isinstance(data, dict):
                if 'generated_text' in data:
                    text = data['generated_text']
                elif 'error' in data:
                    raise RuntimeError(data['error'])
                else:
                    # Last-resort: stringify
                    text = str(data)
            elif isinstance(data, str):
                text = data
        except Exception:
            text = ''

        return {"provider": "huggingface", "raw": data, "text": text}


async def try_providers(prompt: str, providers: list[str], model_hint: str | None = None):
    """Try each provider in order and return first successful response.

    providers: list of provider keys like ['openai','anthropic','google']
    """
    for p in providers:
        try:
            if p == "openai":
                res = await call_openai(prompt, model=(model_hint or "gpt-4o-mini"))
            elif p == "anthropic":
                res = await call_anthropic(prompt, model=(model_hint or "claude-2.1"))
            elif p in ("google", "google_palm"):
                res = await call_google_palm(prompt, model=(model_hint or "text-bison-001"))
            elif p == "deepseek":
                res = await call_deepseek(prompt, model=(model_hint or "deepseek-coder-33b"))
            elif p == "blackbox":
                res = await call_blackbox(prompt)
            elif p in ("huggingface", "hf"):
                res = await call_huggingface(prompt, model=(model_hint or "gpt2"))
            else:
                logger.warning("Unknown provider: %s", p)
                continue
            # Basic result normalization
            if res and res.get("text"):
                return res
        except Exception as e:
            logger.exception("Provider %s failed: %s", p, e)
            continue

    raise RuntimeError("All providers failed or none configured")
