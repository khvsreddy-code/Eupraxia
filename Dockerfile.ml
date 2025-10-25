FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime

# Set working directory
WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    curl \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Ray dependencies
RUN pip install --no-cache-dir \
    "ray[default,serve]" \
    "ray[train]"

# Install ML dependencies
RUN pip install --no-cache-dir \
    transformers \
    peft \
    accelerate \
    bitsandbytes \
    sentence-transformers \
    faiss-cpu \
    onnxruntime \
    optimum[onnxruntime] \
    tritonclient \
    deepspeed \
    wandb \
    datasets \
    ragas \
    rank_bm25 \
    cohere \
    litellm

# Install monitoring tools
RUN pip install --no-cache-dir \
    prometheus_client \
    grafana-dashboard

# Copy application code
COPY ./backend /app/backend
COPY .env.example /app/.env.example

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV MEMORY_LIMIT_GB=7.5
ENV VECTOR_STORE_LIMIT_GB=3.0
ENV ENABLE_GPU=false
ENV MODEL_PROVIDER=groq
ENV EMBEDDING_MODEL=BAAI/bge-large-en-v1.5

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

# Default command
CMD ["python", "-m", "backend.main"]
