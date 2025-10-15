# RAG & Fine-tuning Scripts

This folder contains lightweight scripts to generate RAG data and fine-tune a small model using PEFT/LoRA. These are intentionally minimal and geared for CPU/low-memory experimentation.

Files
- `scripts/rag_data_gen.py` — builds embeddings, FAISS index in memory, and generates a small `rag_train.jsonl` using a tiny generator.
- `scripts/fine_tune_peft.py` — example fine-tune script (PEFT/LoRA) using a very small model. Adjust `MODEL_NAME` environment variable as needed.
- `scripts/rag_infer.py` — simple retrieval + generation demo that uses the fine-tuned model.

Usage (local, CPU)
1. Create virtualenv and install dependencies:
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   # or install minimal list
   pip install transformers datasets peft sentence-transformers faiss-cpu
   ```

2. Generate RAG dataset:
   ```powershell
   python backend/scripts/rag_data_gen.py
   ```

3. Fine-tune (may be slow on CPU; recommended for small experiments only):
   ```powershell
   python backend/scripts/fine_tune_peft.py
   ```

4. Run RAG inference demo:
   ```powershell
   python backend/scripts/rag_infer.py
   ```

Notes
- These are example scripts. For production or larger training, use proper data chunking, batching, and consider GPU/cloud training.
- Use new tokens stored in `HF_API_TOKEN` environment variable if interacting with Hugging Face hosted endpoints.
