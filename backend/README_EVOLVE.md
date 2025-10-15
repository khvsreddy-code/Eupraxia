Eupraxia — Evolution & Fine-tune Guide
=====================================

This document explains the lightweight self-evolution prototype in `backend/scripts/` and how to run optional fine-tuning in a container.

Quickstart (safe, no heavy ML deps)
----------------------------------
1. Start the backend API (set `PROXY_API_KEY` in your environment):

   ```powershell
   $env:PROXY_API_KEY='your-secret'
   .\.venv\Scripts\Activate.ps1
   .\.venv\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
   ```

2. Generate a small synthetic dataset locally:

   ```powershell
   $env:PROXY_API_KEY='your-secret'
   .\.venv\Scripts\python.exe backend\scripts\evolution_cycle.py
   ```

3. Prepare a fine-tune dataset (creates `backend/evolution_data/ft_dataset.jsonl`):

   ```powershell
   .\.venv\Scripts\python.exe backend\scripts\fine_tune_prototype.py
   ```


Optional: Fine-tune in container (recommended for heavy ML)
-----------------------------------------------------------
1. Build the ML image (Linux host recommended):

   ```bash
   docker build --target ml -t eupraxia-ml:latest .
   ```

2. Run a shell in the container and perform training using your preferred scripts (PEFT/Trainer). Example steps inside container:

   ```bash
   pip install --upgrade pip
   # run or create your training script that uses PEFT / bitsandbytes
   python train_lora.py --data /app/backend/evolution_data/ft_dataset.jsonl
   ```

Eupraxia — Evolution & Fine-tune Guide
=====================================

This document explains the lightweight self-evolution prototype in `backend/scripts/` and how to run optional fine-tuning in a container.

Quickstart (safe, no heavy ML deps)
----------------------------------

1. Start the backend API (set `PROXY_API_KEY` in your environment):

   ```powershell
   $env:PROXY_API_KEY='your-secret'
   .\.venv\Scripts\Activate.ps1
   .\.venv\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
   ```

2. Generate a small synthetic dataset locally:

   ```powershell
   $env:PROXY_API_KEY='your-secret'
   .\.venv\Scripts\python.exe backend\scripts\evolution_cycle.py
   ```

3. Prepare a fine-tune dataset (creates `backend/evolution_data/ft_dataset.jsonl`):

   ```powershell
   .\.venv\Scripts\python.exe backend\scripts\fine_tune_prototype.py
   ```


Optional: Fine-tune in container (recommended for heavy ML)
-----------------------------------------------------------

1. Build the ML image (Linux host recommended):

   ```bash
   docker build --target ml -t eupraxia-ml:latest .
   ```

2. Run a shell in the container and perform training using your preferred scripts (PEFT/Trainer). Example steps inside container:

   ```bash
   pip install --upgrade pip
   # run or create your training script that uses PEFT / bitsandbytes
   python train_lora.py --data /app/backend/evolution_data/ft_dataset.jsonl
   ```


Scheduling examples
-------------------

- Windows Task Scheduler: create a task that runs `python .\backend\scripts\evolution_cycle.py` daily with the proper environment variables.
- cron (Linux): add a crontab entry that runs the script inside a container or virtualenv daily.


Security reminder
-----------------

Never commit API tokens. Revoke any tokens that were exposed publicly and store secrets in environment variables or a secret manager.
