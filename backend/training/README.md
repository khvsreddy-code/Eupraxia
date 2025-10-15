Training guide for finetuning code models with Project_CodeNet

Overview
--------
This directory contains tooling to prepare the IBM Project_CodeNet dataset and finetune Hugging Face code models using PEFT/LoRA and Accelerate. The provided scripts are scaffolding and will require a GPU machine with sufficient VRAM and disk space. Some of the models you listed (Qwen2.5 32B, OpenCodeInterpreter 33B) are very large and may require multi-GPU or specialized setups.

Hardware recommendations
----------------------
- 1x A100 80GB or multi-GPU setups are recommended for 32B+ models.
- For smaller experiments, a single 24-48GB GPU may be sufficient when using LoRA or 8-bit optimizations (bitsandbytes).
- Plenty of disk space: Project_CodeNet can be large when cloned/expanded.

Install dependencies
--------------------
Create a Python virtualenv and install training deps (this can take a while):

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -r training/requirements.txt
```

Prepare Project_CodeNet dataset
------------------------------
Run the prepare script to clone the repo and produce a JSONL file suitable for fine-tuning:

```bash
python training/prepare_codenet.py --target ./Project_CodeNet --out ./codenet_dataset.jsonl
```

Start training (example)
------------------------
Use the training script. For large models you should use `accelerate launch` with an appropriate config.

Single-node example (small models / experiments):

```bash
accelerate launch training/train_finetune.py --model Qwen/CodeQwen1.5-7B-Chat --data ./codenet_dataset.jsonl --output_dir ./fine_tuned_codeqwen --per_device_train_batch_size 1 --epochs 1
```

For very large models (Qwen2.5 32B), you must use multi-GPU with an accelerate config and consider using bitsandbytes + 8-bit loading + DeepSpeed.

Notes & cautions
----------------
- The scripts are scaffolding: you will likely need to adapt tokenization, instruction format, and prompt templates to get the best results for code generation.
- Training costs can be high. Consider using LoRA/PEFT to reduce cost. Evaluate on held-out tasks.
- Respect model licensing and usage policies for each HF model you use.

Next steps
----------
- Add prompt templates and few-shot examples for code synthesis and bug-fixing tasks.
- Build an evaluation harness to run unit tests or static checks on generated code.
- Optionally build a safety/sandboxed executor to run generated code in Docker for dynamic validation.
