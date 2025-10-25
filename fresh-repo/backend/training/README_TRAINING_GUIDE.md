
Memory-safe training guide
==========================

This short guide explains practical, low-risk ways to "teach" (fine-tune) a transformer-based AI model while minimizing GPU/host memory usage and keeping operations safe.

Principles
----------

- Prefer parameter-efficient fine-tuning (PEFT) techniques such as LoRA to avoid having to train full model weights.

- Use 8-bit or 4-bit quantized weights (bitsandbytes) to reduce GPU memory for very large models.

- Use gradient checkpointing to trade computation for memory.

- Use device offloading or Deepspeed ZeRO to spread memory across CPU/GPU or multiple GPUs.

- Always sandbox any evaluation that executes generated code.

Quick checklist (safe defaults)
-------------------------------

- Use LoRA (PEFT) for model adaptation whenever possible.

- For models > 7B, enable 8-bit loading with bitsandbytes and `device_map="auto"`.

- Enable `gradient_checkpointing` if your GPU memory is the bottleneck.

- Use small batch sizes (1 or 2) and accumulate gradients if needed.

- Keep a copy of your original model weights and only persist PEFT adapters (they are small).

How to run the example trainer
------------------------------

Open PowerShell and from the `backend` folder create/activate a virtualenv and install deps:

```powershell
python -m venv .venv
. .venv\Scripts\Activate.ps1
pip install -r training/requirements_training.txt
```

Run a quick smoke test (this uses a small built-in dataset):

```powershell
python training\train_safe.py --model distilgpt2 --output_dir ./runs/distilgpt2-smoke --epochs 1 --per_device_train_batch_size 2
```

Run with memory optimizations for a larger HF model:

```powershell
python training\train_safe.py --model THE/HF-MODEL --data ./my_dataset.jsonl --output_dir ./runs/my-finetune --use_8bit --use_lora --lora_r 8 --gradient_checkpointing
```

Options overview
----------------

- `--use_8bit`: load weights in 8-bit using bitsandbytes (requires GPU + bitsandbytes)

- `--use_lora`: enable LoRA adapters (PEFT) to only train small additional parameters

- `--gradient_checkpointing`: reduce activation memory during forward passes

- `--fp16` / `--bf16`: mixed precision. bf16 requires compatible GPU and PyTorch build.

Safety notes
------------

- Do not run untrusted training data or evaluation code without sandboxing. Generated code should be executed inside a container or VM with strict resource limits.

- Respect model licenses and API terms when using/pretraining base models.

- Monitor GPU memory with `nvidia-smi` and CPU/memory usage during runs.

Next steps and hardening
------------------------

- If you need production-scale training, consider DeepSpeed + ZeRO(offload) or Multi-node Accelerate configurations.

- Add evaluation harnesses that run unit tests or static linters on generated code rather than executing it directly.

- Integrate logging and checkpoints to allow safe rollback in case of runaway training.

References
----------

- Hugging Face Accelerate: [https://huggingface.co/docs/accelerate](https://huggingface.co/docs/accelerate)

- PEFT / LoRA: [https://huggingface.co/docs/peft](https://huggingface.co/docs/peft)

- bitsandbytes: [https://github.com/facebookresearch/bitsandbytes](https://github.com/facebookresearch/bitsandbytes)
