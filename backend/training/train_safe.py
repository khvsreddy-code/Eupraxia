"""Memory-aware example trainer.

This script is intentionally minimal and safe: it defaults to a tiny synthetic dataset
when no `--data` is provided so you can smoke-test the flow on CPU or a small GPU.

Features:
- Optional 8-bit loading via bitsandbytes
- Optional LoRA adapters (PEFT)
- Gradient checkpointing toggle
"""
import argparse
import os
from pathlib import Path

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--data", default=None)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--per_device_train_batch_size", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--use_8bit", action="store_true")
    p.add_argument("--use_lora", action="store_true")
    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--gradient_checkpointing", action="store_true")
    p.add_argument("--deepspeed_config", default=None, help="Path to DeepSpeed JSON config for ZeRO/offload")
    return p.parse_args()


def main():
    args = parse_args()

    # Lazy imports
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
    import datasets

    # Optional imports
    if args.use_8bit:
        try:
            import bitsandbytes as bnb  # noqa: F401
        except Exception:
            print("bitsandbytes requested but not installed. Install training/requirements_training.txt")
            raise

    if args.use_lora:
        try:
            from peft import get_peft_model, LoraConfig, TaskType
        except Exception:
            print("peft/LoRA requested but not installed. Install training/requirements_training.txt")
            raise

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    # Tiny synthetic dataset when none provided
    if not args.data:
        ds = datasets.Dataset.from_dict({
            "prompt": ["print(1)", "def add(a,b): return a+b"],
            "completion": ["1", "def add(a,b):\n    return a + b"]
        })
    else:
        ds = datasets.load_dataset('json', data_files={'train': args.data}, split='train')

    def preprocess(example):
        text = example.get('prompt', '') + "\n\n" + example.get('completion', '')
        tok = tokenizer(text, truncation=True, max_length=512)
        tok['labels'] = tok['input_ids'].copy()
        return tok

    tok_ds = ds.map(preprocess, remove_columns=ds.column_names, batched=False)

    # Model loading options
    load_kwargs = {"torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32}
    if args.use_8bit:
        load_kwargs["load_in_8bit"] = True
    # Use low_cpu_mem_usage where available
    load_kwargs["low_cpu_mem_usage"] = True

    model = AutoModelForCausalLM.from_pretrained(args.model, **load_kwargs)

    if args.use_lora:
        peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, r=args.lora_r, lora_alpha=16, lora_dropout=0.05)
        model = get_peft_model(model, peft_config)

    if args.gradient_checkpointing and hasattr(model, 'gradient_checkpointing_enable'):
        try:
            model.gradient_checkpointing_enable()
        except Exception:
            pass

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        fp16=torch.cuda.is_available(),
        save_total_limit=1,
        logging_steps=5,
        remove_unused_columns=False,
        deepspeed=args.deepspeed_config if args.deepspeed_config else None,
    )

    trainer = Trainer(model=model, args=training_args, train_dataset=tok_ds, tokenizer=tokenizer)

    trainer.train()
    trainer.save_model(args.output_dir)


if __name__ == '__main__':
    main()
