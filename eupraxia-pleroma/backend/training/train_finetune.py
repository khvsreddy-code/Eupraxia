"""Fine-tune a HF code model using PEFT (LoRA) and Accelerate.

Usage (example):
  python train_finetune.py --model qwen/Qwen2.5-Coder-32B-Instruct --data ./codenet_dataset.jsonl --output_dir ./fine_tuned

NOTE: Large models require powerful GPUs or multi-GPU setups. This script uses PEFT/LoRA to reduce memory.
"""
import argparse
import os
from pathlib import Path

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="Hugging Face model id (e.g., Qwen/Qwen2.5-Coder-32B-Instruct)")
    p.add_argument("--data", required=True, help="Path to JSONL dataset with prompt/completion pairs")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--per_device_train_batch_size", type=int, default=1)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--max_length", type=int, default=1024)
    return p.parse_args()


def main():
    args = parse_args()

    # Lazy imports to fail later if not installed
    from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
    import datasets
    from peft import get_peft_model, LoraConfig, TaskType

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
    # Some tokenizers require special settings
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    # Load dataset
    data = datasets.load_dataset('json', data_files={'train': args.data}, split='train')

    def preprocess(example):
        prompt = example.get('prompt', '')
        completion = example.get('completion', '')
        # Combine into single sequence with sentinel tokens
        text = prompt + '\n\n' + completion
        out = tokenizer(text, truncation=True, max_length=args.max_length)
        out['labels'] = out['input_ids'].copy()
        return out

    tok = data.map(preprocess, remove_columns=data.column_names, batched=False)

    # Prepare model with PEFT
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype='auto', low_cpu_mem_usage=True)

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )

    model = get_peft_model(model, peft_config)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        fp16=True,
        save_total_limit=2,
        logging_steps=10,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tok,
        tokenizer=tokenizer,
    )

    trainer.train()
    trainer.save_model(args.output_dir)


if __name__ == '__main__':
    main()
