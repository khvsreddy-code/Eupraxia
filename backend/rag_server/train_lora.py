"""Train a LoRA adapter on domain data.
Designed for cloud GPUs but can run locally on small datasets.

Usage (after activating venv):
1. Set hyperparams in config.yaml
2. Prepare dataset:
   python train_lora.py prepare-data --input data/my_docs.jsonl

3. Train (prefer cloud GPU):
   python train_lora.py train --config config.yaml
   
4. Merge & export:
   python train_lora.py export --adapter results/my_adapter
"""
import os
import sys
import yaml
import json
import torch
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    PeftModel,
    prepare_model_for_kbit_training
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import Dataset
import bitsandbytes as bnb
from tqdm import tqdm
from accelerate import Accelerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TrainingConfig:
    # Model settings
    base_model: str = "meta-llama/Llama-2-7b-hf"
    tokenizer: Optional[str] = None  # if different from base_model
    
    # LoRA hyperparams
    lora_r: int = 8
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: List[str] = None  # if None, use default for architecture
    
    # Training params
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    num_epochs: int = 3
    learning_rate: float = 2e-4
    max_steps: Optional[int] = None
    warmup_ratio: float = 0.03
    
    # Data & output
    train_data: str = "data/train.jsonl"
    val_data: Optional[str] = "data/val.jsonl"
    output_dir: str = "results/lora_adapter"
    
    # Resource limits
    max_gpu_memory: Optional[str] = None  # e.g. "24GiB"
    
    @classmethod
    def from_yaml(cls, path: str) -> "TrainingConfig":
        with open(path) as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)

def prepare_dataset(input_path: str, train_path: str, val_path: Optional[str] = None, val_split: float = 0.1):
    """Convert raw text/docs into instruction format and split train/val."""
    logger.info(f"Preparing dataset from {input_path}")
    
    # Load raw data (assumes JSONL with 'text' field)
    data = []
    with open(input_path) as f:
        for line in f:
            data.append(json.loads(line))
    
    # Convert to instruction format
    formatted = []
    for item in data:
        text = item['text']
        # Split into chunks and format as instructions
        chunks = text.split('\n\n')
        for chunk in chunks:
            if len(chunk.strip()) < 10:  # Skip very short chunks
                continue
            formatted.append({
                'instruction': f"Complete or expand upon this text:\n{chunk[:100]}...",
                'input': '',
                'output': chunk
            })
    
    # Train/val split
    if val_split > 0 and val_path:
        split_idx = int(len(formatted) * (1 - val_split))
        train_data = formatted[:split_idx]
        val_data = formatted[split_idx:]
        
        with open(val_path, 'w') as f:
            for item in val_data:
                f.write(json.dumps(item) + '\n')
    else:
        train_data = formatted
    
    with open(train_path, 'w') as f:
        for item in train_data:
            f.write(json.dumps(item) + '\n')
    
    logger.info(f"Saved {len(train_data)} training examples")
    if val_path:
        logger.info(f"Saved {len(val_data)} validation examples")

def load_dataset(path: str, tokenizer) -> Dataset:
    """Load and tokenize dataset from JSONL."""
    data = []
    with open(path) as f:
        for line in f:
            data.append(json.loads(line))
    
    def format_prompt(item):
        return f"### Instruction:\n{item['instruction']}\n\n### Input:\n{item['input']}\n\n### Response:\n{item['output']}"
    
    texts = [format_prompt(item) for item in data]
    
    def tokenize(text):
        return tokenizer(
            text,
            truncation=True,
            max_length=2048,
            padding=False,
            return_tensors=None
        )
    
    tokenized = Dataset.from_dict({
        'input_ids': [tokenize(text)['input_ids'] for text in texts],
        'attention_mask': [tokenize(text)['attention_mask'] for text in texts]
    })
    
    return tokenized

def create_bnb_config():
    """Config for loading model in 4-bit with nested quantization."""
    return bnb.BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )

def train(config_path: str):
    """Main training loop."""
    config = TrainingConfig.from_yaml(config_path)
    logger.info(f"Starting training with config:\n{yaml.dump(config.__dict__)}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.tokenizer or config.base_model,
        trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load base model in 4-bit
    logger.info("Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        config.base_model,
        quantization_config=create_bnb_config(),
        device_map="auto",
        trust_remote_code=True,
        max_memory={"0": config.max_gpu_memory} if config.max_gpu_memory else None
    )
    
    # Prepare for k-bit training
    model = prepare_model_for_kbit_training(model)
    
    # Add LoRA adapter
    logger.info("Adding LoRA adapter...")
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=config.target_modules
    )
    model = get_peft_model(model, lora_config)
    
    # Load & prepare datasets
    logger.info("Loading datasets...")
    train_dataset = load_dataset(config.train_data, tokenizer)
    if config.val_data:
        val_dataset = load_dataset(config.val_data, tokenizer)
    
    # Training args
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        max_steps=config.max_steps,
        warmup_ratio=config.warmup_ratio,
        logging_steps=1,
        save_strategy="steps",
        save_steps=50,
        evaluation_strategy="steps" if config.val_data else "no",
        eval_steps=50 if config.val_data else None,
        load_best_model_at_end=True if config.val_data else False,
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset if config.val_data else None,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
    )
    
    # Train
    logger.info("Starting training...")
    trainer.train()
    
    # Save adapter
    logger.info(f"Saving adapter to {config.output_dir}")
    trainer.save_model()

def export_model(adapter_path: str, output_path: Optional[str] = None):
    """Merge LoRA adapter with base model and export."""
    if not output_path:
        output_path = str(Path(adapter_path).parent / "merged_model")
    
    logger.info(f"Loading base model and adapter from {adapter_path}")
    
    # Load adapter config to get base model name
    with open(os.path.join(adapter_path, "adapter_config.json")) as f:
        adapter_config = json.load(f)
    base_model_name = adapter_config["base_model_name_or_path"]
    
    # Load base model and adapter
    model = AutoModelForCausalLM.from_pretrained(base_model_name)
    model = PeftModel.from_pretrained(model, adapter_path)
    
    # Merge and save
    logger.info("Merging adapter with base model...")
    merged_model = model.merge_and_unload()
    
    logger.info(f"Saving merged model to {output_path}")
    merged_model.save_pretrained(output_path)

def main():
    if len(sys.argv) < 2:
        print("""
Usage: python train_lora.py <command> [args]
Commands:
  prepare-data --input <input_jsonl> [--train <train_out>] [--val <val_out>] [--split <ratio>]
  train --config <config.yaml>
  export --adapter <adapter_path> [--output <output_path>]
""")
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == "prepare-data":
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--input", required=True)
        parser.add_argument("--train", default="data/train.jsonl")
        parser.add_argument("--val", default="data/val.jsonl")
        parser.add_argument("--split", type=float, default=0.1)
        args = parser.parse_args(sys.argv[2:])
        
        prepare_dataset(args.input, args.train, args.val, args.split)
    
    elif command == "train":
        if len(sys.argv) != 4 or sys.argv[2] != "--config":
            print("Usage: python train_lora.py train --config <config.yaml>")
            sys.exit(1)
        train(sys.argv[3])
    
    elif command == "export":
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--adapter", required=True)
        parser.add_argument("--output")
        args = parser.parse_args(sys.argv[2:])
        
        export_model(args.adapter, args.output)
    
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)

if __name__ == "__main__":
    main()