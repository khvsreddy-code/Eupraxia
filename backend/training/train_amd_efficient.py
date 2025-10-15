"""Optimized training script for AMD Ryzen 5 7535HS with integrated Radeon Graphics.
Uses ROCm backend when possible and careful memory management.
"""
import os
import json
import logging
from pathlib import Path
from typing import Optional
import psutil
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_system_memory():
    """Get available system memory in GB."""
    mem = psutil.virtual_memory()
    return mem.available / (1024 ** 3)

def check_rocm_available():
    """Check if ROCm (AMD GPU acceleration) is available."""
    try:
        import torch
        return hasattr(torch, 'has_rocm') and torch.has_rocm
    except ImportError:
        return False

def setup_memory_efficient_training(total_gpu_mem_gb: float = 2.0):
    """Configure memory settings based on available system resources."""
    available_mem = get_system_memory()
    logger.info(f"Available system memory: {available_mem:.2f} GB")
    
    # Reserve some memory for system
    training_mem = min(available_mem - 2.0, 4.0)  # Leave 2GB for system, max 4GB for training
    
    settings = {
        "batch_size": 1,  # Start conservative
        "max_length": 256,  # Reduced from 512
        "gradient_accumulation_steps": 4,
        "use_8bit": True,
        "use_4bit": available_mem < 6,  # Use 4-bit if very memory constrained
        "max_memory_split": f"{training_mem:.1f}GB"
    }
    
    if check_rocm_available():
        logger.info("ROCm acceleration available")
        settings["device"] = "cuda"  # ROCm uses CUDA API
        settings["max_memory_split"] = f"{total_gpu_mem_gb:.1f}GB"
    else:
        logger.info("Using CPU only mode")
        settings["device"] = "cpu"
    
    return settings

def load_dataset_batched(jsonl_path: str, max_length: int = 256, skip_long: bool = True):
    """Memory-efficient dataset loading."""
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        while True:
            try:
                line = next(f)
                item = json.loads(line)
                
                # Skip very long sequences if requested
                if skip_long and (
                    len(item["prompt"]) + len(item["completion"]) > max_length * 4
                ):
                    continue
                    
                yield item
            except StopIteration:
                break
            except Exception as e:
                logger.warning(f"Skipping problematic line: {e}")
                continue

def train_efficient(
    jsonl_path: str,
    model_name: str,
    output_dir: str,
    use_rocm: bool = True,
    eval_steps: int = 50
):
    """Train with optimizations for AMD Ryzen + Radeon."""
    try:
        import torch
        from transformers import (
            AutoTokenizer, AutoModelForCausalLM,
            TrainingArguments, Trainer
        )
        from peft import prepare_model_for_kbit_training
        from accelerate import load_checkpoint_and_dispatch
        
        # Get system-specific settings
        settings = setup_memory_efficient_training(
            total_gpu_mem_gb=2.0  # Conservative estimate for integrated GPU
        )
        
        logger.info(f"Training settings: {settings}")
        
        # Initialize tokenizer with padding
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        # Load model with extreme memory optimization
        model_kwargs = {
            "low_cpu_mem_usage": True,
            "torch_dtype": torch.float16,
        }
        
        if settings["use_4bit"]:
            logger.info("Using 4-bit quantization")
            try:
                import bitsandbytes as bnb
                model_kwargs["load_in_4bit"] = True
                model_kwargs["bnb_4bit_compute_dtype"] = torch.float16
            except ImportError:
                logger.warning("bitsandbytes not available, falling back to 8-bit")
                model_kwargs["load_in_8bit"] = True
        elif settings["use_8bit"]:
            model_kwargs["load_in_8bit"] = True
            
        logger.info("Loading model (this may take a while)...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **model_kwargs
        )
        
        # Prepare for training
        model = prepare_model_for_kbit_training(model)
        
        # Set up checkpointing to save memory
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        training_args = TrainingArguments(
            output_dir=str(output_dir),
            per_device_train_batch_size=settings["batch_size"],
            gradient_accumulation_steps=settings["gradient_accumulation_steps"],
            learning_rate=1e-5,  # Very conservative
            max_steps=100,  # Start small
            save_steps=50,
            eval_steps=eval_steps,
            logging_steps=10,
            max_grad_norm=0.3,  # Prevent extreme gradients
            # Memory optimizations
            fp16=True,
            dataloader_pin_memory=False,  # Less memory pressure
            gradient_checkpointing=True,
            # Push everything possible to disk
            offload_state_dict=True,
            optim="adamw_bnb_8bit"
        )
        
        class MemoryEfficientTrainer(Trainer):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.max_length = settings["max_length"]
            
            def _prepare_inputs(self, *args, **kwargs):
                inputs = super()._prepare_inputs(*args, **kwargs)
                # Ensure we don't exceed memory
                if "input_ids" in inputs:
                    inputs["input_ids"] = inputs["input_ids"][:, :self.max_length]
                if "attention_mask" in inputs:
                    inputs["attention_mask"] = inputs["attention_mask"][:, :self.max_length]
                return inputs
        
        trainer = MemoryEfficientTrainer(
            model=model,
            args=training_args,
            train_dataset=load_dataset_batched(jsonl_path, settings["max_length"]),
            tokenizer=tokenizer
        )
        
        logger.info("Starting training loop...")
        trainer.train()
        
        # Save final model with safe cleanup
        logger.info("Saving model...")
        trainer.save_model(str(output_dir / "final"))
        del trainer
        del model
        torch.cuda.empty_cache()
        
        logger.info("Training complete!")
        
    except Exception as e:
        logger.error(f"Training error: {e}")
        raise

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="HF model name/path")
    parser.add_argument("--data", required=True, help="Path to JSONL dataset")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--eval_steps", type=int, default=50)
    parser.add_argument("--no_rocm", action="store_true", help="Disable ROCm/GPU acceleration")
    args = parser.parse_args()
    
    train_efficient(
        args.data,
        args.model,
        args.output_dir,
        use_rocm=not args.no_rocm,
        eval_steps=args.eval_steps
    )

if __name__ == "__main__":
    main()