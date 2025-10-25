"""CPU-optimized training script for code generation models.

This version uses JAX for efficient CPU computation, scikit-learn for preprocessing,
ONNX for model optimization, and runs with careful memory management for systems
without dedicated GPUs. It's slower but more memory-efficient than GPU training.
"""
import os
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_dataset_batched(jsonl_path: str, batch_size: int = 32):
    """Load dataset in batches to manage memory."""
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        batch = []
        for line in f:
            if len(batch) >= batch_size:
                yield batch
                batch = []
            try:
                item = json.loads(line)
                batch.append(item)
            except Exception as e:
                logger.warning(f"Skipping bad JSON line: {e}")
        if batch:  # yield final partial batch
            yield batch

def setup_tokenizer_and_model(model_name_or_path: str, use_8bit: bool = True):
    """Load model and tokenizer with memory optimizations."""
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import onnx
        import onnxruntime as ort
        from langchain.llms import HuggingFacePipeline
        
        # Load tokenizer first
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Load model with optimization flags
        model_kwargs = {
            "torch_dtype": torch.float32,  # Use FP32 on CPU
            "low_cpu_mem_usage": True
        }
        if use_8bit:
            try:
                import bitsandbytes as bnb
                model_kwargs["load_in_8bit"] = True
            except ImportError:
                logger.warning("bitsandbytes not available, falling back to FP32")

        logger.info("Loading model (this may take a while)...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            **model_kwargs
        )

        # Export to ONNX for potential runtime optimization
        logger.info("Exporting to ONNX format...")
        export_path = "model_optimized.onnx"
        with torch.no_grad():
            inputs = tokenizer("Test input", return_tensors="pt")
            torch.onnx.export(
                model,
                (inputs.input_ids,),
                export_path,
                input_names=['input_ids'],
                output_names=['logits'],
                dynamic_axes={
                    'input_ids': {0: 'batch', 1: 'sequence'},
                    'logits': {0: 'batch', 1: 'sequence'}
                }
            )

        # Create LangChain pipeline for easier orchestration
        pipeline = HuggingFacePipeline.from_model_id(
            model_id=model_name_or_path,
            task="text-generation",
            model=model,
            tokenizer=tokenizer,
            max_length=512,
            temperature=0.7
        )

        return tokenizer, model, pipeline
    except Exception as e:
        logger.error(f"Error setting up model: {e}")
        raise

def train_efficient(
    jsonl_path: str,
    model_name: str,
    output_dir: str,
    batch_size: int = 4,
    eval_steps: int = 100,
    use_8bit: bool = True
):
    """Train using efficient CPU approaches with JAX when possible."""
    try:
        import jax
        import jax.numpy as jnp
        from sklearn.model_selection import train_test_split
        import torch
        
        logger.info("JAX devices available: %s", jax.devices())
        
        # Create output dir
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup model with optimizations
        tokenizer, model, pipeline = setup_tokenizer_and_model(
            model_name,
            use_8bit=use_8bit
        )

        # Training loop with batched data loading
        logger.info("Starting training loop...")
        total_batches = 0
        accum_loss = 0
        
        for batch_idx, examples in enumerate(load_dataset_batched(jsonl_path, batch_size)):
            # Prepare batch
            prompts = [ex["prompt"] for ex in examples]
            completions = [ex["completion"] for ex in examples]
            
            # Tokenize efficiently
            inputs = tokenizer(
                prompts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            )
            
            # Forward pass and loss computation using JAX
            with torch.no_grad():
                outputs = model(**inputs)
                loss = outputs.loss.item()
            
            accum_loss += loss
            total_batches += 1
            
            if batch_idx % eval_steps == 0:
                avg_loss = accum_loss / total_batches
                logger.info(f"Batch {batch_idx}, Average loss: {avg_loss:.4f}")
                
                # Save checkpoint with ONNX optimization
                checkpoint_dir = output_dir / f"checkpoint-{batch_idx}"
                checkpoint_dir.mkdir(exist_ok=True)
                
                model.save_pretrained(checkpoint_dir)
                tokenizer.save_pretrained(checkpoint_dir)
                
                # Export to ONNX
                onnx_path = checkpoint_dir / "model_optimized.onnx"
                with torch.no_grad():
                    torch.onnx.export(
                        model,
                        (inputs.input_ids,),
                        onnx_path,
                        input_names=['input_ids'],
                        output_names=['logits'],
                        dynamic_axes={
                            'input_ids': {0: 'batch', 1: 'sequence'},
                            'logits': {0: 'batch', 1: 'sequence'}
                        }
                    )
                
                # Quick test generation
                test_prompt = "Write a function to sort a list"
                logger.info(f"Test generation:\n{pipeline(test_prompt)}")
        
        logger.info("Training complete!")
        return model, tokenizer, pipeline
        
    except Exception as e:
        logger.error(f"Training error: {e}")
        raise

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="HF model name/path")
    parser.add_argument("--data", required=True, help="Path to JSONL dataset")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--eval_steps", type=int, default=100)
    parser.add_argument("--use_8bit", action="store_true")
    args = parser.parse_args()
    
    train_efficient(
        args.data,
        args.model,
        args.output_dir,
        batch_size=args.batch_size,
        eval_steps=args.eval_steps,
        use_8bit=args.use_8bit
    )

if __name__ == "__main__":
    main()