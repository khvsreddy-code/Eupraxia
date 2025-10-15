"""Phased training script with aggressive memory optimizations.
Trains the model in stages, monitoring system resources carefully.
"""
import os
import sys
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import json

# Import our environment preparation
from prepare_environment import MemoryMonitor, safe_import, load_minimal_requirements

class PhasedTrainer:
    """Trains a model in phases with aggressive memory management."""
    
    def __init__(self, 
                 model_name: str,
                 data_path: str,
                 output_dir: str,
                 memory_limit: int = 85,
                 sequence_length: int = 128):
        self.model_name = model_name
        self.data_path = Path(data_path)
        self.output_dir = Path(output_dir)
        self.sequence_length = sequence_length
        self.monitor = MemoryMonitor(memory_limit_percent=memory_limit)
        self.imports: Dict[str, Any] = {}
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def setup_phase1(self) -> bool:
        """Phase 1: Load minimal requirements and verify system."""
        self.imports, self.monitor = load_minimal_requirements()
        if not self.imports.get('torch'):
            logging.error("Failed to load PyTorch")
            return False
            
        # Configure for absolute minimal memory usage
        torch = self.imports['torch']
        torch.set_grad_enabled(False)
        if hasattr(torch, 'cuda') and torch.cuda.is_available():
            torch.cuda.set_per_process_memory_fraction(0.6)
        
        return True
        
    def setup_phase2(self) -> bool:
        """Phase 2: Load model-specific libraries with minimal settings."""
        try:
            # Import transformers with minimal memory
            transformers = safe_import('transformers')
            if not transformers:
                return False
            self.imports['transformers'] = transformers
            
            # Configure transformers for minimal memory
            transformers.utils.logging.set_verbosity_error()
            
            # Load tokenizer first (smaller memory footprint)
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(
                self.model_name,
                use_fast=False  # Slower but less memory
            )
            
            is_safe, mem_usage = self.monitor.check_memory()
            if not is_safe:
                logging.error(f"Memory usage too high after tokenizer: {mem_usage}%")
                return False
            
            return True
            
        except Exception as e:
            logging.error(f"Phase 2 setup failed: {e}")
            return False
            
    def setup_phase3(self) -> bool:
        """Phase 3: Load model with extreme memory optimizations."""
        try:
            transformers = self.imports['transformers']
            torch = self.imports['torch']
            
            # Try to import 4-bit training support
            bnb = safe_import('bitsandbytes')
            self.imports['bitsandbytes'] = bnb
            
            # Load model with minimal settings
            model_kwargs = {
                "low_cpu_mem_usage": True,
                "torch_dtype": torch.float16,
            }
            
            if bnb:
                logging.info("Using 4-bit quantization")
                model_kwargs.update({
                    "load_in_4bit": True,
                    "bnb_4bit_compute_dtype": torch.float16,
                    "bnb_4bit_quant_type": "nf4",
                    "bnb_4bit_use_double_quant": True,
                })
            
            self.model = transformers.AutoModelForCausalLM.from_pretrained(
                self.model_name,
                **model_kwargs
            )
            
            is_safe, mem_usage = self.monitor.check_memory()
            if not is_safe:
                logging.error(f"Memory usage too high after model load: {mem_usage}%")
                return False
                
            return True
            
        except Exception as e:
            logging.error(f"Phase 3 setup failed: {e}")
            return False
            
    def train_phase1(self, num_steps: int = 10) -> bool:
        """Initial training phase with minimal batch size."""
        try:
            transformers = self.imports['transformers']
            torch = self.imports['torch']
            
            # Enable gradient checkpointing
            self.model.gradient_checkpointing_enable()
            
            # Training args with extreme memory optimization
            training_args = transformers.TrainingArguments(
                output_dir=str(self.output_dir),
                per_device_train_batch_size=1,
                gradient_accumulation_steps=4,
                max_steps=num_steps,
                learning_rate=1e-5,
                logging_steps=1,
                save_steps=5,
                # Memory optimizations
                fp16=True,
                dataloader_pin_memory=False,
                gradient_checkpointing=True,
                optim="adamw_bnb_8bit",
                # Offload everything possible
                deepspeed={
                    "zero_optimization": {
                        "stage": 2,
                        "offload_optimizer": {
                            "device": "cpu",
                            "pin_memory": False
                        },
                        "offload_param": {
                            "device": "cpu",
                            "pin_memory": False
                        }
                    }
                }
            )
            
            def data_generator():
                """Generate data in tiny batches."""
                with open(self.data_path, 'r') as f:
                    for line in f:
                        try:
                            item = json.loads(line)
                            yield {
                                "prompt": item["prompt"][:self.sequence_length],
                                "completion": item["completion"][:self.sequence_length]
                            }
                        except Exception as e:
                            logging.warning(f"Skipping bad line: {e}")
                            continue
            
            # Custom trainer with memory monitoring
            class MemoryAwareTrainer(transformers.Trainer):
                def __init__(self, monitor, *args, **kwargs):
                    super().__init__(*args, **kwargs)
                    self.monitor = monitor
                
                def training_step(self, *args, **kwargs):
                    is_safe, mem_usage = self.monitor.check_memory()
                    if not is_safe:
                        raise RuntimeError(f"Memory usage too high: {mem_usage}%")
                    return super().training_step(*args, **kwargs)
            
            trainer = MemoryAwareTrainer(
                monitor=self.monitor,
                model=self.model,
                args=training_args,
                train_dataset=data_generator(),
                tokenizer=self.tokenizer
            )
            
            trainer.train()
            return True
            
        except Exception as e:
            logging.error(f"Training phase 1 failed: {e}")
            return False
    
    def run_all_phases(self, steps_per_phase: int = 10):
        """Run all training phases with safety checks."""
        try:
            # Phase 1: Basic setup
            logging.info("Starting Phase 1: Basic setup")
            if not self.setup_phase1():
                raise RuntimeError("Phase 1 failed")
            
            # Phase 2: Load tokenizer
            logging.info("Starting Phase 2: Loading tokenizer")
            if not self.setup_phase2():
                raise RuntimeError("Phase 2 failed")
            
            # Phase 3: Load model
            logging.info("Starting Phase 3: Loading model")
            if not self.setup_phase3():
                raise RuntimeError("Phase 3 failed")
            
            # Phase 4: Initial training
            logging.info("Starting Phase 4: Initial training")
            if not self.train_phase1(steps_per_phase):
                raise RuntimeError("Training phase failed")
            
            logging.info("All phases completed successfully!")
            return True
            
        except Exception as e:
            logging.error(f"Training pipeline failed: {e}")
            return False

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="NTQAI/Nxcode-CQ-7B-orpo")
    parser.add_argument("--data", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--memory-limit", type=int, default=85,
                       help="Memory usage limit in percent")
    parser.add_argument("--sequence-length", type=int, default=128,
                       help="Max sequence length (shorter = less memory)")
    parser.add_argument("--steps", type=int, default=10,
                       help="Steps per training phase")
    args = parser.parse_args()
    
    trainer = PhasedTrainer(
        model_name=args.model,
        data_path=args.data,
        output_dir=args.output_dir,
        memory_limit=args.memory_limit,
        sequence_length=args.sequence_length
    )
    
    success = trainer.run_all_phases(steps_per_phase=args.steps)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()