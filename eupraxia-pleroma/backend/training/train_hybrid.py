"""Hybrid training approach that combines CPU and disk offloading for memory efficiency.
Designed specifically for systems with limited RAM and no dedicated GPU.
"""
import os
import sys
import logging
from pathlib import Path
import json
import tempfile
import shutil
from typing import Dict, Any, Optional
from dataclasses import dataclass

from prepare_environment import MemoryMonitor, safe_import, load_minimal_requirements

@dataclass
class TrainingConfig:
    """Configuration for hybrid training."""
    model_name: str
    sequence_length: int = 128
    batch_size: int = 1
    gradient_accumulation_steps: int = 16
    disk_offload: bool = True
    cpu_offload: bool = True
    memory_limit: Optional[int] = None  # No hard limit, just monitoring
    use_amp: bool = True  # Use automatic mixed precision
    disk_cache_dir: Optional[str] = None
    dynamic_batching: bool = True  # Dynamically adjust batch size based on memory

class HybridTrainer:
    """Trains with hybrid CPU-disk offloading for optimal memory usage."""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.imports: Dict[str, Any] = {}
        self.temp_dir = None
        
        # Set up temp directory before memory monitor to ensure disk space
        if config.disk_offload:
            self.temp_dir = Path(tempfile.mkdtemp(prefix="hybrid_trainer_"))
            logging.info(f"Created temporary directory for disk offloading: {self.temp_dir}")
        
        # Set up memory monitoring with adaptive thresholds
        self.monitor = MemoryMonitor(warning_threshold=85, critical_threshold=95)
        
        # Pre-configure environment variables for optimal memory usage
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
        os.environ["TORCH_USE_CUDA_DSA"] = "0"  # Disable CUDA device synchronous allocator
        os.environ["PYTORCH_NO_CUDA_MEMORY_CACHING"] = "1"
        
    def cleanup(self):
        """Clean up temporary resources."""
        if self.temp_dir and self.temp_dir.exists():
            logging.info(f"Cleaning up temporary directory: {self.temp_dir}")
            try:
                shutil.rmtree(self.temp_dir)
            except Exception as e:
                logging.error(f"Failed to clean up temporary directory: {e}")
    
    def setup_training(self) -> bool:
        """Set up the training environment with minimal memory usage."""
        try:
            # Pre-setup cleanup
            import gc
            gc.collect()
            
            # Clear any existing module imports that might hold memory
            for mod in list(sys.modules.keys()):
                if mod.startswith(('torch', 'numpy', 'transformers', 'accelerate')):
                    sys.modules.pop(mod, None)
            
            # Phase 1: Load minimal requirements with reduced memory target
            self.imports, self.monitor = load_minimal_requirements()
            if not self.imports.get('torch'):
                logging.error("Failed to load PyTorch")
                return False
            
            # Configure PyTorch for minimal memory
            torch = self.imports['torch']
            torch.set_grad_enabled(False)  # Enable only during training
            
            # Phase 2: Load other libraries
            transformers = safe_import('transformers')
            accelerate = safe_import('accelerate')
            peft = safe_import('peft')
            
            if not all([transformers, accelerate, peft]):
                logging.error("Failed to load required libraries")
                return False
            
            # Configure transformers for disk offloading if enabled
            if self.config.disk_offload:
                os.environ['TRANSFORMERS_CACHE'] = str(self.temp_dir)
                transformers.utils.set_cache_dir(str(self.temp_dir))
            
            # Configure for CPU offloading
            if self.config.cpu_offload:
                self.setup_cpu_offload(accelerate)
            
            return True
            
        except Exception as e:
            logging.error(f"Failed to setup training: {e}")
            self.cleanup()
            return False
    
    def setup_cpu_offload(self, accelerate):
        """Configure CPU offloading settings."""
        from accelerate import Accelerator
        from accelerate.utils import set_seed
        
        set_seed(42)  # For reproducibility
        
        # Configure accelerator with CPU offload
        self.accelerator = Accelerator(
            cpu=True,  # Force CPU training
            mixed_precision="fp16" if self.config.use_amp else "no",
            gradient_accumulation_steps=self.config.gradient_accumulation_steps
        )
    
    def load_model(self):
        """Load model with memory optimizations."""
        transformers = self.imports.get('transformers')
        if not transformers:
            raise RuntimeError("transformers not loaded")
            
        # Configure model loading for minimal memory
        config = transformers.AutoConfig.from_pretrained(
            self.config.model_name,
            low_cpu_mem_usage=True,
            torch_dtype='auto'
        )
        
        # Load tokenizer first to verify download works
        try:
            tokenizer = transformers.AutoTokenizer.from_pretrained(
                self.config.model_name,
                cache_dir=self.temp_dir if self.config.disk_offload else None
            )
        except Exception as e:
            logging.error(f"Failed to load tokenizer: {e}")
            return None, None
            
        # Now try loading model with memory optimizations
        try:
            # Use device_map='auto' for automatic memory optimization
            model = transformers.AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                config=config,
                cache_dir=self.temp_dir if self.config.disk_offload else None,
                device_map='cpu',  # Force CPU
                low_cpu_mem_usage=True,
                torch_dtype='auto'
            )
        except Exception as e:
            logging.error(f"Failed to load model: {e}")
            return None, None
            
        return model, tokenizer
    
    def train(self, train_data_path: str, eval_data_path: Optional[str] = None):
        """Run training with memory-efficient settings."""
        if not self.setup_training():
            return False
            
        try:
            model, tokenizer = self.load_model()
            if not model or not tokenizer:
                return False
                
            # TODO: Implement actual training loop with:
            # - Gradient checkpointing
            # - Gradient accumulation
            # - Mixed precision training
            # - Regular memory monitoring
            # - Checkpoint saving with disk offloading
            
            return True
            
        except Exception as e:
            logging.error(f"Training failed: {e}")
            return False
            
        finally:
            self.cleanup()

def main():
    """Main entry point with error handling."""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True,
                       help="Name or path of the model to train")
    parser.add_argument("--train-data", required=True,
                       help="Path to training data")
    parser.add_argument("--eval-data",
                       help="Optional path to evaluation data")
    parser.add_argument("--sequence-length", type=int, default=128,
                       help="Maximum sequence length")
    parser.add_argument("--batch-size", type=int, default=1,
                       help="Training batch size")
    parser.add_argument("--grad-accum", type=int, default=16,
                       help="Gradient accumulation steps")
    parser.add_argument("--no-disk-offload", action="store_true",
                       help="Disable disk offloading")
    parser.add_argument("--no-cpu-offload", action="store_true",
                       help="Disable CPU offloading")
    parser.add_argument("--memory-limit", type=int, default=75,
                       help="Memory usage limit in percent")
    parser.add_argument("--dry-run", action="store_true",
                       help="Only setup environment and exit before loading model weights")
    args = parser.parse_args()
    
    config = TrainingConfig(
        model_name=args.model,
        sequence_length=args.sequence_length,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        disk_offload=not args.no_disk_offload,
        cpu_offload=not args.no_cpu_offload,
        memory_limit=args.memory_limit
    )
    
    trainer = HybridTrainer(config)
    # If dry-run requested, only perform setup and do not load model weights
    if args.dry_run:
        ok = trainer.setup_training()
        if ok:
            print("Dry-run setup completed successfully")
            sys.exit(0)
        else:
            print("Dry-run setup failed")
            sys.exit(2)

    success = trainer.train(args.train_data, args.eval_data)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()