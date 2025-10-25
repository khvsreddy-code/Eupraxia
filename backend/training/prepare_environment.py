"""Memory-safe training script that loads libraries incrementally and monitors system resources.
Implements aggressive memory optimizations and safety checks.
"""
import os
import sys
import gc
import json
import logging
import psutil
from typing import Any

# Reduce parallelism for native BLAS/OMP libraries to limit memory and thread usage
# These must be set before importing numpy/torch to take effect
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('NUMEXPR_MAX_THREADS', '1')
os.environ.setdefault('POPLAR_ENGINE_OPTIONS', '{"BLACKLIST_CPU": true}')
from pathlib import Path
from datetime import datetime

# Configure logging
LOG_DIR = Path("training_logs")
LOG_DIR.mkdir(exist_ok=True)
log_file = LOG_DIR / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class MemoryMonitor:
    """Monitor system memory and enforce safety limits."""
    
    def __init__(self, memory_limit_percent=85):
        self.memory_limit = memory_limit_percent
        self.process = psutil.Process(os.getpid())
        self.initial_available = None
        self._log_initial_state()
    
    def _log_initial_state(self):
        mem = psutil.virtual_memory()
        self.initial_available = mem.available
        logger.info(f"Total system memory: {mem.total / (1024**3):.2f} GB")
        logger.info(f"Available memory: {mem.available / (1024**3):.2f} GB")
        logger.info(f"Memory limit set to {self.memory_limit}%")
        
        # Lower memory limit if available memory is too low
        if mem.available / mem.total < 0.2:  # If less than 20% available
            self.memory_limit = 75  # Use more conservative limit
            logger.warning(f"Low memory detected, reducing limit to {self.memory_limit}%")
    
    def check_memory(self):
        """Check if memory usage is safe. Returns (is_safe, usage_percent)."""
        mem = psutil.virtual_memory()
        percent = mem.percent
        is_safe = percent < self.memory_limit
        
        if not is_safe:
            logger.warning(f"Memory usage critical: {percent:.1f}%")
            self.emergency_cleanup()
        
        return is_safe, percent
    
    def emergency_cleanup(self):
        """Aggressive memory cleanup."""
        logger.info("Running emergency memory cleanup...")
        gc.collect()
        try:
            import torch
            if hasattr(torch, 'cuda') and torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        
        # Release any cached memory from libraries
        try:
            import numpy as np
            np.clear_typecodes()
        except (ImportError, AttributeError):
            pass
        
def safe_import(module_name: str):
    """Safely import a module with memory monitoring."""
    logger.info(f"Attempting to import {module_name}...")
    try:
        module = __import__(module_name)
        logger.info(f"Successfully imported {module_name}")
        return module
    except ImportError as e:
        logger.error(f"Failed to import {module_name}: {e}")
        return None

def load_minimal_requirements():
    """Load bare minimum requirements first."""
    
    # Try to free up memory by restarting Python's memory allocator
    import gc
    gc.collect()
    gc.disable()  # Temporarily disable automatic collection
    
    monitor = MemoryMonitor(memory_limit_percent=85)
    
    # Try importing core libraries one by one
    imports = {}
    
    # Clear any pre-existing numpy/torch imports
    for mod in list(sys.modules.keys()):
        if mod.startswith(('numpy', 'torch')):
            del sys.modules[mod]
    
    gc.enable()  # Re-enable automatic collection
    gc.collect()
    
    # Phase 1: Essential numeric libraries
    imports['numpy'] = safe_import('numpy')
    is_safe, mem_usage = monitor.check_memory()
    if not is_safe:
        logger.error("Memory usage too high after numpy import!")
        # Try one last aggressive cleanup
        gc.collect()
        is_safe, mem_usage = monitor.check_memory()
        if not is_safe:
            return imports, monitor  # Continue anyway to see what happens
        
    # Phase 2: PyTorch with minimal features
    os.environ['PYTORCH_JIT'] = '0'  # Disable JIT to save memory
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
    
    imports['torch'] = safe_import('torch')
    if imports['torch']:
        # Configure PyTorch for minimal memory usage
        torch = imports['torch']
        torch.set_grad_enabled(False)  # Start with gradients off
        if hasattr(torch, 'cuda') and torch.cuda.is_available():
            torch.cuda.set_per_process_memory_fraction(0.7)  # Limit GPU memory
    
    monitor.check_memory()
    
    return imports, monitor

def verify_system_compatibility():
    """Check if the system can handle minimal ML operations."""
    logger.info("Verifying system compatibility...")
    
    imports, monitor = load_minimal_requirements()
    if not imports:
        return False
        
    if not imports.get('torch'):
        logger.error("PyTorch import failed - system may be incompatible")
        return False
    
    # Try a minimal tensor operation
    try:
        torch = imports['torch']
        # Small tensor to test basic operations
        x = torch.randn(100, 100)
        del x
        torch.cuda.empty_cache() if hasattr(torch, 'cuda') else None
        logger.info("Basic PyTorch operations successful")
    except Exception as e:
        logger.error(f"Failed basic PyTorch test: {e}")
        return False
    
    is_safe, mem_usage = monitor.check_memory()
    logger.info(f"Current memory usage: {mem_usage:.1f}%")
    
    return is_safe

def prepare_training_environment(model_name: str):
    """Prepare training environment with minimal memory footprint."""
    logger.info(f"Preparing training environment for {model_name}...")
    
    monitor = MemoryMonitor(memory_limit_percent=85)
    
    # Clean memory before imports
    gc.collect()
    
    # Phase 1: Import essential libraries
    imports = {}
    essential_libs = [
        'numpy', 'torch', 'transformers', 'accelerate', 
        'peft'  # Removed bitsandbytes as it's not available for Windows
    ]
    
    for lib in essential_libs:
        imports[lib] = safe_import(lib)
        is_safe, mem_usage = monitor.check_memory()
        if not is_safe:
            logger.error(f"Memory limit exceeded while importing {lib}")
            return None
        
    if not all(imports.values()):
        logger.error("Failed to import all required libraries")
        return None
    
    # Phase 2: Configure for minimal memory usage
    torch = imports['torch']
    transformers = imports['transformers']
    
    # Aggressive memory settings
    torch.set_grad_enabled(False)
    if hasattr(torch, 'cuda') and torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.7)
    
    # Configure transformers for minimal memory
    transformers.utils.logging.set_verbosity_error()
    return imports, monitor

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify-only", action="store_true", 
                       help="Only verify system compatibility")
    parser.add_argument("--model", default="NTQAI/Nxcode-CQ-7B-orpo",
                       help="Model to prepare for (default: smallest compatible model)")
    args = parser.parse_args()
    
    if args.verify_only:
        is_compatible = verify_system_compatibility()
        logger.info(f"System compatibility check: {'PASSED' if is_compatible else 'FAILED'}")
        sys.exit(0 if is_compatible else 1)
    
    # Prepare environment for training
    imports, monitor = prepare_training_environment(args.model)
    if not imports:
        logger.error("Failed to prepare training environment")
        sys.exit(1)
    
    logger.info("Environment prepared successfully")
    is_safe, mem_usage = monitor.check_memory()
    logger.info(f"Final memory usage: {mem_usage:.1f}%")

if __name__ == "__main__":
    main()