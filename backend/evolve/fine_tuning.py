"""
Multi-task fine-tuning system for continuous evolution.
Implements:
- LoRA adaptation for efficient training
- Multi-task learning
- Automated feedback integration
- Memory-efficient training
"""

import os
import logging
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import time
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from peft import LoraConfig, TaskType, get_peft_model
import pytorch_lightning as pl
from datasets import load_dataset
import wandb
from transformers import AutoModelForCausalLM, AutoTokenizer
import ray
from ray import tune

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TrainingConfig:
    """Configuration for fine-tuning."""
    model_name: str = "meta-llama/Llama-2-7b-hf"
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    learning_rate: float = 2e-5
    batch_size: int = 4
    max_steps: int = 1000
    eval_steps: int = 100
    save_steps: int = 200
    memory_limit_gb: float = 7.5

class EvolvingLLM(pl.LightningModule):
    """LLM with continuous evolution capabilities."""
    
    def __init__(self, config: TrainingConfig):
        super().__init__()
        self.config = config
        self.memory_tracker = MemoryTracker(config.memory_limit_gb)
        
        # Initialize base model with quantization
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        # Add LoRA adapter
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=["q_proj", "v_proj"]
        )
        self.model = get_peft_model(self.model, peft_config)
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        
    def forward(self, input_ids, attention_mask=None, labels=None):
        """Forward pass with memory tracking."""
        self.memory_tracker.check()
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        return outputs
    
    def training_step(self, batch, batch_idx):
        """Training step with metrics logging."""
        outputs = self(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"]
        )
        
        # Log metrics
        self.log("train_loss", outputs.loss)
        self.memory_tracker.log_metrics()
        
        return outputs.loss
    
    def configure_optimizers(self):
        """Configure optimizer with learning rate."""
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.config.learning_rate
        )
    
    def save_evolved_model(self, path: str):
        """Save evolved model weights."""
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

class MemoryTracker:
    """Track and optimize memory usage."""
    
    def __init__(self, limit_gb: float):
        self.limit_bytes = limit_gb * 1024 * 1024 * 1024
        self.peak_usage = 0
        
    def check(self):
        """Check current memory usage."""
        current = torch.cuda.max_memory_allocated() \
            if torch.cuda.is_available() \
            else torch.tensor([0])
            
        self.peak_usage = max(self.peak_usage, current)
        
        if current > self.limit_bytes:
            torch.cuda.empty_cache()
            logger.warning("Memory limit reached, optimizing...")
    
    def log_metrics(self):
        """Log memory metrics."""
        wandb.log({
            "memory_used_gb": self.peak_usage / 1024**3,
            "memory_limit_gb": self.limit_bytes / 1024**3
        })

@ray.remote
class EvolutionManager:
    """Manage the evolution process across tasks."""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.model = None
        self.metrics = []
        
    def evolve_on_task(self, task_data: Dict[str, Any]):
        """Evolve model on a specific task."""
        # Initialize model if needed
        if self.model is None:
            self.model = EvolvingLLM(self.config)
        
        # Train on task
        trainer = pl.Trainer(
            max_steps=self.config.max_steps,
            val_check_interval=self.config.eval_steps,
            logger=True
        )
        
        trainer.fit(self.model, task_data)
        
        # Track metrics
        self.metrics.append(trainer.callback_metrics)
        
        return {
            "task_completed": True,
            "metrics": trainer.callback_metrics
        }
    
    def get_evolution_status(self) -> Dict[str, Any]:
        """Get current evolution status and metrics."""
        return {
            "total_tasks": len(self.metrics),
            "average_loss": sum(m["train_loss"] for m in self.metrics) / len(self.metrics) \
                if self.metrics else None,
            "peak_memory": max(m.get("memory_used_gb", 0) for m in self.metrics) \
                if self.metrics else 0
        }

def setup_evolution_pipeline(config: TrainingConfig):
    """Setup the complete evolution pipeline."""
    # Initialize Ray
    ray.init(
        runtime_env={
            "pip": ["torch", "transformers", "peft", "datasets"]
        }
    )
    
    # Create evolution manager
    manager = EvolutionManager.remote(config)
    
    # Setup wandb logging
    wandb.init(
        project="llm-evolution",
        config=vars(config)
    )
    
    return manager

def run_evolution_cycle(
    manager: "ray.actor.ActorHandle",
    task_data: Dict[str, Any]
) -> Dict[str, Any]:
    """Run one evolution cycle."""
    try:
        result = ray.get(manager.evolve_on_task.remote(task_data))
        status = ray.get(manager.get_evolution_status.remote())
        
        wandb.log({
            "evolution_cycle": status["total_tasks"],
            **status
        })
        
        return {
            "success": True,
            "metrics": result["metrics"],
            "status": status
        }
    
    except Exception as e:
        logger.error(f"Evolution cycle failed: {e}")
        return {
            "success": False,
            "error": str(e)
        }