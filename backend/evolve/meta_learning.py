"""
Advanced meta-learning system for rapid task adaptation.
Implements:
- Meta-model for quick learning
- Task-specific adaptation
- Performance tracking
- Memory-efficient processing
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
from dataclasses import dataclass
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import learn2learn as l2l
import wandb

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class MetaConfig:
    """Configuration for meta-learning."""
    inner_lr: float = 0.01
    meta_lr: float = 0.001
    adaptation_steps: int = 5
    num_tasks: int = 100
    shots_per_task: int = 4
    memory_limit_gb: float = 2.0

class TaskBatch(Dataset):
    """Dataset for meta-learning tasks."""
    def __init__(self, samples: List[Dict[str, str]], tokenizer):
        self.samples = samples
        self.tokenizer = tokenizer
        
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        sample = self.samples[idx]
        inputs = self.tokenizer(
            sample["input"],
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        
        labels = self.tokenizer(
            sample["output"],
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        
        return {
            "input_ids": inputs["input_ids"].squeeze(),
            "attention_mask": inputs["attention_mask"].squeeze(),
            "labels": labels["input_ids"].squeeze()
        }

class MetaLearner:
    """Advanced meta-learning system."""
    
    def __init__(
        self,
        base_model: str,
        config: Optional[MetaConfig] = None,
        device: str = "cpu"
    ):
        self.config = config or MetaConfig()
        self.device = device
        
        # Initialize model and tokenizer
        self.model = AutoModelForCausalLM.from_pretrained(base_model)
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        
        # Setup meta-learning
        self.meta_model = l2l.algorithms.MAML(
            self.model,
            lr=self.config.inner_lr,
            first_order=True
        )
        
        # Optimizer for meta-update
        self.meta_optimizer = torch.optim.Adam(
            self.meta_model.parameters(),
            lr=self.config.meta_lr
        )
        
        # Memory tracking
        self._setup_memory_tracking()
        
    def _setup_memory_tracking(self):
        """Setup memory monitoring."""
        self.peak_memory = 0
        self.memory_warnings = 0
        
    def _check_memory(self) -> bool:
        """Check if memory usage is within limits."""
        if torch.cuda.is_available():
            current = torch.cuda.max_memory_allocated() / 1024**3
        else:
            current = self.model.get_memory_footprint() / 1024**3
            
        self.peak_memory = max(self.peak_memory, current)
        
        if current > self.config.memory_limit_gb:
            self.memory_warnings += 1
            logger.warning(
                f"Memory usage ({current:.1f}GB) exceeds limit "
                f"({self.config.memory_limit_gb}GB)"
            )
            return False
        return True
        
    def _create_task_batch(
        self,
        task_data: List[Dict[str, str]]
    ) -> DataLoader:
        """Create a task batch for meta-learning."""
        dataset = TaskBatch(task_data, self.tokenizer)
        return DataLoader(
            dataset,
            batch_size=self.config.shots_per_task,
            shuffle=True
        )
        
    def adapt_to_task(
        self,
        task_data: List[Dict[str, str]]
    ) -> None:
        """Quick adaptation to a new task."""
        try:
            # Check memory
            if not self._check_memory():
                self._optimize_memory()
            
            # Prepare task batch
            task_loader = self._create_task_batch(task_data)
            learner = self.meta_model.clone()
            
            # Adaptation loop
            for step in range(self.config.adaptation_steps):
                batch = next(iter(task_loader))
                
                # Move to device
                batch = {
                    k: v.to(self.device)
                    for k, v in batch.items()
                }
                
                # Forward pass
                outputs = learner(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"]
                )
                
                # Compute loss and adapt
                loss = outputs.loss
                learner.adapt(loss)
                
                # Log metrics
                if wandb.run is not None:
                    wandb.log({
                        "adaptation_loss": loss.item(),
                        "adaptation_step": step
                    })
                    
            # Update meta-model
            self.meta_model = learner
            
        except Exception as e:
            logger.error(f"Task adaptation failed: {e}")
            
    def meta_train(
        self,
        train_tasks: List[List[Dict[str, str]]],
        valid_tasks: Optional[List[List[Dict[str, str]]]] = None
    ) -> Dict[str, float]:
        """Train meta-learner on multiple tasks."""
        try:
            metrics = {
                "train_loss": [],
                "valid_loss": [],
                "memory_usage": []
            }
            
            for task_idx, task_data in enumerate(train_tasks):
                if not self._check_memory():
                    break
                    
                # Clone model for this task
                learner = self.meta_model.clone()
                task_loader = self._create_task_batch(task_data)
                
                # Inner loop - task adaptation
                for step in range(self.config.adaptation_steps):
                    batch = next(iter(task_loader))
                    batch = {k: v.to(self.device) for k, v in batch.items()}
                    
                    outputs = learner(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        labels=batch["labels"]
                    )
                    
                    loss = outputs.loss
                    learner.adapt(loss)
                    metrics["train_loss"].append(loss.item())
                
                # Outer loop - meta update
                if valid_tasks:
                    valid_data = valid_tasks[task_idx % len(valid_tasks)]
                    valid_loader = self._create_task_batch(valid_data)
                    
                    with torch.no_grad():
                        batch = next(iter(valid_loader))
                        batch = {k: v.to(self.device) for k, v in batch.items()}
                        
                        outputs = learner(
                            input_ids=batch["input_ids"],
                            attention_mask=batch["attention_mask"],
                            labels=batch["labels"]
                        )
                        
                        valid_loss = outputs.loss
                        metrics["valid_loss"].append(valid_loss.item())
                
                # Meta-update
                self.meta_optimizer.zero_grad()
                loss.backward()
                self.meta_optimizer.step()
                
                # Track memory
                if torch.cuda.is_available():
                    current_memory = torch.cuda.max_memory_allocated() / 1024**3
                else:
                    current_memory = self.model.get_memory_footprint() / 1024**3
                metrics["memory_usage"].append(current_memory)
                
                # Log progress
                if wandb.run is not None:
                    wandb.log({
                        "meta_train_loss": np.mean(metrics["train_loss"][-10:]),
                        "meta_valid_loss": np.mean(metrics["valid_loss"][-10:])
                        if metrics["valid_loss"] else 0,
                        "memory_gb": current_memory,
                        "task_index": task_idx
                    })
                    
            return metrics
            
        except Exception as e:
            logger.error(f"Meta-training failed: {e}")
            return metrics
            
    def _optimize_memory(self):
        """Optimize memory usage."""
        try:
            # Clear unused tensors
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Move less used components to CPU
            for name, param in self.model.named_parameters():
                if "attention" not in name:  # Keep attention in GPU
                    param.data = param.data.cpu()
                    
            # Log memory state
            if wandb.run is not None:
                wandb.log({
                    "peak_memory_gb": self.peak_memory,
                    "memory_warnings": self.memory_warnings
                })
                
        except Exception as e:
            logger.error(f"Memory optimization failed: {e}")
            
    def generate(
        self,
        prompt: str,
        task_context: Optional[List[Dict[str, str]]] = None,
        max_length: int = 512
    ) -> str:
        """Generate response with optional task adaptation."""
        try:
            # Adapt to task if context provided
            if task_context:
                self.adapt_to_task(task_context)
            
            # Prepare input
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt"
            ).to(self.device)
            
            # Generate
            with torch.no_grad():
                outputs = self.meta_model.generate(
                    **inputs,
                    max_length=max_length,
                    temperature=0.7,
                    num_return_sequences=1
                )
                
            return self.tokenizer.decode(outputs[0])
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return ""
            
    def save(self, path: str):
        """Save meta-learned model."""
        try:
            save_dir = Path(path)
            save_dir.mkdir(parents=True, exist_ok=True)
            
            # Save model
            self.meta_model.save_pretrained(save_dir / "model")
            self.tokenizer.save_pretrained(save_dir / "tokenizer")
            
            # Save config
            with open(save_dir / "meta_config.json", "w") as f:
                json.dump(vars(self.config), f, indent=2)
                
        except Exception as e:
            logger.error(f"Save failed: {e}")
            
    def load(self, path: str):
        """Load meta-learned model."""
        try:
            load_dir = Path(path)
            
            # Load config
            with open(load_dir / "meta_config.json", "r") as f:
                config = json.load(f)
                self.config = MetaConfig(**config)
            
            # Load model
            self.meta_model = l2l.algorithms.MAML(
                AutoModelForCausalLM.from_pretrained(load_dir / "model"),
                lr=self.config.inner_lr,
                first_order=True
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                load_dir / "tokenizer"
            )
            
        except Exception as e:
            logger.error(f"Load failed: {e}")
            
    def get_metrics(self) -> Dict[str, float]:
        """Get current performance metrics."""
        return {
            "peak_memory_gb": self.peak_memory,
            "memory_warnings": self.memory_warnings,
            "adaptation_steps": self.config.adaptation_steps,
            "tasks_processed": self.config.num_tasks
        }
