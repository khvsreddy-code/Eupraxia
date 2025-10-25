"""
Advanced knowledge distillation system for model compression.
Implements:
- Teacher-student distillation
- Loss matching
- Attention transfer
- Efficient compression
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
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer
)
import wandb

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DistillConfig:
    """Configuration for knowledge distillation."""
    temperature: float = 2.0
    alpha: float = 0.5  # Weight for distillation loss
    hidden_layers: int = 4  # Number of layers in student
    batch_size: int = 8
    max_epochs: int = 10
    memory_limit_gb: float = 2.0

class DistillationDataset(Dataset):
    """Dataset for knowledge distillation."""
    def __init__(
        self,
        texts: List[str],
        teacher_outputs: List[torch.Tensor],
        tokenizer: PreTrainedTokenizer
    ):
        self.texts = texts
        self.teacher_outputs = teacher_outputs
        self.tokenizer = tokenizer
        
    def __len__(self):
        return len(self.texts)
        
    def __getitem__(self, idx):
        text = self.texts[idx]
        teacher_output = self.teacher_outputs[idx]
        
        inputs = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        
        return {
            "input_ids": inputs["input_ids"].squeeze(),
            "attention_mask": inputs["attention_mask"].squeeze(),
            "teacher_logits": teacher_output
        }

class DistillationLoss(nn.Module):
    """Custom loss for knowledge distillation."""
    def __init__(self, temperature: float = 2.0, alpha: float = 0.5):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        
    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute distillation loss."""
        # Soften probability distributions
        soft_targets = F.softmax(teacher_logits / self.temperature, dim=-1)
        soft_preds = F.log_softmax(student_logits / self.temperature, dim=-1)
        
        # Distillation loss
        distill_loss = F.kl_div(
            soft_preds,
            soft_targets,
            reduction="batchmean"
        ) * (self.temperature ** 2)
        
        # Hard loss if labels provided
        if labels is not None:
            hard_loss = F.cross_entropy(student_logits, labels)
            return self.alpha * distill_loss + (1 - self.alpha) * hard_loss
            
        return distill_loss

class KnowledgeDistiller:
    """Advanced knowledge distillation system."""
    
    def __init__(
        self,
        teacher_model: str,
        config: Optional[DistillConfig] = None,
        device: str = "cpu"
    ):
        self.config = config or DistillConfig()
        self.device = device
        
        # Initialize teacher model
        self.teacher = AutoModelForCausalLM.from_pretrained(teacher_model)
        self.teacher.eval()  # Set to evaluation mode
        
        # Create smaller student model
        self.student = self._create_student_model(teacher_model)
        
        # Setup tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(teacher_model)
        
        # Loss function
        self.criterion = DistillationLoss(
            temperature=self.config.temperature,
            alpha=self.config.alpha
        )
        
        # Memory tracking
        self._setup_memory_tracking()
        
    def _create_student_model(self, base_model: str) -> PreTrainedModel:
        """Create smaller student model."""
        try:
            # Load config and modify for smaller model
            config = AutoModelForCausalLM.from_pretrained(
                base_model
            ).config
            
            # Reduce model size
            config.num_hidden_layers = self.config.hidden_layers
            config.intermediate_size //= 2
            
            # Initialize student
            student = AutoModelForCausalLM.from_config(config)
            
            # Quantize student model
            student = torch.quantization.quantize_dynamic(
                student,
                {torch.nn.Linear},
                dtype=torch.qint8
            )
            
            return student
            
        except Exception as e:
            logger.error(f"Student creation failed: {e}")
            raise
            
    def _setup_memory_tracking(self):
        """Setup memory monitoring."""
        self.peak_memory = 0
        self.memory_warnings = 0
        
    def _check_memory(self) -> bool:
        """Check if memory usage is within limits."""
        if torch.cuda.is_available():
            current = torch.cuda.max_memory_allocated() / 1024**3
        else:
            current = (
                self.teacher.get_memory_footprint() +
                self.student.get_memory_footprint()
            ) / 1024**3
            
        self.peak_memory = max(self.peak_memory, current)
        
        if current > self.config.memory_limit_gb:
            self.memory_warnings += 1
            logger.warning(
                f"Memory usage ({current:.1f}GB) exceeds limit "
                f"({self.config.memory_limit_gb}GB)"
            )
            return False
        return True
        
    def _get_teacher_outputs(
        self,
        texts: List[str]
    ) -> List[torch.Tensor]:
        """Get teacher model predictions."""
        outputs = []
        
        try:
            with torch.no_grad():
                for text in texts:
                    # Tokenize
                    inputs = self.tokenizer(
                        text,
                        return_tensors="pt"
                    ).to(self.device)
                    
                    # Get teacher predictions
                    teacher_outputs = self.teacher(**inputs)
                    outputs.append(teacher_outputs.logits)
                    
            return outputs
            
        except Exception as e:
            logger.error(f"Teacher prediction failed: {e}")
            return []
            
    def distill(
        self,
        texts: List[str],
        epochs: Optional[int] = None
    ) -> Dict[str, float]:
        """Perform knowledge distillation."""
        try:
            if not self._check_memory():
                self._optimize_memory()
            
            # Get teacher outputs
            teacher_outputs = self._get_teacher_outputs(texts)
            if not teacher_outputs:
                return {}
            
            # Create dataset
            dataset = DistillationDataset(
                texts,
                teacher_outputs,
                self.tokenizer
            )
            dataloader = DataLoader(
                dataset,
                batch_size=self.config.batch_size,
                shuffle=True
            )
            
            # Training setup
            optimizer = torch.optim.AdamW(self.student.parameters())
            num_epochs = epochs or self.config.max_epochs
            
            metrics = {
                "train_loss": [],
                "memory_usage": []
            }
            
            # Training loop
            for epoch in range(num_epochs):
                epoch_loss = 0.0
                
                for batch in dataloader:
                    if not self._check_memory():
                        break
                    
                    # Move to device
                    batch = {
                        k: v.to(self.device)
                        for k, v in batch.items()
                    }
                    
                    # Forward pass
                    student_outputs = self.student(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"]
                    )
                    
                    # Compute loss
                    loss = self.criterion(
                        student_outputs.logits,
                        batch["teacher_logits"]
                    )
                    
                    # Backward pass
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    epoch_loss += loss.item()
                    
                # Log metrics
                avg_loss = epoch_loss / len(dataloader)
                metrics["train_loss"].append(avg_loss)
                
                if torch.cuda.is_available():
                    current_memory = torch.cuda.max_memory_allocated() / 1024**3
                else:
                    current_memory = (
                        self.teacher.get_memory_footprint() +
                        self.student.get_memory_footprint()
                    ) / 1024**3
                metrics["memory_usage"].append(current_memory)
                
                if wandb.run is not None:
                    wandb.log({
                        "distill_loss": avg_loss,
                        "memory_gb": current_memory,
                        "epoch": epoch
                    })
                    
            return metrics
            
        except Exception as e:
            logger.error(f"Distillation failed: {e}")
            return {}
            
    def _optimize_memory(self):
        """Optimize memory usage."""
        try:
            # Clear cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Move teacher to CPU if needed
            if self.peak_memory > self.config.memory_limit_gb:
                self.teacher.cpu()
                
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
        max_length: int = 512,
        use_teacher: bool = False
    ) -> str:
        """Generate text using student or teacher model."""
        try:
            # Prepare input
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt"
            ).to(self.device)
            
            # Select model
            model = self.teacher if use_teacher else self.student
            
            # Generate
            with torch.no_grad():
                outputs = model.generate(
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
        """Save distilled model."""
        try:
            save_dir = Path(path)
            save_dir.mkdir(parents=True, exist_ok=True)
            
            # Save student model
            self.student.save_pretrained(save_dir / "student")
            self.tokenizer.save_pretrained(save_dir / "tokenizer")
            
            # Save config
            with open(save_dir / "distill_config.json", "w") as f:
                json.dump(vars(self.config), f, indent=2)
                
        except Exception as e:
            logger.error(f"Save failed: {e}")
            
    def load(self, path: str):
        """Load distilled model."""
        try:
            load_dir = Path(path)
            
            # Load config
            with open(load_dir / "distill_config.json", "r") as f:
                config = json.load(f)
                self.config = DistillConfig(**config)
            
            # Load student
            self.student = AutoModelForCausalLM.from_pretrained(
                load_dir / "student"
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                load_dir / "tokenizer"
            )
            
        except Exception as e:
            logger.error(f"Load failed: {e}")
            
    def get_metrics(self) -> Dict[str, float]:
        """Get performance metrics."""
        return {
            "peak_memory_gb": self.peak_memory,
            "memory_warnings": self.memory_warnings,
            "compression_ratio": (
                self.teacher.get_memory_footprint() /
                self.student.get_memory_footprint()
            )
        }
