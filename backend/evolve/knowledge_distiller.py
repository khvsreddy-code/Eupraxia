"""
Knowledge Distillation and Model Compression System
Implements advanced knowledge distillation techniques with dynamic pruning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from transformers import PreTrainedModel, PreTrainedTokenizer
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from dataclasses import dataclass

@dataclass
class DistillationConfig:
    temperature: float = 2.0
    alpha: float = 0.5  # Weight for distillation loss
    beta: float = 0.5   # Weight for task-specific loss
    pruning_threshold: float = 0.1
    batch_size: int = 16
    num_epochs: int = 5
    learning_rate: float = 1e-4

class KnowledgeDistiller:
    def __init__(
        self,
        teacher_model: PreTrainedModel,
        student_model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        config: DistillationConfig
    ):
        self.teacher = teacher_model
        self.student = student_model
        self.tokenizer = tokenizer
        self.config = config
        self.optimizer = Adam(student_model.parameters(), lr=config.learning_rate)
        
        # Freeze teacher model
        for param in self.teacher.parameters():
            param.requires_grad = False
            
        self.teacher.eval()

    def compute_distillation_loss(
        self,
        teacher_logits: torch.Tensor,
        student_logits: torch.Tensor,
        temperature: float
    ) -> torch.Tensor:
        """
        Compute the distillation loss between teacher and student logits
        using temperature scaling for smoother probability distribution
        """
        scaled_teacher_logits = teacher_logits / temperature
        scaled_student_logits = student_logits / temperature
        
        return F.kl_div(
            F.log_softmax(scaled_student_logits, dim=-1),
            F.softmax(scaled_teacher_logits, dim=-1),
            reduction='batchmean'
        ) * (temperature ** 2)

    def compute_task_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """Compute the task-specific loss"""
        return F.cross_entropy(logits, labels)

    def train_step(
        self,
        batch: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """Perform one training step of knowledge distillation"""
        self.optimizer.zero_grad()
        
        # Forward pass through both models
        with torch.no_grad():
            teacher_outputs = self.teacher(**batch)
            
        student_outputs = self.student(**batch)
        
        # Compute losses
        distillation_loss = self.compute_distillation_loss(
            teacher_outputs.logits,
            student_outputs.logits,
            self.config.temperature
        )
        
        task_loss = self.compute_task_loss(
            student_outputs.logits,
            batch['labels']
        )
        
        # Combined loss
        total_loss = (
            self.config.alpha * distillation_loss +
            self.config.beta * task_loss
        )
        
        # Backward pass and optimization
        total_loss.backward()
        self.optimizer.step()
        
        return {
            'distillation_loss': distillation_loss.item(),
            'task_loss': task_loss.item(),
            'total_loss': total_loss.item()
        }

    def evaluate(
        self,
        eval_dataloader: torch.utils.data.DataLoader
    ) -> Dict[str, float]:
        """Evaluate the student model"""
        self.student.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in eval_dataloader:
                outputs = self.student(**batch)
                loss = self.compute_task_loss(outputs.logits, batch['labels'])
                total_loss += loss.item()
                
                predictions = torch.argmax(outputs.logits, dim=-1)
                correct += (predictions == batch['labels']).sum().item()
                total += batch['labels'].size(0)
                
        accuracy = correct / total
        avg_loss = total_loss / len(eval_dataloader)
        
        self.student.train()
        return {'eval_loss': avg_loss, 'eval_accuracy': accuracy}

    def prune_weights(self, threshold: Optional[float] = None):
        """Prune model weights below threshold"""
        if threshold is None:
            threshold = self.config.pruning_threshold
            
        with torch.no_grad():
            for name, param in self.student.named_parameters():
                if 'weight' in name:
                    mask = torch.abs(param.data) > threshold
                    param.data *= mask

    def quantize_model(self, bits: int = 8):
        """Quantize the student model to reduce size"""
        # Simple quantization for demonstration
        self.student.half()  # Convert to FP16
        
        # More advanced quantization could be implemented here
        # using techniques like dynamic quantization or QAT

    def distill_knowledge(
        self,
        train_dataloader: torch.utils.data.DataLoader,
        eval_dataloader: Optional[torch.utils.data.DataLoader] = None,
        callback: Optional[callable] = None
    ) -> Dict[str, List[float]]:
        """
        Main knowledge distillation training loop
        """
        history = {
            'distillation_loss': [],
            'task_loss': [],
            'total_loss': [],
            'eval_loss': [],
            'eval_accuracy': []
        }
        
        for epoch in range(self.config.num_epochs):
            epoch_losses = {
                'distillation_loss': 0.,
                'task_loss': 0.,
                'total_loss': 0.
            }
            
            # Training
            self.student.train()
            for batch in train_dataloader:
                step_losses = self.train_step(batch)
                for k, v in step_losses.items():
                    epoch_losses[k] += v
                    
            # Average losses for the epoch
            for k in epoch_losses:
                epoch_losses[k] /= len(train_dataloader)
                history[k].append(epoch_losses[k])
                
            # Evaluation
            if eval_dataloader is not None:
                eval_metrics = self.evaluate(eval_dataloader)
                history['eval_loss'].append(eval_metrics['eval_loss'])
                history['eval_accuracy'].append(eval_metrics['eval_accuracy'])
                
            # Optional callback for monitoring
            if callback is not None:
                callback(epoch, epoch_losses, eval_metrics if eval_dataloader else None)
                
        return history

    def save_student_model(self, path: str):
        """Save the distilled student model"""
        self.student.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    def get_compression_stats(self) -> Dict[str, float]:
        """Get model compression statistics"""
        teacher_size = sum(p.numel() for p in self.teacher.parameters())
        student_size = sum(p.numel() for p in self.student.parameters())
        
        return {
            'compression_ratio': teacher_size / student_size,
            'teacher_params': teacher_size,
            'student_params': student_size
        }