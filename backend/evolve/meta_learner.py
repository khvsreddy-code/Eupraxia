"""
Enhanced Meta-Learning System with Dynamic Adaptation
Implements advanced meta-learning strategies for continuous model improvement
"""

import torch
import torch.nn as nn
from torch.optim import Adam
import numpy as np
from typing import Dict, List, Optional, Tuple
from transformers import PreTrainedModel
import copy
from dataclasses import dataclass

@dataclass
class MetaLearningConfig:
    inner_lr: float = 0.001
    meta_lr: float = 0.0001
    n_inner_steps: int = 5
    n_outer_steps: int = 3
    task_batch_size: int = 4
    eval_interval: int = 100

class MetaLearner:
    def __init__(
        self,
        base_model: PreTrainedModel,
        config: MetaLearningConfig,
        device: str = "cuda"
    ):
        self.base_model = base_model.to(device)
        self.config = config
        self.device = device
        self.meta_optimizer = Adam(self.base_model.parameters(), lr=config.meta_lr)
        self.task_memories = {}
        self.adaptation_stats = {}

    def clone_model(self) -> PreTrainedModel:
        """Create a copy of the model for task-specific adaptation"""
        return copy.deepcopy(self.base_model)

    def compute_loss(
        self,
        model: PreTrainedModel,
        batch: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Compute task-specific loss"""
        outputs = model(**batch)
        return outputs.loss

    def inner_loop_update(
        self,
        task_model: PreTrainedModel,
        support_data: List[Dict[str, torch.Tensor]]
    ) -> Tuple[PreTrainedModel, float]:
        """Perform inner loop updates for task adaptation"""
        task_optimizer = Adam(task_model.parameters(), lr=self.config.inner_lr)
        total_loss = 0.0

        for step in range(self.config.n_inner_steps):
            task_optimizer.zero_grad()
            batch_loss = 0
            
            for batch in support_data:
                loss = self.compute_loss(task_model, batch)
                batch_loss += loss
            
            batch_loss.backward()
            task_optimizer.step()
            total_loss += batch_loss.item()

        return task_model, total_loss / self.config.n_inner_steps

    def outer_loop_update(
        self,
        tasks: List[Dict],
        query_sets: List[List[Dict[str, torch.Tensor]]]
    ):
        """Perform outer loop update for meta-learning"""
        self.meta_optimizer.zero_grad()
        meta_loss = 0.0

        for task_idx, (task, query_set) in enumerate(zip(tasks, query_sets)):
            # Clone model for task-specific adaptation
            task_model = self.clone_model()
            
            # Perform inner loop updates
            adapted_model, inner_loss = self.inner_loop_update(
                task_model,
                task['support_data']
            )

            # Compute loss on query set
            query_loss = 0.0
            for batch in query_set:
                query_loss += self.compute_loss(adapted_model, batch)
            
            meta_loss += query_loss
            
            # Track adaptation statistics
            self.adaptation_stats[task['id']] = {
                'inner_loss': inner_loss,
                'query_loss': query_loss.item()
            }

        # Update meta-parameters
        meta_loss = meta_loss / len(tasks)
        meta_loss.backward()
        self.meta_optimizer.step()

        return meta_loss.item()

    def adapt_to_task(
        self,
        task_id: str,
        support_data: List[Dict[str, torch.Tensor]],
        n_adaptation_steps: int = None
    ) -> PreTrainedModel:
        """Adapt the model to a specific task"""
        if n_adaptation_steps is None:
            n_adaptation_steps = self.config.n_inner_steps

        # Initialize or retrieve task-specific model
        if task_id in self.task_memories:
            task_model = self.task_memories[task_id]
        else:
            task_model = self.clone_model()

        # Perform adaptation
        adapted_model, _ = self.inner_loop_update(
            task_model,
            support_data,
        )

        # Update task memory
        self.task_memories[task_id] = adapted_model
        return adapted_model

    def get_adaptation_metrics(self) -> Dict:
        """Get metrics about the adaptation process"""
        return {
            'n_tasks_learned': len(self.task_memories),
            'adaptation_stats': self.adaptation_stats
        }

    def save_state(self, path: str):
        """Save meta-learner state"""
        state = {
            'base_model': self.base_model.state_dict(),
            'meta_optimizer': self.meta_optimizer.state_dict(),
            'task_memories': self.task_memories,
            'adaptation_stats': self.adaptation_stats,
            'config': self.config
        }
        torch.save(state, path)

    def load_state(self, path: str):
        """Load meta-learner state"""
        state = torch.load(path)
        self.base_model.load_state_dict(state['base_model'])
        self.meta_optimizer.load_state_dict(state['meta_optimizer'])
        self.task_memories = state['task_memories']
        self.adaptation_stats = state['adaptation_stats']
        self.config = state['config']