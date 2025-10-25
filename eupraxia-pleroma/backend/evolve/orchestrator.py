"""
Orchestrates the entire RAG system evolution pipeline:
1. Evaluation cycle
2. Fine-tuning decisions
3. Performance monitoring
4. Memory management
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import time
from dataclasses import dataclass
import threading
import queue

import torch
import ray
from transformers import AutoTokenizer, AutoModelForCausalLM
import wandb

from .evaluation import RAGEvaluator
from .fine_tuning import EvolvingLLM
from .memory_store import MemoryTracker

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class EvolutionConfig:
    """Configuration for evolution cycle."""
    base_model: str = "meta-llama/Llama-2-7b-hf"
    eval_frequency: int = 100  # Eval every N queries
    memory_limit_gb: float = 7.5
    min_performance_threshold: float = 0.7
    max_fine_tune_steps: int = 1000
    eval_batch_size: int = 16
    save_dir: str = "evolution_data"

class EvolutionOrchestrator:
    """Orchestrates the RAG system evolution."""
    
    def __init__(
        self,
        config: Optional[EvolutionConfig] = None,
        load_checkpoint: Optional[str] = None
    ):
        self.config = config or EvolutionConfig()
        self.save_dir = Path(self.config.save_dir)
        self.save_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self._setup_ray()
        self.evaluator = RAGEvaluator(
            memory_limit_gb=self.config.memory_limit_gb,
            save_dir=str(self.save_dir / "eval_results")
        )
        self.memory_tracker = MemoryTracker(
            max_memory_gb=self.config.memory_limit_gb
        )
        self.llm = EvolvingLLM(
            base_model=self.config.base_model,
            save_dir=str(self.save_dir / "model_checkpoints")
        )
        
        # Runtime state
        self.query_counter = 0
        self.evolution_active = False
        self._query_buffer = queue.Queue()
        
        # Load checkpoint if provided
        if load_checkpoint:
            self._load_checkpoint(load_checkpoint)
        
        # Start monitoring thread
        self._start_monitoring()
    
    def _setup_ray(self):
        """Initialize Ray for distributed processing."""
        if not ray.is_initialized():
            ray.init(
                runtime_env={
                    "working_dir": ".",
                    "pip": ["transformers", "torch", "ragas"]
                }
            )
    
    def _start_monitoring(self):
        """Start background monitoring thread."""
        def monitor_loop():
            while True:
                try:
                    # Check memory usage
                    if self.memory_tracker.should_optimize():
                        logger.info("Memory optimization triggered")
                        self._optimize_memory()
                    
                    # Process query buffer
                    while not self._query_buffer.empty():
                        query_batch = []
                        try:
                            while len(query_batch) < self.config.eval_batch_size:
                                query_batch.append(
                                    self._query_buffer.get_nowait()
                                )
                        except queue.Empty:
                            pass
                        
                        if query_batch:
                            self._process_query_batch(query_batch)
                    
                    time.sleep(1)  # Prevent tight loop
                    
                except Exception as e:
                    logger.error(f"Monitor error: {e}")
                    time.sleep(5)  # Back off on error
        
        self.monitor_thread = threading.Thread(
            target=monitor_loop,
            daemon=True
        )
        self.monitor_thread.start()
    
    def _optimize_memory(self):
        """Optimize memory usage."""
        # Clear unused caches
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Offload less frequently used components
        self.llm.optimize_memory()
        
        # Log memory state
        self.memory_tracker.log_memory_state()
    
    @ray.remote
    def _evaluate_query_batch(
        self,
        queries: List[Tuple[str, str, List[str]]]
    ):
        """Evaluate a batch of queries."""
        questions = [q[0] for q in queries]
        answers = [q[1] for q in queries]
        contexts = [q[2] for q in queries]
        
        return self.evaluator.evaluate_rag_system(
            questions=questions,
            answers=answers,
            contexts=contexts
        )
    
    def _process_query_batch(self, queries: List[Tuple[str, str, List[str]]]):
        """Process a batch of queries with evaluation."""
        try:
            # Evaluate batch
            metrics = ray.get(self._evaluate_query_batch.remote(queries))
            
            # Update counters
            self.query_counter += len(queries)
            
            # Check if evolution cycle needed
            if (self.query_counter % self.config.eval_frequency == 0 and
                metrics.answer_relevancy < self.config.min_performance_threshold):
                self._trigger_evolution_cycle()
            
        except Exception as e:
            logger.error(f"Query batch processing error: {e}")
    
    def _trigger_evolution_cycle(self):
        """Trigger model evolution cycle."""
        if self.evolution_active:
            logger.info("Evolution already in progress")
            return
        
        try:
            self.evolution_active = True
            
            # Get recent performance data
            performance = self.evaluator.get_performance_summary()
            
            # Decide on fine-tuning
            if performance["avg_relevancy"] < self.config.min_performance_threshold:
                logger.info("Starting fine-tuning cycle")
                
                # Fine-tune model
                self.llm.fine_tune(
                    max_steps=self.config.max_fine_tune_steps
                )
                
                # Evaluate after fine-tuning
                new_performance = self.evaluator.get_performance_summary()
                
                # Log improvements
                wandb.log({
                    "fine_tune_improvement": (
                        new_performance["avg_relevancy"] -
                        performance["avg_relevancy"]
                    )
                })
            
        except Exception as e:
            logger.error(f"Evolution cycle error: {e}")
        
        finally:
            self.evolution_active = False
    
    def process_query(
        self,
        question: str,
        answer: str,
        contexts: List[str]
    ):
        """Process a single query through the system."""
        try:
            # Add to processing queue
            self._query_buffer.put((question, answer, contexts))
            
            # Update memory tracking
            self.memory_tracker.track_memory()
            
        except Exception as e:
            logger.error(f"Query processing error: {e}")
    
    def get_system_status(self) -> Dict:
        """Get current system status."""
        return {
            "queries_processed": self.query_counter,
            "evolution_active": self.evolution_active,
            "memory_usage": self.memory_tracker.get_memory_usage(),
            "performance": self.evaluator.get_performance_summary()
        }
    
    def save_checkpoint(self, path: str):
        """Save system checkpoint."""
        checkpoint = {
            "config": vars(self.config),
            "query_counter": self.query_counter,
            "performance": self.evaluator.get_performance_summary(),
            "timestamp": time.time()
        }
        
        # Save model state
        model_path = Path(path) / "model"
        self.llm.save(str(model_path))
        
        # Save checkpoint data
        with open(Path(path) / "checkpoint.json", "w") as f:
            json.dump(checkpoint, f, indent=2)
    
    def _load_checkpoint(self, path: str):
        """Load system checkpoint."""
        try:
            # Load checkpoint data
            with open(Path(path) / "checkpoint.json", "r") as f:
                checkpoint = json.load(f)
            
            # Restore state
            self.config = EvolutionConfig(**checkpoint["config"])
            self.query_counter = checkpoint["query_counter"]
            
            # Load model
            model_path = Path(path) / "model"
            self.llm.load(str(model_path))
            
            logger.info(f"Checkpoint loaded from {path}")
            
        except Exception as e:
            logger.error(f"Checkpoint loading error: {e}")
    
    def __enter__(self):
        """Context manager enter."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        try:
            # Save final checkpoint
            self.save_checkpoint(str(self.save_dir / "final_checkpoint"))
            
            # Shutdown Ray
            if ray.is_initialized():
                ray.shutdown()
            
            # Close W&B
            if wandb.run is not None:
                wandb.finish()
                
        except Exception as e:
            logger.error(f"Cleanup error: {e}")