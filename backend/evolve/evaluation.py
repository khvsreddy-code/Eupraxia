"""
Advanced evaluation system with RAGAS metrics and performance monitoring.
Implements:
- RAGAS evaluation suite
- Memory profiling
- Performance benchmarks
- Quality metrics tracking
"""

import os
import time
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import psutil
import torch
from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
    MultiClassCorrectness
)
import wandb

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class EvalMetrics:
    """Container for evaluation metrics."""
    answer_relevancy: float
    context_precision: float
    context_recall: float
    faithfulness: float
    latency: float
    memory_used: float
    throughput: float

class RAGEvaluator:
    """Advanced RAG system evaluator."""
    
    def __init__(
        self,
        memory_limit_gb: float = 7.5,
        save_dir: str = "eval_results"
    ):
        self.memory_limit_bytes = memory_limit_gb * 1024 * 1024 * 1024
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        
        # Initialize tracking
        self.metrics_history: List[EvalMetrics] = []
        self._setup_wandb()
    
    def _setup_wandb(self):
        """Setup W&B tracking."""
        try:
            wandb.init(
                project="rag-evaluation",
                config={
                    "memory_limit_gb": self.memory_limit_bytes / 1024**3,
                    "save_dir": str(self.save_dir)
                }
            )
        except Exception as e:
            logger.warning(f"W&B initialization failed: {e}")
    
    def evaluate_rag_system(
        self,
        questions: List[str],
        answers: List[str],
        contexts: List[str],
        ground_truth: Optional[List[str]] = None
    ) -> EvalMetrics:
        """
        Evaluate RAG system comprehensively.
        
        Args:
            questions: List of test questions
            answers: Model's answers
            contexts: Retrieved contexts
            ground_truth: Optional correct answers
            
        Returns:
            Complete evaluation metrics
        """
        start_time = time.time()
        
        # Get RAGAS scores
        results = evaluate(
            questions=questions,
            answers=answers,
            contexts=contexts,
            ground_truths=ground_truth,
            metrics=[
                answer_relevancy,
                context_precision,
                context_recall,
                faithfulness,
                MultiClassCorrectness()
            ]
        )
        
        # Calculate performance metrics
        elapsed = time.time() - start_time
        memory_used = psutil.Process().memory_info().rss
        throughput = len(questions) / elapsed
        
        # Create metrics object
        metrics = EvalMetrics(
            answer_relevancy=float(results["answer_relevancy"]),
            context_precision=float(results["context_precision"]),
            context_recall=float(results["context_recall"]),
            faithfulness=float(results["faithfulness"]),
            latency=elapsed / len(questions),
            memory_used=memory_used / 1024**3,  # Convert to GB
            throughput=throughput
        )
        
        # Track metrics
        self.metrics_history.append(metrics)
        self._log_metrics(metrics)
        
        return metrics
    
    def _log_metrics(self, metrics: EvalMetrics):
        """Log metrics to W&B and filesystem."""
        # Log to W&B
        if wandb.run is not None:
            wandb.log({
                "answer_relevancy": metrics.answer_relevancy,
                "context_precision": metrics.context_precision,
                "context_recall": metrics.context_recall,
                "faithfulness": metrics.faithfulness,
                "latency": metrics.latency,
                "memory_used_gb": metrics.memory_used,
                "throughput": metrics.throughput
            })
        
        # Save to file
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        save_path = self.save_dir / f"eval_metrics_{timestamp}.json"
        
        with open(save_path, "w") as f:
            json.dump(vars(metrics), f, indent=2)
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get summary of system performance."""
        if not self.metrics_history:
            return {"error": "No evaluation data available"}
        
        metrics_array = np.array([
            [
                m.answer_relevancy,
                m.context_precision,
                m.context_recall,
                m.faithfulness
            ]
            for m in self.metrics_history
        ])
        
        return {
            "avg_relevancy": float(np.mean(metrics_array[:, 0])),
            "avg_precision": float(np.mean(metrics_array[:, 1])),
            "avg_recall": float(np.mean(metrics_array[:, 2])),
            "avg_faithfulness": float(np.mean(metrics_array[:, 3])),
            "avg_latency": np.mean([m.latency for m in self.metrics_history]),
            "peak_memory_gb": max(m.memory_used for m in self.metrics_history),
            "avg_throughput": np.mean([m.throughput for m in self.metrics_history]),
            "total_evaluations": len(self.metrics_history)
        }
    
    def plot_metrics_history(self, save_path: Optional[str] = None):
        """Plot metrics history."""
        try:
            import matplotlib.pyplot as plt
            
            metrics_array = np.array([
                [
                    m.answer_relevancy,
                    m.context_precision,
                    m.context_recall,
                    m.faithfulness
                ]
                for m in self.metrics_history
            ])
            
            plt.figure(figsize=(12, 6))
            
            # Plot quality metrics
            plt.subplot(1, 2, 1)
            plt.plot(metrics_array)
            plt.title("RAG Quality Metrics Over Time")
            plt.legend([
                "Answer Relevancy",
                "Context Precision",
                "Context Recall",
                "Faithfulness"
            ])
            
            # Plot performance metrics
            plt.subplot(1, 2, 2)
            latencies = [m.latency for m in self.metrics_history]
            memories = [m.memory_used for m in self.metrics_history]
            
            plt.plot(latencies, label="Latency (s)")
            plt.plot(memories, label="Memory (GB)")
            plt.title("Performance Metrics Over Time")
            plt.legend()
            
            if save_path:
                plt.savefig(save_path)
            else:
                plt.show()
                
        except ImportError:
            logger.warning("matplotlib required for plotting")
            
    def generate_report(self, save_path: Optional[str] = None) -> str:
        """Generate comprehensive evaluation report."""
        summary = self.get_performance_summary()
        
        report = [
            "RAG System Evaluation Report",
            "==========================",
            "",
            "Quality Metrics:",
            f"- Answer Relevancy: {summary['avg_relevancy']:.3f}",
            f"- Context Precision: {summary['avg_precision']:.3f}",
            f"- Context Recall: {summary['avg_recall']:.3f}",
            f"- Faithfulness: {summary['avg_faithfulness']:.3f}",
            "",
            "Performance Metrics:",
            f"- Average Latency: {summary['avg_latency']:.3f} seconds",
            f"- Peak Memory Usage: {summary['peak_memory_gb']:.2f} GB",
            f"- Average Throughput: {summary['avg_throughput']:.1f} queries/second",
            "",
            f"Total Evaluations: {summary['total_evaluations']}"
        ]
        
        report_text = "\n".join(report)
        
        if save_path:
            with open(save_path, "w") as f:
                f.write(report_text)
        
        return report_text