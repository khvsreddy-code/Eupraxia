"""
Evolution cycle tracker and logger.
"""

import logging
from pathlib import Path
import time
from datetime import datetime
import json

# Configure logging
log_path = Path(__file__).parent.parent / "training_logs" / "evolution.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(log_path),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class EvolutionTracker:
    """Track and log evolution progress."""
    
    def __init__(self):
        self.start_time = time.time()
        self.cycle_count = 0
        self.total_samples = 0
        self.quality_scores = []
        self.sources_processed = set()
        
    def log_cycle_start(self, cycle_num: int):
        """Log the start of an evolution cycle."""
        logger.info(f"\n{'='*50}")
        logger.info(f"Starting Evolution Cycle {cycle_num}")
        logger.info(f"Time running: {(time.time() - self.start_time) / 3600:.1f} hours")
        
    def log_data_ingestion(self, sources: list, new_data_size: int):
        """Log data ingestion progress."""
        logger.info(f"\nData Ingestion:")
        logger.info(f"- New sources: {len(sources)}")
        logger.info(f"- Data size: {new_data_size / 1024:.1f}KB")
        self.sources_processed.update(sources)
        
    def log_learning_progress(self, samples: int, quality: float):
        """Log learning progress."""
        self.total_samples += samples
        self.quality_scores.append(quality)
        
        logger.info(f"\nLearning Progress:")
        logger.info(f"- New samples: {samples}")
        logger.info(f"- Quality score: {quality:.3f}")
        logger.info(f"- Total samples: {self.total_samples}")
        logger.info(f"- Avg quality: {sum(self.quality_scores) / len(self.quality_scores):.3f}")
        
    def log_memory_usage(self, memory_gb: float):
        """Log memory usage."""
        logger.info(f"\nMemory Usage:")
        logger.info(f"- Current: {memory_gb:.2f}GB")
        
    def log_cycle_complete(self, cycle_stats: dict):
        """Log cycle completion stats."""
        self.cycle_count += 1
        
        logger.info(f"\nCycle {self.cycle_count} Complete:")
        logger.info(f"- Duration: {cycle_stats.get('duration_seconds', 0):.1f}s")
        logger.info(f"- Sources processed: {len(self.sources_processed)}")
        logger.info(f"- Knowledge graph nodes: {cycle_stats.get('graph_nodes', 0)}")
        logger.info(f"- Cache hit rate: {cycle_stats.get('cache_hit_rate', 0):.2%}")
        
    def save_stats(self):
        """Save evolution statistics."""
        stats = {
            "total_runtime_hours": (time.time() - self.start_time) / 3600,
            "cycles_completed": self.cycle_count,
            "total_samples": self.total_samples,
            "sources_processed": len(self.sources_processed),
            "average_quality": sum(self.quality_scores) / len(self.quality_scores) if self.quality_scores else 0,
            "timestamp": datetime.now().isoformat()
        }
        
        stats_path = Path(__file__).parent.parent / "training_logs" / "evolution_stats.json"
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
            
# Global tracker instance
tracker = EvolutionTracker()