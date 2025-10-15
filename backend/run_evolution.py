"""
Initialize and start the Ultimate Evolution AI system.
"""

import os
from pathlib import Path
import torch
import logging
from dotenv import load_dotenv

from evolve_ai import UltimateEvolution
from evolve.predictive_cache import PredictiveCache, CacheConfig
from evolve.meta_learning import MetaLearner
from evolve.knowledge_distillation import KnowledgeDistiller

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_environment():
    """Setup environment and validate configs."""
    # Load environment variables
    load_dotenv()
    
    required_vars = [
        "OPENAI_API_KEY",
        "GROQ_API_KEY",
        "HF_API_TOKEN"
    ]
    
    missing = [var for var in required_vars if not os.getenv(var)]
    if missing:
        raise ValueError(f"Missing required environment variables: {missing}")
        
    # Setup CUDA if available
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
def main():
    """Initialize and start the system."""
    try:
        logger.info("🚀 Setting up Ultimate Evolution System...")
        setup_environment()
        
        # Initialize evolution system
        config_path = Path(__file__).parent / "evolve/evolution_data/config.json"
        evolution = UltimateEvolution(
            config_path=str(config_path),
            base_model="mistralai/Mistral-7B-v0.1"
        )
        
        logger.info("✨ System initialized successfully!")
        logger.info("📊 Initial stats:")
        stats = evolution.get_stats()
        for key, value in stats.items():
            logger.info(f"  {key}: {value}")
            
        # Keep the main thread alive
        try:
            while True:
                stats = evolution.get_stats()
                logger.info("\n=== System Status Update ===")
                logger.info(f"Cycles completed: {stats['cycles_completed']}")
                logger.info(f"Average quality: {stats['avg_quality']:.3f}")
                logger.info(f"Memory usage: {stats['memory_gb']:.2f}GB")
                logger.info(f"Total samples: {stats['total_samples']}")
                logger.info(f"Graph nodes: {stats['graph_nodes']}")
                logger.info(f"Graph edges: {stats['graph_edges']}")
                
                if stats["memory_gb"] > 7.0:
                    logger.warning("⚠️ High memory usage detected!")
                    
                # Sleep for status update interval
                import time
                time.sleep(300)  # Update every 5 minutes
                
        except KeyboardInterrupt:
            logger.info("Gracefully shutting down...")
            # Cleanup will be handled by Python's GC
            
    except Exception as e:
        logger.error(f"❌ System initialization failed: {e}")
        raise

if __name__ == "__main__":
    main()
