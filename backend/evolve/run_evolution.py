"""
Main entry point for the evolving RAG system.
Handles startup, configuration, and system initialization.
"""

import argparse
import logging
from pathlib import Path
import json
import os

import torch
import wandb

from .orchestrator import EvolutionOrchestrator, EvolutionConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_config(config_path: str) -> dict:
    """Load configuration from file."""
    with open(config_path, 'r') as f:
        return json.load(f)

def setup_wandb(config: dict):
    """Setup Weights & Biases tracking."""
    try:
        wandb.init(
            project="evolving-rag",
            config=config,
            name=f"run_{wandb.util.generate_id()}"
        )
    except Exception as e:
        logger.warning(f"W&B initialization failed: {e}")

def main():
    parser = argparse.ArgumentParser(description="Start the evolving RAG system")
    parser.add_argument(
        "--config",
        type=str,
        default="config.json",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Path to checkpoint to load"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode"
    )
    args = parser.parse_args()

    try:
        # Load configuration
        if os.path.exists(args.config):
            config = load_config(args.config)
            logger.info(f"Loaded configuration from {args.config}")
        else:
            logger.warning(f"Config file {args.config} not found, using defaults")
            config = {}

        # Set debug mode if requested
        if args.debug:
            logging.getLogger().setLevel(logging.DEBUG)
            config['debug'] = True

        # Initialize W&B
        setup_wandb(config)

        # Create evolution config
        evolution_config = EvolutionConfig(
            base_model=config.get('base_model', "meta-llama/Llama-2-7b-hf"),
            eval_frequency=config.get('eval_frequency', 100),
            memory_limit_gb=config.get('memory_limit_gb', 7.5),
            min_performance_threshold=config.get('min_performance_threshold', 0.7),
            max_fine_tune_steps=config.get('max_fine_tune_steps', 1000),
            eval_batch_size=config.get('eval_batch_size', 16),
            save_dir=config.get('save_dir', "evolution_data")
        )

        # Create and start orchestrator
        with EvolutionOrchestrator(
            config=evolution_config,
            load_checkpoint=args.checkpoint
        ) as orchestrator:
            logger.info("System initialized successfully")
            
            try:
                # Run forever, handle interrupts gracefully
                while True:
                    # System will run in background threads
                    # Main thread just monitors and handles interrupts
                    status = orchestrator.get_system_status()
                    logger.info(f"System status: {status}")
                    
                    # Log to W&B
                    if wandb.run is not None:
                        wandb.log(status)
                        
            except KeyboardInterrupt:
                logger.info("Shutting down gracefully...")
                
    except Exception as e:
        logger.error(f"Startup error: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main()