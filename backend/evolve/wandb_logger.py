"""
Weights & Biases configuration for evolution tracking.
"""

import os
import wandb
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

def setup_wandb(config: dict = None):
    """Initialize W&B tracking."""
    try:
        # Load API key from env
        api_key = os.getenv("WANDB_API_KEY")
        if not api_key:
            logger.warning("WANDB_API_KEY not found in environment")
            return None

        # Initialize wandb
        wandb.login(key=api_key)
        
        # Start a new run
        run = wandb.init(
            project="eupraxia-evolution",
            config=config or {},
            resume=True,
            settings=wandb.Settings(
                start_method="thread",
                _disable_stats=True
            )
        )
        
        logger.info("✨ W&B tracking initialized successfully")
        return run
        
    except Exception as e:
        logger.error(f"Failed to initialize W&B: {e}")
        return None

def log_evolution_metrics(
    cycle: int,
    metrics: dict,
    artifacts: dict = None
):
    """Log evolution metrics to W&B."""
    if not wandb.run:
        return
        
    try:
        # Log metrics
        wandb.log({
            "cycle": cycle,
            **metrics
        })
        
        # Log artifacts if provided
        if artifacts:
            for name, artifact_path in artifacts.items():
                artifact = wandb.Artifact(
                    name=f"cycle_{cycle}_{name}",
                    type=name
                )
                artifact.add_file(artifact_path)
                wandb.log_artifact(artifact)
                
    except Exception as e:
        logger.error(f"Failed to log to W&B: {e}")

def save_model_checkpoint(
    model,
    cycle: int,
    metrics: dict
):
    """Save model checkpoint to W&B."""
    if not wandb.run:
        return
        
    try:
        # Save model checkpoint
        checkpoint_dir = Path("checkpoints")
        checkpoint_dir.mkdir(exist_ok=True)
        
        checkpoint_path = checkpoint_dir / f"model_cycle_{cycle}.pt"
        model.save_pretrained(checkpoint_path)
        
        # Create artifact
        artifact = wandb.Artifact(
            name=f"model_cycle_{cycle}",
            type="model",
            metadata=metrics
        )
        artifact.add_dir(str(checkpoint_path))
        wandb.log_artifact(artifact)
        
    except Exception as e:
        logger.error(f"Failed to save checkpoint: {e}")

def finish_run():
    """Finish the current W&B run."""
    if wandb.run:
        try:
            wandb.run.finish()
        except Exception as e:
            logger.error(f"Failed to finish W&B run: {e}")