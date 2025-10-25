import wandb
import os
from typing import Dict, Any

def setup_wandb(config: Dict[str, Any]) -> wandb.Run:
    """Initialize W&B tracking for evolution system.
    
    Args:
        config: Evolution system configuration dictionary
        
    Returns:
        wandb.Run: Initialized W&B run object
    """
    wandb_api_key = os.getenv("WANDB_API_KEY")
    if not wandb_api_key:
        print("⚠️ WANDB_API_KEY not found in environment. W&B tracking disabled.")
        return None
        
    project_name = "eupraxia-evolution"
    
    run = wandb.init(
        project=project_name,
        config=config,
        tags=["evolution"],
        notes="Eupraxia Evolution System Run"
    )
    
    return run

def log_evolution_metrics(
    run: wandb.Run,
    cycle: int,
    memory_usage: float,
    critic_scores: Dict[str, float],
    model_outputs: Dict[str, Any]
):
    """Log evolution metrics to W&B.
    
    Args:
        run: Active W&B run
        cycle: Current evolution cycle number
        memory_usage: Current memory usage in GB
        critic_scores: Dictionary of critic evaluation scores
        model_outputs: Dictionary containing model generation metrics
    """
    if run is None:
        return
        
    metrics = {
        "cycle": cycle,
        "memory_gb": memory_usage,
        **critic_scores,
        "num_responses": len(model_outputs.get("responses", [])),
        "avg_response_length": sum(len(r) for r in model_outputs.get("responses", [])) / max(len(model_outputs.get("responses", [])), 1)
    }
    
    run.log(metrics)
    
def finish_run(run: wandb.Run):
    """Cleanly finish W&B run.
    
    Args:
        run: Active W&B run to finish
    """
    if run is not None:
        run.finish()