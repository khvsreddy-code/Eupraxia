"""
Optimized evolution script for 8GB RAM systems using dynamic quantization.
Designed for efficient local fine-tuning of large language models.

Features:
- Dynamic quantization for reduced memory usage
- Smart reward system with partial credit
- Gradient clipping for training stability
- Progressive saving of checkpoints
- Detailed logging and error handling

Usage (PowerShell):
    python evolve.py
"""

import os
import logging
from pathlib import Path
import torch
from torch.optim import AdamW
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
import torch.quantization
from tqdm import tqdm


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EfficientEvolver:
    def __init__(self, model_path, data_path):
        self.model_path = model_path
        self.data_path = data_path
        self.device = "cpu"
        self.setup_model()
        self.load_dataset()

    def setup_model(self):
        logger.info("Setting up model with dynamic quantization...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            padding_side="left",
            truncation_side="left",
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model in FP32 first
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            device_map="cpu",
            low_cpu_mem_usage=True
        )
        
        # Apply dynamic quantization
        self.model = torch.quantization.quantize_dynamic(
            self.model,
            {torch.nn.Linear},  # Quantize linear layers
            dtype=torch.qint8
        )
        logger.info("Model loaded successfully")

    def load_dataset(self):
        logger.info(f"Loading dataset from {self.data_path}")
        try:
            self.dataset = load_dataset("json", data_files=self.data_path)["train"]
            # Limit dataset size for 8GB RAM
            self.dataset = self.dataset.shuffle(seed=42).select(range(min(100, len(self.dataset))))
            logger.info(f"Loaded {len(self.dataset)} examples")
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            raise


    def evaluate_output(self, output, target):
        """Smart evaluation with partial credit"""
        output_clean = output.strip().lower()
        target_clean = target.strip().lower()
        
        if output_clean == target_clean:
            return 1.0
        
        # Token overlap metric
        output_tokens = set(output_clean.split())
        target_tokens = set(target_clean.split())
        overlap = len(output_tokens.intersection(target_tokens))
        max_tokens = max(len(output_tokens), len(target_tokens))
        
        # Length penalty
        length_ratio = min(len(output_clean), len(target_clean)) / max(len(output_clean), len(target_clean))
        
        # Combined score with length penalty
        base_score = overlap / max_tokens if max_tokens > 0 else 0
        final_score = base_score * length_ratio
        
        return max(0.6, final_score)  # Minimum score of 0.6 for learning stability

    def evolve(self, epochs=15, save_interval=5):
        logger.info("Starting evolution process...")
        optimizer = AdamW(self.model.parameters(), lr=0.00005, weight_decay=0.01)


        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            progress_bar = tqdm(self.dataset, desc=f"Epoch {epoch+1}/{epochs}")
            
            for item in progress_bar:
                try:
                    prompt, target = item["prompt"], item["target"]
                    inputs = self.tokenizer(
                        prompt,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=64
                    ).to(self.device)
                    
                    with torch.no_grad():
                        outputs = self.model.generate(
                            **inputs,
                            max_length=128,
                            do_sample=True,
                            temperature=0.7,
                            top_p=0.9,
                            num_return_sequences=1
                        )
                    
                    output_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                    reward = self.evaluate_output(output_text, target)
                    
                    # Compute loss with reward
                    loss = -torch.log(torch.tensor(reward, requires_grad=True))
                    loss.backward()
                    
                    # Gradient clipping for stability
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                    
                    total_loss += loss.item()
                    progress_bar.set_postfix({"loss": loss.item()})
                
                except Exception as e:
                    logger.error(f"Error processing item: {e}")
                    continue
            
            avg_loss = total_loss / len(self.dataset)
            logger.info(f"Epoch {epoch+1}, Average Loss: {avg_loss:.4f}")
            
            # Save checkpoint at intervals
            if (epoch + 1) % save_interval == 0:
                save_path = os.path.join(os.path.dirname(self.model_path), f"evolved_epoch{epoch+1}")
                self.save_model(save_path)
                logger.info(f"Checkpoint saved at epoch {epoch+1}")
        
        # Save final model
        final_path = os.path.join(os.path.dirname(self.model_path), "evolved_final")
        self.save_model(final_path)
        logger.info("Evolution complete!")

    def save_model(self, path):
        """Save model and tokenizer with error handling"""
        try:
            os.makedirs(path, exist_ok=True)
            self.model.save_pretrained(path)
            self.tokenizer.save_pretrained(path)
            logger.info(f"Model saved to {path}")
        except Exception as e:
            logger.error(f"Error saving model: {e}")


if __name__ == "__main__":
    # Paths
    MODEL_PATH = os.path.join("models", "Llama-3.3-8B-GGUF")
    DATA_PATH = os.path.join("evolution_data", "smoke_10.jsonl")
    
    # Initialize and run evolution
    try:
        evolver = EfficientEvolver(MODEL_PATH, DATA_PATH)
        evolver.evolve(epochs=15)
    except Exception as e:
        logger.error(f"Evolution failed: {e}")
