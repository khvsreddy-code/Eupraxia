import torch
from typing import List, Dict, Any
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import time

class ModelEvaluator:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        
    def measure_inference_speed(self, prompt: str, num_runs: int = 5) -> Dict[str, float]:
        """Measure model inference speed metrics."""
        times = []
        tokens_generated = []
        
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)
        input_length = len(inputs["input_ids"][0])
        
        for _ in range(num_runs):
            start_time = time.time()
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs["input_ids"],
                    max_length=200,
                    temperature=0.7,
                    num_return_sequences=1
                )
            end_time = time.time()
            
            generation_time = end_time - start_time
            output_length = len(outputs[0]) - input_length
            tokens_generated.append(output_length)
            times.append(generation_time)
            
        return {
            "avg_generation_time": np.mean(times),
            "std_generation_time": np.std(times),
            "tokens_per_second": np.mean(tokens_generated) / np.mean(times)
        }
    
    def evaluate_response_quality(self, prompt: str, expected_keywords: List[str] = None) -> Dict[str, Any]:
        """Evaluate the quality of model responses."""
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs["input_ids"],
                max_length=200,
                temperature=0.7,
                num_return_sequences=3,
                do_sample=True,
                top_p=0.95
            )
        
        responses = [self.tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
        
        # Calculate response diversity
        response_lengths = [len(response.split()) for response in responses]
        unique_words = set()
        for response in responses:
            unique_words.update(response.split())
        
        metrics = {
            "avg_response_length": np.mean(response_lengths),
            "vocabulary_diversity": len(unique_words) / sum(response_lengths),
            "responses": responses
        }
        
        if expected_keywords:
            keyword_coverage = []
            for response in responses:
                covered = sum(1 for keyword in expected_keywords if keyword.lower() in response.lower())
                keyword_coverage.append(covered / len(expected_keywords))
            metrics["keyword_coverage"] = np.mean(keyword_coverage)
        
        return metrics
    
    def analyze_memory_usage(self) -> Dict[str, float]:
        """Analyze model memory usage."""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "model_size_gb": total_params * 4 / (1024**3)  # Assuming float32
        }

def evaluate_model_performance(model_name: str, test_prompts: List[str], expected_keywords: List[str] = None):
    """Comprehensive model evaluation function."""
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            trust_remote_code=True
        )
        
        evaluator = ModelEvaluator(model, tokenizer)
        results = {
            "model_name": model_name,
            "memory_analysis": evaluator.analyze_memory_usage(),
            "inference_metrics": {},
            "quality_metrics": {}
        }
        
        # Test inference speed
        for prompt in test_prompts:
            results["inference_metrics"][f"prompt_{len(results['inference_metrics'])}"] = (
                evaluator.measure_inference_speed(prompt)
            )
            
        # Test response quality
        for prompt in test_prompts:
            results["quality_metrics"][f"prompt_{len(results['quality_metrics'])}"] = (
                evaluator.evaluate_response_quality(prompt, expected_keywords)
            )
            
        return results
        
    except Exception as e:
        return {"error": str(e)}