from typing import List, Dict, Any
import numpy as np
from transformers import PreTrainedModel, PreTrainedTokenizer
import torch
from scipy.stats import entropy
import time
from collections import Counter

class AdvancedMetrics:
    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer):
        self.model = model
        self.tokenizer = tokenizer
        
    def measure_perplexity(self, text: str) -> float:
        """Calculate model perplexity on input text."""
        inputs = self.tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
            loss = outputs.loss
        return torch.exp(loss).item()
    
    def calculate_token_diversity(self, generated_texts: List[str]) -> Dict[str, float]:
        """Calculate token-level diversity metrics."""
        all_tokens = []
        for text in generated_texts:
            tokens = self.tokenizer.tokenize(text)
            all_tokens.extend(tokens)
            
        token_counts = Counter(all_tokens)
        probs = np.array(list(token_counts.values())) / len(all_tokens)
        
        return {
            "unique_tokens": len(token_counts),
            "token_entropy": entropy(probs),
            "token_diversity": len(token_counts) / len(all_tokens)
        }
    
    def measure_response_coherence(self, prompt: str, response: str) -> Dict[str, float]:
        """Measure semantic coherence between prompt and response."""
        # Encode both prompt and response
        prompt_enc = self.tokenizer.encode(prompt, return_tensors="pt")
        response_enc = self.tokenizer.encode(response, return_tensors="pt")
        
        with torch.no_grad():
            # Get hidden states for both
            prompt_outputs = self.model(prompt_enc, output_hidden_states=True)
            response_outputs = self.model(response_enc, output_hidden_states=True)
            
            # Get last hidden states
            prompt_hidden = prompt_outputs.hidden_states[-1].mean(dim=1)
            response_hidden = response_outputs.hidden_states[-1].mean(dim=1)
            
            # Calculate cosine similarity
            similarity = torch.nn.functional.cosine_similarity(
                prompt_hidden, response_hidden
            ).item()
            
        return {
            "semantic_similarity": similarity,
            "length_ratio": len(response_enc[0]) / len(prompt_enc[0])
        }
    
    def analyze_model_capabilities(self, test_cases: List[Dict[str, str]]) -> Dict[str, Any]:
        """Comprehensive analysis of model capabilities."""
        results = {
            "general_metrics": {},
            "task_specific_metrics": {},
            "generation_speed": {},
            "memory_usage": {}
        }
        
        # Test general generation capabilities
        start_time = time.time()
        total_tokens = 0
        perplexities = []
        
        for case in test_cases:
            prompt = case["prompt"]
            expected = case.get("expected", None)
            
            # Generate response
            inputs = self.tokenizer(prompt, return_tensors="pt", return_attention_mask=True)
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_length=200,
                    num_return_sequences=3,
                    do_sample=True,
                    top_p=0.95,
                    top_k=50,
                    temperature=0.7,
                    repetition_penalty=1.2,
                    return_dict_in_generate=True,
                    output_scores=True
                )
            
            responses = [
                self.tokenizer.decode(output, skip_special_tokens=True)
                for output in outputs.sequences
            ]
            
            # Calculate metrics
            total_tokens += sum(len(self.tokenizer.encode(r)) for r in responses)
            perplexity = self.measure_perplexity(responses[0])
            perplexities.append(perplexity)
            
            if expected:
                coherence = self.measure_response_coherence(prompt, responses[0])
                results["task_specific_metrics"][case.get("type", "general")] = {
                    "perplexity": perplexity,
                    "semantic_similarity": coherence["semantic_similarity"]
                }
        
        # Calculate overall metrics
        total_time = time.time() - start_time
        results["generation_speed"] = {
            "tokens_per_second": total_tokens / total_time,
            "average_latency": total_time / len(test_cases)
        }
        
        results["general_metrics"] = {
            "average_perplexity": np.mean(perplexities),
            "perplexity_std": np.std(perplexities)
        }
        
        # Memory analysis
        with torch.cuda.device(0):
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            
            # Run a sample generation
            self.model.generate(inputs["input_ids"], max_length=100)
            
            results["memory_usage"] = {
                "peak_allocated": torch.cuda.max_memory_allocated() / 1024**2,
                "current_allocated": torch.cuda.memory_allocated() / 1024**2
            }
        
        return results