import torch
from transformers import PreTrainedModel, PreTrainedTokenizer
from typing import List, Dict, Any, Optional
import numpy as np
from dataclasses import dataclass
from threading import Lock

@dataclass
class GenerationConfig:
    max_length: int = 200
    min_length: int = 10
    temperature: float = 0.7
    top_p: float = 0.95
    top_k: int = 50
    repetition_penalty: float = 1.2
    no_repeat_ngram_size: int = 3
    num_return_sequences: int = 1
    do_sample: bool = True
    typical_p: float = 0.95
    diversity_penalty: float = 0.0
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0

class EnhancedGenerator:
    def __init__(self, 
                 model: PreTrainedModel, 
                 tokenizer: PreTrainedTokenizer,
                 device: str = "auto"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self._generation_lock = Lock()
        
        # Cache for context window management
        self._context_cache = {}
        self._max_cache_size = 1000
        
    def _prepare_inputs(self, prompt: str, system_prompt: Optional[str] = None) -> Dict[str, torch.Tensor]:
        """Prepare inputs with optional system prompt."""
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"
        else:
            full_prompt = prompt
            
        inputs = self.tokenizer(
            full_prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.model.config.max_position_embeddings
        )
        
        return inputs
    
    def _apply_sliding_window(self, inputs: Dict[str, torch.Tensor], window_size: int = 1024) -> Dict[str, torch.Tensor]:
        """Apply sliding window for long context handling."""
        if len(inputs["input_ids"][0]) <= window_size:
            return inputs
            
        # Keep the last window_size tokens
        for key in inputs:
            if torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key][:, -window_size:]
                
        return inputs
    
    def generate(self, 
                prompt: str,
                config: Optional[GenerationConfig] = None,
                system_prompt: Optional[str] = None,
                stream: bool = False) -> Dict[str, Any]:
        """Enhanced generation with streaming support and advanced features."""
        if config is None:
            config = GenerationConfig()
            
        with self._generation_lock:
            # Prepare inputs
            inputs = self._prepare_inputs(prompt, system_prompt)
            inputs = self._apply_sliding_window(inputs)
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            generation_config = {
                "max_length": config.max_length,
                "min_length": config.min_length,
                "temperature": config.temperature,
                "top_p": config.top_p,
                "top_k": config.top_k,
                "repetition_penalty": config.repetition_penalty,
                "no_repeat_ngram_size": config.no_repeat_ngram_size,
                "num_return_sequences": config.num_return_sequences,
                "do_sample": config.do_sample,
                "typical_p": config.typical_p,
                "diversity_penalty": config.diversity_penalty,
                "return_dict_in_generate": True,
                "output_scores": True
            }
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    **generation_config
                )
            
            # Process outputs
            generated_sequences = []
            sequence_scores = []
            
            for sequence, scores in zip(outputs.sequences, outputs.sequences_scores):
                decoded = self.tokenizer.decode(sequence, skip_special_tokens=True)
                # Remove the prompt from the generation
                prompt_length = len(self.tokenizer.encode(prompt, add_special_tokens=False))
                response = self.tokenizer.decode(sequence[prompt_length:], skip_special_tokens=True)
                
                generated_sequences.append(response)
                sequence_scores.append(scores.item())
            
            result = {
                "generations": generated_sequences,
                "scores": sequence_scores,
                "prompt_tokens": len(inputs["input_ids"][0]),
                "completion_tokens": [len(seq) - len(inputs["input_ids"][0]) for seq in outputs.sequences],
                "total_tokens": [len(seq) for seq in outputs.sequences],
            }
            
            return result
    
    def generate_stream(self, prompt: str, config: Optional[GenerationConfig] = None):
        """Stream generation results token by token."""
        if config is None:
            config = GenerationConfig(num_return_sequences=1)
            
        inputs = self._prepare_inputs(prompt)
        inputs = self._apply_sliding_window(inputs)
        
        # Stream generation
        generated = []
        with torch.no_grad():
            for _ in range(config.max_length):
                outputs = self.model.generate(
                    **inputs,
                    max_length=len(inputs["input_ids"][0]) + 1,
                    do_sample=config.do_sample,
                    temperature=config.temperature,
                    top_p=config.top_p,
                    top_k=config.top_k,
                    repetition_penalty=config.repetition_penalty,
                    pad_token_id=self.tokenizer.pad_token_id
                )
                
                next_token = outputs[0][-1]
                generated.append(next_token.item())
                
                # Decode and yield the next token
                token_text = self.tokenizer.decode([next_token])
                yield {
                    "token": token_text,
                    "token_id": next_token.item(),
                    "generated_so_far": self.tokenizer.decode(generated)
                }
                
                if next_token.item() == self.tokenizer.eos_token_id:
                    break
                    
                inputs = {
                    "input_ids": torch.cat([inputs["input_ids"], next_token.unsqueeze(0).unsqueeze(0)], dim=1),
                    "attention_mask": torch.ones(1, len(inputs["input_ids"][0]) + 1, dtype=torch.long)
                }