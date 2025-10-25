"""
Dynamic Model Manager for Superhuman AI System
Handles model loading/unloading, knowledge distillation, and memory optimization
"""

import torch
import gc
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import threading
from queue import Queue
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from diffusers import StableDiffusionXLPipeline, DDIMScheduler
import numpy as np

@dataclass
class ModelConfig:
    """Configuration for model management"""
    max_memory_usage: float = 7.0  # GB
    enable_dynamic_loading: bool = True
    enable_knowledge_distillation: bool = True
    enable_gradient_checkpointing: bool = True
    batch_size: int = 1

class ModelRegistry:
    """Registry of available models with memory requirements"""
    MODELS = {
        "code": {
            "high": {
                "name": "codellama/CodeLlama-34b-hf",
                "memory": 34.0
            },
            "medium": {
                "name": "mistralai/Mistral-7B-Instruct-v0.2",
                "memory": 7.0
            },
            "low": {
                "name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                "memory": 1.1
            }
        },
        "image": {
            "high": {
                "name": "stabilityai/sdxl-turbo",
                "memory": 6.5
            },
            "medium": {
                "name": "stabilityai/stable-diffusion-xl-base-1.0",
                "memory": 4.5
            },
            "low": {
                "name": "CompVis/stable-diffusion-v1-4",
                "memory": 2.5
            }
        },
        "3d": {
            "high": {
                "name": "stabilityai/shapE",
                "memory": 5.0
            },
            "medium": {
                "name": "stabilityai/zero123-xl",
                "memory": 3.5
            }
        },
        "audio": {
            "high": {
                "name": "facebook/musicgen-large",
                "memory": 5.0
            },
            "medium": {
                "name": "facebook/musicgen-small",
                "memory": 2.5
            }
        },
        "voice": {
            "high": {
                "name": "suno/bark",
                "memory": 3.0
            },
            "medium": {
                "name": "tortoise-tts/tortoise-tts",
                "memory": 2.0
            }
        }
    }
    
    @classmethod
    def get_model_config(cls, domain: str, quality: str = "high") -> Dict[str, Any]:
        """Get model configuration based on domain and quality level"""
        return cls.MODELS[domain].get(quality, cls.MODELS[domain]["medium"])

class MemoryManager:
    """Manages memory usage and model loading/unloading"""
    def __init__(self, config: ModelConfig):
        self.config = config
        self.loaded_models: Dict[str, Any] = {}
        self.model_locks: Dict[str, threading.Lock] = {}
        self.loading_queue = Queue()
        self._start_loading_thread()
    
    def _start_loading_thread(self):
        """Start background thread for model loading"""
        def loader_thread():
            while True:
                task = self.loading_queue.get()
                if task is None:
                    break
                model_key, model_config = task
                self._load_model_internal(model_key, model_config)
                self.loading_queue.task_done()
        
        self.loader_thread = threading.Thread(target=loader_thread, daemon=True)
        self.loader_thread.start()
    
    def _load_model_internal(self, model_key: str, model_config: Dict[str, Any]):
        """Internal model loading logic"""
        try:
            if model_key in self.loaded_models:
                return
            
            # Free memory if needed
            self._ensure_memory_available(model_config["memory"])
            
            # Configure quantization
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                llm_int8_enable_fp32_cpu_offload=True
            )
            
            # Load model based on type
            if "llama" in model_config["name"].lower() or "mistral" in model_config["name"].lower():
                model = AutoModelForCausalLM.from_pretrained(
                    model_config["name"],
                    device_map="auto",
                    quantization_config=bnb_config,
                    torch_dtype=torch.float16
                )
            elif "sd" in model_config["name"].lower() or "stable-diffusion" in model_config["name"].lower():
                model = StableDiffusionXLPipeline.from_pretrained(
                    model_config["name"],
                    torch_dtype=torch.float16,
                    use_safetensors=True
                )
                model.enable_model_cpu_offload()
            
            if self.config.enable_gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
                model.gradient_checkpointing_enable()
            
            self.loaded_models[model_key] = model
            
        except Exception as e:
            print(f"Error loading model {model_key}: {str(e)}")
            raise
    
    def _ensure_memory_available(self, required_memory: float):
        """Ensure enough memory is available by unloading models if needed"""
        if not self.config.enable_dynamic_loading:
            return
        
        current_memory = self._get_current_memory_usage()
        while current_memory + required_memory > self.config.max_memory_usage:
            if not self._unload_least_used_model():
                break
            current_memory = self._get_current_memory_usage()
    
    def _get_current_memory_usage(self) -> float:
        """Get current GPU/CPU memory usage in GB"""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1e9
        return 0.0  # TODO: Add CPU memory tracking
    
    def _unload_least_used_model(self) -> bool:
        """Unload the least recently used model"""
        if not self.loaded_models:
            return False
        
        # Find least recently used model
        least_used = min(self.loaded_models.keys())
        return self.unload_model(least_used)
    
    def load_model(self, model_key: str, model_config: Dict[str, Any]):
        """Queue model for loading"""
        if model_key not in self.model_locks:
            self.model_locks[model_key] = threading.Lock()
        
        with self.model_locks[model_key]:
            if model_key not in self.loaded_models:
                self.loading_queue.put((model_key, model_config))
    
    def unload_model(self, model_key: str) -> bool:
        """Unload a model and free its memory"""
        if model_key not in self.loaded_models:
            return False
        
        with self.model_locks[model_key]:
            if model_key in self.loaded_models:
                del self.loaded_models[model_key]
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                return True
        return False
    
    def get_model(self, model_key: str) -> Optional[Any]:
        """Get a loaded model"""
        return self.loaded_models.get(model_key)

class ModelManager:
    """High-level model management interface"""
    def __init__(self, config: Optional[ModelConfig] = None):
        self.config = config or ModelConfig()
        self.memory_manager = MemoryManager(self.config)
        self.registry = ModelRegistry()
    
    def get_model(self, domain: str, quality: str = "high") -> Any:
        """Get a model for a specific domain and quality level"""
        model_config = self.registry.get_model_config(domain, quality)
        model_key = f"{domain}_{quality}"
        
        # Ensure model is loaded
        self.memory_manager.load_model(model_key, model_config)
        
        # Wait for model to be available
        while True:
            model = self.memory_manager.get_model(model_key)
            if model is not None:
                return model
            time.sleep(0.1)
    
    def cleanup(self):
        """Clean up all loaded models"""
        for model_key in list(self.memory_manager.loaded_models.keys()):
            self.memory_manager.unload_model(model_key)