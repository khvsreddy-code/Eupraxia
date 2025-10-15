"""
Memory-efficient code generation using Microsoft's Phi-2 model (2.7B parameters).
This script uses 4-bit quantization and efficient attention to run on 8GB RAM systems.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional
import logging
import os

# Configure basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CodeGenerator:
    def __init__(self, model_name: str = "microsoft/phi-2", device: str = "cpu"):
        """Initialize the code generator with a small, efficient model."""
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            
            # Load model with 4-bit quantization for memory efficiency
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",  # Let transformers handle device placement
                load_in_4bit=True,  # Use 4-bit quantization
                low_cpu_mem_usage=True,
                trust_remote_code=True
            )
            
            logger.info(f"Successfully loaded {model_name} in 4-bit mode")
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

    def generate_code(
        self,
        prompt: str,
        max_length: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.95,
        num_return_sequences: int = 1
    ) -> Optional[str]:
        """Generate code from a text prompt using memory-efficient settings."""
        try:
            # Format prompt for code generation
            formatted_prompt = f"Write code for the following task:\n{prompt}\n\nCode:"
            
            # Encode prompt
            inputs = self.tokenizer(formatted_prompt, return_tensors="pt")
            
            # Generate with memory-efficient settings
            with torch.no_grad():  # Disable gradient computation to save memory
                outputs = self.model.generate(
                    inputs["input_ids"],
                    max_length=max_length,
                    temperature=temperature,
                    top_p=top_p,
                    num_return_sequences=num_return_sequences,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    do_sample=True
                )
            
            # Decode and return the generated code
            generated_code = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the generated part (remove prompt)
            generated_code = generated_code[len(formatted_prompt):]
            
            return generated_code.strip()
            
        except Exception as e:
            logger.error(f"Error during code generation: {str(e)}")
            return None

    def __del__(self):
        """Clean up resources."""
        try:
            del self.model
            del self.tokenizer
            torch.cuda.empty_cache()  # Clean GPU memory if available
        except:
            pass


def main():
    """Example usage of the code generator."""
    # Initialize generator with Phi-2 (small but powerful coding model)
    generator = CodeGenerator()
    
    # Example prompts to test
    test_prompts = [
        "Write a Python function that sorts a list of numbers using quicksort",
        "Create a simple Flask API endpoint that returns the current time",
        "Write a function to check if a string is a palindrome"
    ]
    
    # Generate and print code for each prompt
    for prompt in test_prompts:
        print(f"\nPrompt: {prompt}\n")
        code = generator.generate_code(prompt)
        if code:
            print("Generated Code:")
            print(code)
        print("\n" + "="*80)


if __name__ == "__main__":
    main()