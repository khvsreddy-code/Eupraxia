"""
Quick test script to verify our AI environment is working.
Tests: PyTorch CPU, transformers, and model loading.
"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging
import sys

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_environment():
    """Run basic tests to verify our setup."""
    try:
        # 1. Check Python version
        logger.info(f"Python version: {sys.version}")
        
        # 2. Check PyTorch
        logger.info(f"PyTorch version: {torch.__version__}")
        logger.info(f"PyTorch CPU available: {torch.cuda.is_available()}")
        
        # 3. Try loading a small model
        logger.info("Testing model loading (this may take a minute)...")
        model_name = "microsoft/phi-2"  # Small but capable model
        
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,  # Use float32 for CPU
            device_map="auto",
            trust_remote_code=True
        )
        
        # 4. Test basic inference
        prompt = "Write a Python function to check if a number is prime:"
        inputs = tokenizer(prompt, return_tensors="pt")
        
        logger.info("Testing inference...")
        with torch.no_grad():
            outputs = model.generate(
                inputs["input_ids"],
                max_length=200,
                temperature=0.7,
                do_sample=True
            )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        logger.info("\nTest generation result:")
        logger.info("-" * 40)
        logger.info(generated_text)
        logger.info("-" * 40)
        
        logger.info("\n✓ All tests passed! Environment is ready for code generation.")
        return True
        
    except Exception as e:
        logger.error(f"Error during testing: {str(e)}")
        return False

if __name__ == "__main__":
    test_environment()