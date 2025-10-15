import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import numpy as np
import bitsandbytes as bnb

def test_environment():
    print("Testing environment setup...")
    print("PyTorch version:", torch.__version__)
    print("Transformers version:", transformers.__version__)
    print("NumPy version:", np.__version__)
    print("BitsAndBytes version:", bnb.__version__)
    print("CUDA available:", torch.cuda.is_available())
    
    print("\nSetting up 4-bit quantization config...")
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float32,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )
    
    print("\nLoading model with 4-bit quantization...")
    model_id = "microsoft/phi-2"
    
    try:
        print(f"Loading tokenizer from {model_id}...")
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        
        print(f"Loading model from {model_id} with 4-bit quantization...")
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True
        )
        
        print("Model loaded successfully!")
        
        # Test inference
        prompt = "Write a hello world program in Python"
        print(f"\nTesting inference with prompt: {prompt}")
        
        inputs = tokenizer(prompt, return_tensors="pt")
        print("Tokenization successful")
        
        with torch.no_grad():
            outputs = model.generate(
                inputs["input_ids"],
                max_length=200,
                temperature=0.7,
                num_return_sequences=1,
                pad_token_id=tokenizer.pad_token_id
            )
            
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print("\nGeneration successful!")
        print("Response:", response)
        
    except Exception as e:
        print("Error occurred:", str(e))
        raise

if __name__ == "__main__":
    test_environment()