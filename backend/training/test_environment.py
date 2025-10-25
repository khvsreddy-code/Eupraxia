import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

def test_environment():
    # Test PyTorch
    print("PyTorch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    
    # Test transformers
    print("\nTransformers version:", transformers.__version__)
    
    # Test numpy
    print("NumPy version:", np.__version__)
    
    # Test loading a small model (Microsoft/phi-2)
    # Configure 4-bit quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float32
    )
    print("\nTesting model loading...")
    try:
    print("\nTesting model loading with 4-bit quantization...")
        model = AutoModelForCausalLM.from_pretrained(
            "microsoft/phi-2",
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
            quantization_config=bnb_config,
            device_map="auto",
        print("Successfully loaded Microsoft Phi-2 model")
        
        print("Successfully loaded Microsoft Phi-2 model with 4-bit quantization")
        prompt = "Write a hello world program in Python"
        inputs = tokenizer(prompt, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model.generate(
                inputs["input_ids"],
                max_length=200,
                temperature=0.7,
                num_return_sequences=1
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print("\nTest generation successful!")
        print("Prompt:", prompt)
        print("Response:", response)
        
    except Exception as e:
        print("Error loading model:", str(e))

if __name__ == "__main__":
    test_environment()