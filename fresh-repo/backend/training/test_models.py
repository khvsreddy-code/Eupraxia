import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import gc

def test_model(model_name, description):
    print(f"\n{'='*80}")
    print(f"Testing {model_name}")
    print(f"Description: {description}")
    print('='*80)
    
    try:
        # Configure 4-bit quantization
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float32,
            bnb_4bit_use_double_quant=True
        )
        
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        print("Loading model...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True
        )
        
        # Test generation
        prompt = "def fibonacci(n):\n    # Function to calculate fibonacci number\n    "
        inputs = tokenizer(prompt, return_tensors="pt")
        
        print("\nGenerating test output...")
        with torch.no_grad():
            outputs = model.generate(
                inputs["input_ids"],
                max_length=200,
                temperature=0.7,
                num_return_sequences=1
            )
            
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print("\nTest output:")
        print(response)
        
        # Memory cleanup
        del model
        del tokenizer
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        print("\nTest completed successfully!")
        return True
        
    except Exception as e:
        print(f"\nError testing model: {str(e)}")
        return False

# List of models to test, starting with smallest
models_to_test = [
    ("Salesforce/codegen-350M-mono", "350M parameters, specialized for Python/JS"),
    ("Salesforce/codet5p-770m", "770M parameters, encoder-decoder model"),
    ("bigcode/starcoderbase-1b", "1B parameters, multi-language model"),
    ("deepseek-ai/deepseek-coder-1.3b-base", "1.3B parameters, trained on 2T tokens"),
    ("bigcode/starcoder2-3b", "3B parameters, newest model"),
    ("WizardLM/WizardCoder-3B-V1.0", "3B parameters, instruction-tuned"),
    ("microsoft/phi-2", "2.7B parameters, efficient architecture"),
    ("bigcode/santacoder", "1.1B parameters, multi-language code model"),
    ("EleutherAI/gpt-neo-1.3B", "1.3B parameters, general code generation"),
    ("openai-community/gpt2-xl", "1.5B parameters, classic GPT-2 XL"),
    ("gpt2", "124M parameters, baseline GPT-2")
]

# Run tests
successful_models = []
failed_models = []

for model_name, description in models_to_test:
    success = test_model(model_name, description)
    if success:
        successful_models.append(model_name)
    else:
        failed_models.append(model_name)

print("\n\nTesting Summary")
print("=" * 50)
print("\nSuccessful models:")
for model in successful_models:
    print(f"✓ {model}")
print("\nFailed models:")
for model in failed_models:
    print(f"✗ {model}")