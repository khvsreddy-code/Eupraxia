# -*- coding: utf-8 -*-
import os

def update_config():
    # Backup original file
    with open('backend/ai/superhuman_ai_system.py', 'r', encoding='utf-8') as f:
        content = f.read()
    with open('backend/ai/superhuman_ai_system.py.bak', 'w', encoding='utf-8') as f:
        f.write(content)
    
    # Replace device initialization
    content = content.replace(
        'self.device = "cuda" if torch.cuda.is_available() else "cpu"',
        'self.device = "cpu"  # Force CPU for stability'
    )
    
    # Update BitsAndBytesConfig
    old_config = """        self.bnb_config = BitsAndBytesConfig(
            load_in_4bit=self.config.use_4bit,
            load_in_8bit=self.config.use_8bit,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )"""
    new_config = """        self.bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            llm_int8_enable_fp32_cpu_offload=True
        )"""
    content = content.replace(old_config, new_config)
    
    # Update model loading
    old_load = """            self.code_generator = AutoModelForCausalLM.from_pretrained(
                "mistralai/Mistral-7B-Instruct-v0.2",
                device_map="auto",
                quantization_config=self.bnb_config,
                trust_remote_code=True
            )"""
    new_load = """            self.code_generator = AutoModelForCausalLM.from_pretrained(
                "mistralai/Mistral-7B-Instruct-v0.2",
                device_map={"": "cpu"},
                low_cpu_mem_usage=True,
                quantization_config=self.bnb_config,
                torch_dtype=torch.float16,
                trust_remote_code=True
            )"""
    content = content.replace(old_load, new_load)
    
    # Write updated file
    with open('backend/ai/superhuman_ai_system.py', 'w', encoding='utf-8') as f:
        f.write(content)
    print('Successfully updated configuration')

update_config()
