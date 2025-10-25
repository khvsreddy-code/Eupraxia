import torch
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import (
    LoraConfig, 
    get_peft_model, 
    prepare_model_for_kbit_training,
    TaskType
)
from datasets import load_dataset
import os

def setup_model(model_name):
    print(f"\nSetting up model: {model_name}")
    
    # Quantization configuration
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )
    
    # Load model and tokenizer
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16
    )
    
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Enable memory optimizations
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)
    
    # Setup LoRA
    print("Configuring LoRA...")
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    
    model = get_peft_model(model, peft_config)
    print("\nTrainable parameters:")
    model.print_trainable_parameters()
    
    return model, tokenizer

def prepare_dataset():
    print("\nPreparing dataset...")
    # Start with a small subset of The Stack for testing
    dataset = load_dataset(
        "bigcode/the-stack-dedup",
        data_dir="data/python",
        streaming=True,
        split="train"
    )
    
    def tokenize_function(examples):
        return tokenizer(
            examples["content"],
            truncation=True,
            max_length=512,
            padding="max_length"
        )
    
    print("Tokenizing dataset...")
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names
    )
    
    return tokenized_dataset

def train_model(model, tokenizer, dataset):
    print("\nConfiguring training...")
    training_args = TrainingArguments(
        output_dir="./code-model-output",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        learning_rate=2e-4,
        num_train_epochs=1,  # Start with 1 epoch for testing
        fp16=True,
        logging_steps=10,
        save_strategy="steps",
        save_steps=200,
        evaluation_strategy="no",
        optim="paged_adamw_8bit",
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        report_to="none",
        gradient_checkpointing=True
    )
    
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator
    )
    
    print("\nStarting training...")
    trainer.train()
    
    print("\nSaving model...")
    output_dir = "./my-code-model"
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

def main():
    # Choose the best performing model from the test
    model_name = "bigcode/starcoderbase-1b"  # We'll update this based on test results
    
    # Setup
    model, tokenizer = setup_model(model_name)
    
    # Prepare dataset
    dataset = prepare_dataset()
    
    # Train
    train_model(model, tokenizer, dataset)
    
    print("\nTraining complete!")

if __name__ == "__main__":
    main()