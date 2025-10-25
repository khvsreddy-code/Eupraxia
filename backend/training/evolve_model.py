from test_models import test_model
from model_metrics import evaluate_model_performance
import json
from datetime import datetime

def run_model_evolution(model_name: str):
    # First, run basic model test
    test_model(
        model_name,
        "Evolution testing suite",
        advanced_config={
            'load_in_4bit': True,
            'bnb_4bit_quant_type': "nf4",
            'bnb_4bit_compute_dtype': "float32",
            'bnb_4bit_use_double_quant': True,
            'llm_int8_enable_fp32_cpu_offload': True
        }
    )
    
    # Run comprehensive evaluation
    test_prompts = [
        "Explain the concept of neural networks in simple terms:",
        "Write a Python function to implement binary search:",
        "What are the key principles of software engineering?",
        "Create a regular expression to validate email addresses:",
        "Explain the difference between supervised and unsupervised learning:"
    ]
    
    expected_keywords = [
        "neural", "layers", "weights", "learning",
        "algorithm", "function", "implementation",
        "principles", "design", "patterns",
        "regular expression", "validation",
        "supervised", "unsupervised", "training"
    ]
    
    print("\nRunning comprehensive model evaluation...")
    results = evaluate_model_performance(model_name, test_prompts, expected_keywords)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"model_evaluation_{timestamp}.json"
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nEvaluation results saved to {results_file}")
    
    # Print summary metrics
    if "error" not in results:
        print("\nSummary Metrics:")
        print("-" * 40)
        print(f"Total Parameters: {results['memory_analysis']['total_parameters']:,}")
        print(f"Model Size (GB): {results['memory_analysis']['model_size_gb']:.2f}")
        
        avg_tokens_per_second = sum(
            m['tokens_per_second'] 
            for m in results['inference_metrics'].values()
        ) / len(results['inference_metrics'])
        
        print(f"Average Tokens/Second: {avg_tokens_per_second:.2f}")
        
        if 'keyword_coverage' in next(iter(results['quality_metrics'].values())):
            avg_coverage = sum(
                m['keyword_coverage'] 
                for m in results['quality_metrics'].values()
            ) / len(results['quality_metrics'])
            print(f"Average Keyword Coverage: {avg_coverage:.2%}")

if __name__ == "__main__":
    # Example usage with a model
    model_name = "meta-llama/Llama-2-7b-chat-hf"  # Replace with your model
    run_model_evolution(model_name)