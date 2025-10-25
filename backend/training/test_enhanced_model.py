import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from advanced_metrics import AdvancedMetrics
from enhanced_generation import EnhancedGenerator, GenerationConfig
from specialized_metrics import SpecializedMetrics
from test_cases import get_all_test_cases
import json
from pathlib import Path
import time
from typing import List, Dict, Any
from collections import defaultdict
import numpy as np

def run_comprehensive_tests(
    model_name: str,
    test_cases: List[Dict[str, Any]],
    output_dir: str
) -> Dict[str, Any]:
    """Run comprehensive tests on the model's capabilities."""
    
    # Initialize model and components
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.float16,
        load_in_8bit=True
    )
    
    # Initialize all components
    metrics = AdvancedMetrics(model, tokenizer)
    generator = EnhancedGenerator(model, tokenizer)
    specialized = SpecializedMetrics(model, tokenizer)
    
    results = {
        "model_name": model_name,
        "timestamp": time.strftime("%Y%m%d-%H%M%S"),
        "metrics": {},
        "generations": {},
        "analysis": {},
        "specialized_metrics": defaultdict(dict)
    }
    
    # Test different generation settings
    generation_configs = {
        "standard": GenerationConfig(),
        "creative": GenerationConfig(
            temperature=0.9,
            top_p=0.95,
            repetition_penalty=1.1
        ),
        "precise": GenerationConfig(
            temperature=0.3,
            top_p=0.85,
            repetition_penalty=1.3
        ),
        "diverse": GenerationConfig(
            num_return_sequences=3,
            diversity_penalty=0.5,
            temperature=0.8
        )
    }
    
    # Run generation tests
    for config_name, config in generation_configs.items():
        results["generations"][config_name] = []
        
        for case in test_cases:
            generation = generator.generate(
                prompt=case["prompt"],
                config=config,
                system_prompt=case.get("system_prompt")
            )
            
            results["generations"][config_name].append({
                "prompt": case["prompt"],
                "outputs": generation["generations"],
                "scores": generation["scores"],
                "tokens": generation["total_tokens"]
            })
    
    # Run metrics analysis
    model_analysis = metrics.analyze_model_capabilities(test_cases)
    results["metrics"] = model_analysis
    
    # Additional analysis
    all_generations = []
    for config_results in results["generations"].values():
        for gen in config_results:
            all_generations.extend(gen["outputs"])
    
    token_diversity = metrics.calculate_token_diversity(all_generations)
    results["analysis"]["token_diversity"] = token_diversity
    
    # Calculate all metrics including specialized ones
    avg_perplexity = []
    avg_coherence = []
    
    for case in test_cases:
        for gen in results["generations"]["standard"]:
            if gen["prompt"] == case["prompt"]:
                response = gen["outputs"][0]
                avg_perplexity.append(metrics.measure_perplexity(response))
                coherence = metrics.measure_response_coherence(case["prompt"], response)
                avg_coherence.append(coherence["semantic_similarity"])
                
                # Add specialized metrics based on content type
                if "type" in case:
                    if case["type"] in ["algorithm", "api_design", "debugging", "testing"]:
                        code_metrics = specialized.evaluate_code_quality(response)
                        results["specialized_metrics"]["code"][gen["prompt"]] = {
                            "complexity": code_metrics.complexity,
                            "readability": code_metrics.readability,
                            "completeness": code_metrics.completeness,
                            "correctness": code_metrics.correctness,
                            "documentation": code_metrics.documentation
                        }
                    elif case["type"] in ["api_docs", "system_design", "code_comments"]:
                        doc_metrics = specialized.evaluate_documentation_quality(response)
                        results["specialized_metrics"]["documentation"][gen["prompt"]] = {
                            "completeness": doc_metrics.completeness,
                            "clarity": doc_metrics.clarity,
                            "structure": doc_metrics.structure,
                            "examples_quality": doc_metrics.examples_quality
                        }
                    elif case["type"] in ["concept_explanation", "step_by_step", "best_practices"]:
                        teaching_metrics = specialized.evaluate_teaching_quality(response)
                        results["specialized_metrics"]["teaching"][gen["prompt"]] = {
                            "clarity": teaching_metrics.clarity,
                            "depth": teaching_metrics.depth,
                            "examples_quality": teaching_metrics.examples_quality,
                            "progression": teaching_metrics.progression
                        }
    
    # Calculate averages for all metrics
    results["analysis"]["average_metrics"] = {
        "perplexity": sum(avg_perplexity) / len(avg_perplexity),
        "coherence": sum(avg_coherence) / len(avg_coherence)
    }
    
    # Calculate specialized metric averages
    for category in results["specialized_metrics"]:
        category_metrics = results["specialized_metrics"][category]
        if category_metrics:
            metrics_dict = {}
            first_metrics = next(iter(category_metrics.values()))
            for metric in first_metrics.keys():
                values = [x[metric] for x in category_metrics.values()]
                metrics_dict[metric] = float(np.mean(values))
            results["analysis"][f"average_{category}_metrics"] = metrics_dict
            }
    }
    
    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    with open(output_path / f"model_analysis_{results['timestamp']}.json", "w") as f:
        json.dump(results, f, indent=2)
    
    return results

def main():
    # Example test cases
    test_cases = [
        {
            "type": "code_generation",
            "prompt": "Write a Python function to implement binary search:",
            "expected": "def binary_search"
        },
        {
            "type": "explanation",
            "prompt": "Explain how neural networks work in simple terms:",
            "expected": "neural networks"
        },
        {
            "type": "analysis",
            "prompt": "Analyze the time complexity of quicksort algorithm:",
            "expected": "O(n log n)"
        },
        {
            "type": "creative",
            "prompt": "Write a creative story about a programmer who discovers an AI:",
            "expected": None
        }
    ]
    
    model_name = "meta-llama/Llama-2-7b-chat-hf"  # Replace with your model
    results = run_comprehensive_tests(
        model_name=model_name,
        test_cases=test_cases,
        output_dir="model_evaluation"
    )
    
    # Print summary
    print("\nTest Results Summary:")
    print("=" * 50)
    print(f"Model: {results['model_name']}")
    print(f"Timestamp: {results['timestamp']}")
    print("\nAverage Metrics:")
    print(f"Perplexity: {results['analysis']['average_metrics']['perplexity']:.2f}")
    print(f"Coherence: {results['analysis']['average_metrics']['coherence']:.2f}")
    print("\nToken Diversity:")
    for metric, value in results['analysis']['token_diversity'].items():
        print(f"{metric}: {value:.3f}")

if __name__ == "__main__":
    main()