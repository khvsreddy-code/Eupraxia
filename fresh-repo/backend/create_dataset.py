"""
Prepare example dataset for model evolution.
Creates a diverse dataset covering multiple capabilities.
"""

import json
import os
from pathlib import Path

EXAMPLES = [
    # Coding examples
    {
        "prompt": "Write a Python function to find the factorial of a number.",
        "target": "def factorial(n):\n    if n == 0:\n        return 1\n    return n * factorial(n-1)"
    },
    {
        "prompt": "Create a function to check if a string is a palindrome.",
        "target": "def is_palindrome(s):\n    s = s.lower()\n    return s == s[::-1]"
    },
    # Math reasoning
    {
        "prompt": "Solve: What is the area of a circle with radius 5?",
        "target": "The area of a circle with radius 5 is 78.54 square units (π * 5²)."
    },
    {
        "prompt": "If a train travels at 60 mph for 2.5 hours, how far does it go?",
        "target": "The train travels 150 miles (60 mph * 2.5 hours)."
    },
    # Scientific explanations
    {
        "prompt": "Explain how photosynthesis works.",
        "target": "Photosynthesis is the process where plants convert sunlight, water, and CO2 into glucose and oxygen. The chlorophyll in leaves captures light energy, which is used to split water molecules. The hydrogen combines with CO2 to form glucose, while oxygen is released."
    },
    # Creative writing
    {
        "prompt": "Write a haiku about autumn leaves.",
        "target": "Crimson leaves falling\nDancing in the autumn breeze\nNature's last display"
    },
    # Logic and reasoning
    {
        "prompt": "If all A are B, and all B are C, what can we conclude about A and C?",
        "target": "We can conclude that all A are C. This is an example of transitive reasoning in syllogistic logic."
    },
    # Image generation prompts
    {
        "prompt": "Generate an image of a futuristic city at sunset.",
        "target": "A stunning cityscape with gleaming glass skyscrapers, flying vehicles, and a brilliant orange sunset casting long shadows. The buildings have sleek, organic curves and are interconnected by transparent skyways."
    },
    # Multi-step problem solving
    {
        "prompt": "Design an algorithm to find the shortest path between two points in a maze.",
        "target": "1. Use Breadth-First Search (BFS)\n2. Start from source node\n3. Explore all neighbors\n4. Keep track of visited cells\n5. Store path information\n6. When target found, backtrack to get path\n7. Return shortest path found"
    },
    # Data analysis
    {
        "prompt": "How would you analyze a dataset for outliers?",
        "target": "1. Calculate mean and standard deviation\n2. Use IQR method (Q3-Q1)\n3. Apply z-score analysis\n4. Create box plots\n5. Check for values beyond 3 standard deviations\n6. Consider domain-specific rules"
    }
]

def create_dataset(output_path, examples=EXAMPLES):
    """Create and save the example dataset."""
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Write examples to JSONL file
    with open(output_path, 'w', encoding='utf-8') as f:
        for example in examples:
            f.write(json.dumps(example) + '\n')
    
    print(f"Created dataset with {len(examples)} examples at {output_path}")

if __name__ == "__main__":
    output_path = Path("evolution_data") / "smoke_10.jsonl"
    create_dataset(output_path)
    print("Dataset creation complete!")