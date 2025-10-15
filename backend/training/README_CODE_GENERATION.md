# Memory-Optimized Code Generation Guide

This guide explains how to run efficient code generation models on systems with 8GB RAM.

## Setup Instructions

1. **Create Python Environment:**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. **Install Dependencies:**
```powershell
pip install -r requirements_minimal.txt
```

3. **Run the Code Generator:**
```powershell
python code_generator.py
```

## Features & Optimizations

- Uses Microsoft's Phi-2 model (2.7B parameters)
- 4-bit quantization for minimal memory usage
- Efficient attention mechanisms
- Automatic memory cleanup
- Built-in error handling and logging

## Memory Usage Tips

- Close other applications before running
- Monitor Task Manager for memory usage
- Use shorter prompts for faster generation
- Increase Windows page file if needed

## Example Usage

```python
from code_generator import CodeGenerator

# Initialize with minimal memory footprint
generator = CodeGenerator()

# Generate code from prompt
code = generator.generate_code(
    prompt="Write a function to check if a number is prime",
    max_length=512,  # Adjust based on your needs
    temperature=0.7  # Higher = more creative, lower = more focused
)

print(code)
```

## Troubleshooting

If you see out-of-memory errors:
1. Close other applications
2. Reduce `max_length` parameter
3. Try shorter prompts
4. Increase page file size