# Eupraxia Superhuman AI - Low RAM Usage Guide

## Overview
This system is now optimized to run on machines with 8GB RAM using quantized, efficient models for all superhuman modules.

## Model Choices
- **Code Generation:** StarCoder-mini/base (4-bit quantized)
- **Image Generation:** Stable Diffusion v1.4 (CPU, low resolution)
- **Writing/Science/Engineering/Business/Art/Health/Education/Meta:** Llama-2-7b (4-bit quantized)

## Memory-Saving Techniques
- All models use `load_in_4bit=True` and `low_cpu_mem_usage=True` for minimal RAM footprint
- Image generation runs on CPU with reduced resolution
- All modules are compatible with 8GB RAM for inference

## Usage
Run the demo script:
```
powershell -ExecutionPolicy ByPass -File backend/ai/run_low_ram_demo.py
```
Or in Python:
```
python backend/ai/run_low_ram_demo.py
```

## Limitations
- Training large models from scratch is not feasible on 8GB RAM; only inference and fine-tuning small models is supported
- Generation speed may be slow, especially for image and text tasks
- For best results, use prompts that fit within the model context window
- For higher quality or larger outputs, use cloud or GPU resources

## Best Practices
- Use short, focused prompts for best performance
- Monitor RAM usage and close other applications during generation
- For image tasks, use small resolutions (e.g., 128x128 or 256x256)
- For text tasks, limit output length (max_length=512-1024)

## Troubleshooting
- If you encounter out-of-memory errors, reduce batch size, output length, or image resolution
- Ensure you have the latest versions of `transformers`, `diffusers`, and `bitsandbytes` installed
- If a model fails to load, check internet connection and available disk space

---
Eupraxia Superhuman AI is now ready for low-RAM deployment!
