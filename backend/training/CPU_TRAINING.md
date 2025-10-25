# Training on CPU Systems (IdeaCentre AIO)

This guide explains how to run model training on systems without dedicated GPUs, specifically optimized for the IdeaCentre AIO 24ARR9 and similar integrated-graphics systems.

## System Requirements

Your IdeaCentre AIO 24ARR9 typically has:
- AMD Ryzen processor (good for CPU compute)
- Integrated graphics (no dedicated GPU)
- 8-16GB RAM
- SSD storage

## Installation (CPU-optimized)

Create a Python virtual environment and install CPU-optimized dependencies:

```powershell
# Create and activate venv
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install CPU-optimized requirements
pip install --upgrade pip
pip install -r training/requirements_cpu.txt
```

## Memory-Efficient Training

The script `train_cpu_efficient.py` includes several optimizations for CPU training:

1. Batched data loading
2. 8-bit quantization (reduces memory usage)
3. ONNX optimization for inference
4. JAX for efficient CPU computation
5. LangChain for pipeline management

### Start Small

Begin with a smaller model first:

```powershell
# Start with CodeQwen-7B (smaller model)
python training/train_cpu_efficient.py `
  --model "Qwen/CodeQwen1.5-7B-Chat" `
  --data ./codenet_dataset.jsonl `
  --output_dir ./fine_tuned_cpu `
  --batch_size 2 `
  --eval_steps 50 `
  --use_8bit
```

### Memory Management Tips

1. Close other applications while training
2. Use small batch sizes (2-4)
3. Enable 8-bit quantization with `--use_8bit`
4. If out of memory:
   - Reduce `batch_size` to 1
   - Use shorter sequences (default max_length=512)
   - Try the 7B model variants first

### Recommended Model Order

Try models in this order (from least to most resource-intensive):

1. NTQAI/Nxcode-CQ-7B-orpo (7B parameters)
2. Qwen/CodeQwen1.5-7B-Chat (7B parameters)
3. Qwen/Qwen2.5-Coder-32B (requires significant RAM; may need disk offloading)

## Monitoring

Watch system resources:
- Open Task Manager (Ctrl+Shift+Esc)
- Monitor CPU usage and memory
- If memory exceeds 90%, reduce batch size or sequence length

## Troubleshooting

Common issues and solutions:

1. Out of Memory
   ```
   Solution: Add --batch_size 1 --use_8bit
   ```

2. Slow Processing
   ```
   Solution: Enable JAX CPU optimizations (automatic in train_cpu_efficient.py)
   ```

3. Process Killed
   ```
   Solution: 
   1. Close other applications
   2. Try a smaller model
   3. Reduce batch_size and sequence length
   ```

## Next Steps

After successful small-scale training:
1. Gradually increase batch size if memory allows
2. Try longer training runs overnight
3. Consider cloud options for larger models