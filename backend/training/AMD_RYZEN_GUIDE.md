# Training Guide for AMD Ryzen 5 7535HS System

## Your System Specifications
- CPU: AMD Ryzen 5 7535HS with Radeon Graphics (3.30 GHz)
- RAM: 8GB (7.15 GB usable)
- GPU: Integrated Radeon Graphics (RDNA2/3)
- System: 64-bit Windows

## Optimized Setup

This guide is specifically tuned for your hardware configuration, using ROCm for AMD GPU acceleration when possible and careful memory management.

### 1. Environment Setup

```powershell
# Create and activate venv
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install PyTorch with ROCm support and other dependencies
pip install --upgrade pip
pip install -r training/requirements_amd.txt
```

### 2. Safe Training Parameters

Your system has 8GB total RAM with integrated graphics. We'll use these safe defaults:

- Batch size: 1 (to minimize memory usage)
- Sequence length: 256 (reduced from standard 512)
- 4-bit quantization enabled
- Gradient checkpointing enabled
- Disk offloading for tensors
- Conservative learning rate

### 3. Start Training

Begin with the smallest model first:

```powershell
python training/train_amd_efficient.py `
  --model "NTQAI/Nxcode-CQ-7B-orpo" `
  --data ./codenet_dataset.jsonl `
  --output_dir ./fine_tuned_amd `
  --eval_steps 25
```

### 4. Memory Management

The script automatically:
- Monitors available system memory
- Uses 4-bit quantization when memory is tight
- Enables gradient checkpointing
- Offloads tensors to disk when needed
- Uses ROCm acceleration if available

### 5. Monitoring

1. Keep Task Manager open (Ctrl+Shift+Esc)
2. Watch these metrics:
   - Memory usage (should stay under 80%)
   - CPU usage
   - GPU usage in Performance tab

### 6. Safe Limits for Your Hardware

Given your 8GB RAM:
- Max model size: 7B parameters
- Safe batch size: 1
- Safe sequence length: 256
- Recommended models:
  1. NTQAI/Nxcode-CQ-7B-orpo (safest)
  2. Qwen/CodeQwen1.5-7B-Chat (slightly more demanding)

DO NOT attempt to load:
- Qwen2.5-Coder-32B (requires >24GB RAM)
- OpenCodeInterpreter-33B (requires >24GB RAM)

### 7. Warning Signs

Stop training if you see:
1. System becomes unresponsive
2. Memory usage exceeds 90%
3. Disk thrashing (constant hard drive activity)

### 8. Troubleshooting

If you encounter issues:

1. Out of Memory
   ```
   Solution: Add --no_rocm flag to use CPU only
   ```

2. System Slow
   ```
   Solution: Close other applications, especially browsers
   ```

3. Training Crashes
   ```
   Solutions:
   1. Reduce batch_size (already at 1)
   2. Reduce sequence length (use --max_length 128)
   3. Use --no_rocm to disable GPU usage
   ```

### 9. Next Steps

1. Start with a small test run (1-2 hours)
2. Monitor system stability
3. If stable, try longer training sessions
4. Consider adding more RAM if possible (16GB would help)

Remember: Your system can handle 7B parameter models with careful optimization, but larger models are not recommended without hardware upgrades.