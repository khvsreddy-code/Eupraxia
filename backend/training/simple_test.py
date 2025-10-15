import torch
import transformers
import numpy as np

def test_imports():
    try:
        print("PyTorch version:", torch.__version__)
        print("Transformers version:", transformers.__version__)
        print("NumPy version:", np.__version__)
        print("CUDA available:", torch.cuda.is_available())
    except Exception as e:
        print("Error:", str(e))

if __name__ == "__main__":
    test_imports()