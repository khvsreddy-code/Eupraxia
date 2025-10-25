"""Setup and test a local quantized LLM (Llama2) for low-RAM Windows machines.
Downloads model, converts to GGUF, runs tests, supports AMD via Vulkan/DirectML.

Usage (PowerShell):
1. Activate venv from RAG server:
   cd backend/rag_server
   .venv/Scripts/Activate.ps1

2. Download base model (with HF token):
   python setup_local_llm.py download llama2-7b
   
3. Convert to GGUF q4 (requires llama-cpp-python):
   python setup_local_llm.py convert llama2-7b

4. Test basic generation:
   python setup_local_llm.py test models/llama-2-7b-chat.Q4_K_M.gguf

5. Optional: run WSL2 setup for Vulkan/DirectML:
   python setup_local_llm.py setup-wsl2
"""
import os
import sys
import shutil
import subprocess
from pathlib import Path
from typing import Optional
import requests
from tqdm import tqdm
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download

load_dotenv()
HF_TOKEN = os.getenv('HF_API_TOKEN')
if not HF_TOKEN:
    print("Warning: HF_API_TOKEN not set in .env")

MODELS_DIR = Path('./models')
MODELS_DIR.mkdir(exist_ok=True)

def download_hf_model(model_id: str, local_dir: Path) -> None:
    """Download a model from Hugging Face using token auth."""
    if not HF_TOKEN:
        print("Error: HF_API_TOKEN required to download models")
        sys.exit(1)
    
    print(f"Downloading {model_id} to {local_dir}")
    try:
        hf_hub_download(
            repo_id=model_id,
            filename="pytorch_model.bin", 
            local_dir=local_dir,
            token=HF_TOKEN
        )
    except Exception as e:
        print(f"Error downloading model: {e}")
        sys.exit(1)

def convert_to_gguf(model_path: Path, output_path: Path) -> None:
    """Convert a model to GGUF format with 4-bit quantization."""
    try:
        subprocess.run([
            "python", "-m", "llama_cpp.convert_to_gguf",
            str(model_path),
            "--outfile", str(output_path),
            "--quantize", "q4_k_m"
        ], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error converting model: {e}")
        sys.exit(1)

def test_generation(model_path: Path) -> None:
    """Test basic generation with the quantized model."""
    try:
        from llama_cpp import Llama
        
        print("Loading model (this may take a minute)...")
        llm = Llama(model_path=str(model_path))
        
        prompt = "Write a haiku about coding:"
        print(f"\nTesting with prompt: {prompt}")
        
        output = llm.create_completion(
            prompt,
            max_tokens=100,
            temperature=0.7,
            stream=True
        )
        
        print("\nGenerated response:")
        for chunk in output:
            print(chunk['choices'][0]['text'], end='', flush=True)
        print("\n")
        
    except Exception as e:
        print(f"Error testing model: {e}")
        sys.exit(1)

def setup_wsl2() -> None:
    """Guide user through WSL2 setup for Vulkan/DirectML support."""
    print("""
WSL2 Setup Steps for Vulkan/DirectML:

1. Enable WSL2 (PowerShell as admin):
   dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart
   dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart
   
2. Set WSL2 as default:
   wsl --set-default-version 2

3. Install Ubuntu (or preferred distro):
   wsl --install -d Ubuntu

4. In Ubuntu, install Vulkan tools:
   sudo apt update
   sudo apt install vulkan-tools mesa-vulkan-drivers

5. Test Vulkan:
   vulkaninfo

6. For AMD cards, install ROCm:
   curl -O https://repo.radeon.com/rocm/apt/debian/rocm.gpg.key
   sudo apt-key add rocm.gpg.key
   echo 'deb [arch=amd64] https://repo.radeon.com/rocm/apt/debian/ ubuntu main' | sudo tee /etc/apt/sources.list.d/rocm.list
   sudo apt update
   sudo apt install rocm-dev
   
7. Set environment vars in .bashrc:
   export HSA_OVERRIDE_GFX_VERSION=10.3.0
   export ROCR_VISIBLE_DEVICES=0
   
8. Test with Python in WSL2:
   python -c "import torch; print(torch.cuda.is_available())"
""")

def main():
    if len(sys.argv) < 2:
        print("Usage: python setup_local_llm.py <command> [model_id]")
        print("Commands: download, convert, test, setup-wsl2")
        sys.exit(1)
        
    command = sys.argv[1]
    
    if command == "download":
        if len(sys.argv) != 3:
            print("Usage: python setup_local_llm.py download <model_id>")
            sys.exit(1)
        model_id = sys.argv[2]
        download_hf_model(model_id, MODELS_DIR)
        
    elif command == "convert":
        if len(sys.argv) != 3:
            print("Usage: python setup_local_llm.py convert <model_path>")
            sys.exit(1)
        model_path = Path(sys.argv[2])
        output_path = MODELS_DIR / f"{model_path.stem}.Q4_K_M.gguf"
        convert_to_gguf(model_path, output_path)
        
    elif command == "test":
        if len(sys.argv) != 3:
            print("Usage: python setup_local_llm.py test <model_path>")
            sys.exit(1)
        model_path = Path(sys.argv[2])
        test_generation(model_path)
        
    elif command == "setup-wsl2":
        setup_wsl2()
        
    else:
        print("Unknown command:", command)
        sys.exit(1)

if __name__ == "__main__":
    main()