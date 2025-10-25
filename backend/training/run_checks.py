"""Run safe, low-memory checks for core ML libraries.
Saves JSON report to training_logs/checks_report.json
"""
import logging
import json
import gc
import psutil
from pathlib import Path

LOG_DIR = Path("training_logs")
LOG_DIR.mkdir(exist_ok=True)
REPORT_PATH = LOG_DIR / "checks_report.json"

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger('run_checks')

results = {}

def mem_percent():
    return psutil.virtual_memory().percent

def run_check(name, func):
    logger.info(f"Checking: {name}")
    try:
        gc.collect()
        before = mem_percent()
        ok, info = func()
        gc.collect()
        after = mem_percent()
        results[name] = {
            'ok': bool(ok),
            'info': info,
            'mem_before': before,
            'mem_after': after
        }
        logger.info(f"{name} -> {'OK' if ok else 'FAIL'}; {info}; mem {before}% -> {after}%")
    except Exception as e:
        results[name] = {'ok': False, 'info': f'exception: {e}'}
        logger.exception(f"Exception during check {name}")

# Individual checks

def check_psutil_numpy():
    try:
        import numpy as np
        import psutil
        v = (np.__version__, psutil.__version__)
        # tiny op
        a = np.array([1,2,3], dtype=np.int8)
        s = int(a.sum())
        del a
        return True, {'numpy': v[0], 'psutil': v[1], 'tiny_sum': s}
    except Exception as e:
        return False, str(e)

def check_torch():
    try:
        import torch
        ver = torch.__version__
        # tiny op on CPU
        t = torch.tensor([1.0,2.0], dtype=torch.float32)
        s = float(t.sum().item())
        del t
        # check CUDA availability
        cuda = hasattr(torch, 'cuda') and torch.cuda.is_available()
        return True, {'torch': ver, 'cuda_available': bool(cuda), 'tiny_sum': s}
    except Exception as e:
        return False, str(e)

def check_transformers_tokenizer():
    try:
        import transformers
        ver = transformers.__version__
        # small tokenizer: download may happen but it's small
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained('distilbert-base-uncased', use_fast=True)
        vocab = len(tok)
        del tok
        return True, {'transformers': ver, 'tokenizer_vocab': vocab}
    except Exception as e:
        return False, str(e)

def check_accelerate():
    try:
        from accelerate import Accelerator
        accel = Accelerator(cpu=True)
        del accel
        return True, {'accelerate': 'ok'}
    except Exception as e:
        return False, str(e)

def check_peft():
    try:
        import peft
        return True, {'peft': getattr(peft, '__version__', 'unknown')}
    except Exception as e:
        return False, str(e)

def check_bitsandbytes():
    try:
        import bitsandbytes
        return True, {'bitsandbytes': getattr(bitsandbytes, '__version__', 'unknown')}
    except Exception as e:
        return False, str(e)

def check_safetensors_tokenizers():
    try:
        import safetensors
        import tokenizers
        return True, {'safetensors': getattr(safetensors, '__version__', 'unknown'), 'tokenizers': getattr(tokenizers, '__version__', 'unknown')}
    except Exception as e:
        return False, str(e)

if __name__ == '__main__':
    checks = [
        ('psutil_numpy', check_psutil_numpy),
        ('torch', check_torch),
        ('transformers_tokenizer', check_transformers_tokenizer),
        ('accelerate', check_accelerate),
        ('peft', check_peft),
        ('bitsandbytes', check_bitsandbytes),
        ('safetensors_tokenizers', check_safetensors_tokenizers)
    ]

    for name, fn in checks:
        run_check(name, fn)

    # Save report
    with open(REPORT_PATH, 'w', encoding='utf-8') as f:
        json.dump({'results': results}, f, indent=2)

    logger.info(f"Checks complete. Report saved to {REPORT_PATH}")
    # Print summary
    for k, v in results.items():
        status = 'OK' if v.get('ok') else 'FAIL'
        logger.info(f"{k}: {status} - {v.get('info')}")
