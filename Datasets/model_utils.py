# model_utils.py - Model and tokenizer loading

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from config import get_model_name, get_model_label


def get_device():
    """Check GPU availability and return device info."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if device == "cuda":
        n_gpus = torch.cuda.device_count()
        print(f"GPUs available: {n_gpus}")
        for i in range(n_gpus):
            mem = torch.cuda.get_device_properties(i).total_memory / 1e9
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)} ({mem:.1f} GB)")
    return device


def load_model_and_tokenizer(model_device: str = "cuda:0"):
    """
    Load the model and tokenizer onto a single GPU.
    
    A 7B fp16 model uses ~14GB, fits comfortably on one 40GB A100.
    Pinning to one GPU avoids cross-GPU tensor transfer overhead
    that device_map='auto' causes when sharding unnecessarily.
    """
    model_name = get_model_name()
    print(f"Loading: {model_name} → {model_device}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model onto a single GPU
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map=model_device,
        trust_remote_code=True,
    )
    model.eval()
    print(f"Model loaded successfully: {get_model_label()} on {model_device}")
    
    return model, tokenizer
