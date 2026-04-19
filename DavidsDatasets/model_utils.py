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
    Load the model and tokenizer.

    Small models (≤7B) are pinned to a single GPU.
    Large models (>7B, e.g. Qwen3-35B) use device_map='auto' to shard
    across all available GPUs automatically.
    """
    from config import MODEL_FAMILY
    model_name = get_model_name()

    large_model_families = {"qwen3"}
    use_auto_device_map = MODEL_FAMILY in large_model_families
    device = "auto" if use_auto_device_map else model_device
    print(f"Loading: {model_name} → device_map={device}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if MODEL_FAMILY == "qwen3" else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=dtype,
        device_map=device,
        trust_remote_code=True,
        use_safetensors=True,
    )
    model.eval()
    print(f"Model loaded successfully: {get_model_label()} on {device}")

    return model, tokenizer