# model_utils.py - Model and tokenizer loading

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from config import get_model_name, get_model_label


def get_device():
    """Check GPU availability and return device info."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    return device


def load_model_and_tokenizer():
    """Load the model and tokenizer."""
    model_name = get_model_name()
    print(f"Loading: {model_name}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    print(f"Model loaded successfully: {get_model_label()}")
    
    return model, tokenizer