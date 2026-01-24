# config.py - Model and experiment configuration

# Choose model family: "qwen", "llama", or "gemma"
MODEL_FAMILY = "llama"

# Choose variant: "instruct" or "base"
MODEL_VARIANT = "instruct"

# Dataset: "gsm8k", "mmlupro", or "strategyqa"
DATASET = "mmlupro"

# Model name mappings
MODEL_NAMES = {
    "qwen": {
        "instruct": "Qwen/Qwen2.5-7B-Instruct",
        "base": "Qwen/Qwen2.5-7B"
    },
    "llama": {
        "instruct": "meta-llama/Llama-3.1-8B-Instruct",
        "base": "meta-llama/Llama-3.1-8B"
    },
    "gemma": {
        "instruct": "google/gemma-2-9b-it",
        "base": "google/gemma-2-9b"
    }
}

N_SAMPLES = 50
RANDOM_SEED = 42
MAX_NEW_TOKENS = 512


def get_model_name():
    """Get the full model name based on family and variant."""
    return MODEL_NAMES[MODEL_FAMILY][MODEL_VARIANT]


def get_model_label():
    """Get a readable label for results/filenames."""
    labels = {
        "qwen": "Qwen2.5-7B",
        "llama": "Llama3.1-8B",
        "gemma": "Gemma2-9B"
    }
    return f"{labels[MODEL_FAMILY]}-{MODEL_VARIANT}"