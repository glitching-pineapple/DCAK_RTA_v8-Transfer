# config.py - Model and experiment configuration

# ============== Model Configuration ==============
# Choose model family: "qwen", "llama", or "gemma"
MODEL_FAMILY = "qwen"

# Choose variant: "instruct" or "base"
MODEL_VARIANT = "instruct"

# Dataset: "gsm8k", "mmlupro", "strategyqa", "medqa", "triviaqa"
DATASET = "medqa"

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

# ============== Experiment Parameters ==============
N_SAMPLES = 10        # Number of evaluation samples 
RANDOM_SEED = 42        # Random seed for reproducibility
MAX_NEW_TOKENS = 1024    # Max tokens for main generation

# ============== Semantic Entropy Parameters ==============
# Based on Kuhn et al. (2023) "Semantic Uncertainty" paper

# Number of samples to draw for semantic entropy calculation
# Paper recommends 5-10 samples; more samples = better estimate but slower
SE_NUM_SAMPLES = 5

# Temperature for sampling answers (for semantic entropy)
# Paper found 0.5 to be optimal, balancing diversity and accuracy
SE_TEMPERATURE = 0.5

# Whether to use length normalization for log-probs
# Paper suggests this helps for datasets with variable-length answers
SE_LENGTH_NORMALIZE = True

# NLI model for bidirectional entailment clustering
# Default is DeBERTa-large fine-tuned on MNLI (as used in paper)
NLI_MODEL = "microsoft/deberta-large-mnli"

# Whether to compute semantic entropy (slower but more informative)
COMPUTE_SEMANTIC_ENTROPY = True


# ============== Helper Functions ==============

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


def print_config():
    """Print current configuration."""
    print("=" * 50)
    print("CONFIGURATION")
    print("=" * 50)
    print(f"Model: {get_model_name()}")
    print(f"Dataset: {DATASET}")
    print(f"Samples: {N_SAMPLES}")
    print(f"Random Seed: {RANDOM_SEED}")
    print(f"\nSemantic Entropy Settings:")
    print(f"  - Num samples: {SE_NUM_SAMPLES}")
    print(f"  - Temperature: {SE_TEMPERATURE}")
    print(f"  - NLI Model: {NLI_MODEL}")
    print(f"  - Enabled: {COMPUTE_SEMANTIC_ENTROPY}")
    print("=" * 50)
