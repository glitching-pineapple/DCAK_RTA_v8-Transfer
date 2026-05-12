# config.py - Model and experiment configuration

# ============== Model Configuration ==============
# Choose model family: "qwen", "qwen3", "llama", "gemma", or "gemma4"
MODEL_FAMILY = "qwen"

# Choose variant: "instruct" or "base"
MODEL_VARIANT = "instruct"

# Dataset: "gsm8k", "mmlupro", "strategyqa", "medqa", "triviaqa", "legalbench"
DATASET = "medqa"

# LegalBench subtask name (only used when DATASET == "legalbench").
# LegalBench is a suite of legal-reasoning tasks; each subtask is a separate
# HuggingFace config under `nguha/legalbench`. The default `hearsay` is a
# Yes/No binary-classification task — switch to e.g. "contract_qa",
# "consumer_contracts_qa", or "definition_classification" for other Yes/No
# subtasks. Non-binary subtasks are not currently wired up.
LEGALBENCH_TASK = "consumer_contracts_qa"

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
    },
    "qwen3": {
        "instruct": "Qwen/Qwen3-30B-A3B",
    },
    "gemma4": {
        "instruct": "google/gemma-4-31b-it",
    }
}

# ============== Experiment Parameters ==============
N_SAMPLES = 10        # Number of evaluation samples
RANDOM_SEED = 42        # Random seed for reproducibility

# Override random sampling with explicit dataset row indices.
# - None  → random sampling (uses N_SAMPLES + RANDOM_SEED, original behavior).
# - list  → evaluate exactly these rows in order. N_SAMPLES is ignored.
# Useful for re-running a single sample to inspect its CoT, or to repro a
# specific failure: e.g. SPECIFIC_INDICES = [258] evaluates only row 258.
SPECIFIC_INDICES = None
# Qwen3 Gen 1 (reasoning-only) needs generous budget — thinking chain alone can exceed 4096 tokens
_MAX_NEW_TOKENS_BY_FAMILY = {"qwen": 1024, "qwen3": 8192, "llama": 1024, "gemma": 1024, "gemma4": 8192}
MAX_NEW_TOKENS = _MAX_NEW_TOKENS_BY_FAMILY.get(MODEL_FAMILY, 1024)

# SE sampling budget — Qwen3 needs room for <think> block + Answer line
_SE_MAX_NEW_TOKENS_BY_FAMILY = {"qwen": 256, "qwen3": 4096, "llama": 256, "gemma": 256, "gemma4": 4096}
SE_MAX_NEW_TOKENS = _SE_MAX_NEW_TOKENS_BY_FAMILY.get(MODEL_FAMILY, 256)

# Two-pass critique budget. Previously hard-coded to 512, which truncated
# Qwen3's <think> block before it could emit the "Confidence:" / "Correct:"
# lines on hard questions.
_TWO_PASS_MAX_NEW_TOKENS_BY_FAMILY = {"qwen": 1024, "qwen3": 4096, "llama": 1024, "gemma": 1024, "gemma4": 4096}
TWO_PASS_MAX_NEW_TOKENS = _TWO_PASS_MAX_NEW_TOKENS_BY_FAMILY.get(MODEL_FAMILY, 1024)

# For Qwen3 reasoning models, skip the <think> block on the critique pass so
# the entire budget goes to the critique + Confidence/Correct lines.
TWO_PASS_DISABLE_THINKING = MODEL_FAMILY in ("qwen3", "gemma4")

# ============== Semantic Entropy Parameters ==============
# Based on Kuhn et al. (2023) "Semantic Uncertainty" paper

# Number of samples to draw for semantic entropy calculation
# Paper recommends 5-10 samples; more samples = better estimate but slower
# Temporarily set to 1 for speed during qwen3 debugging; restore to 5 for full runs
SE_NUM_SAMPLES = 1

# Skip NLI (DeBERTa) clustering for speed during testing; set False for full SE runs
SKIP_NLI_CLUSTERING = True

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

# Whether to compute answer-token logit entropy for MCQ datasets (mmlupro, medqa).
# Requires only 1 forward pass; set False to skip if not needed.
COMPUTE_ANSWER_TOKEN_ENTROPY = True


# ============== Helper Functions ==============

def get_model_name():
    """Get the full model name based on family and variant."""
    return MODEL_NAMES[MODEL_FAMILY][MODEL_VARIANT]


def get_model_label():
    """Get a readable label for results/filenames."""
    labels = {
        "qwen": "Qwen2.5-7B",
        "qwen3": "Qwen3.6-35B-A3B",
        "llama": "Llama3.1-8B",
        "gemma": "Gemma2-9B",
        "gemma4": "Gemma4-31B"
    }
    return f"{labels[MODEL_FAMILY]}-{MODEL_VARIANT}"


def print_config():
    """Print current configuration."""
    print("=" * 50)
    print("CONFIGURATION")
    print("=" * 50)
    print(f"Model: {get_model_name()}")
    if DATASET == "legalbench":
        print(f"Dataset: {DATASET} ({LEGALBENCH_TASK})")
    else:
        print(f"Dataset: {DATASET}")
    if SPECIFIC_INDICES:
        print(f"Indices: {list(SPECIFIC_INDICES)} (override; N_SAMPLES ignored)")
    else:
        print(f"Samples: {N_SAMPLES} (random, seed={RANDOM_SEED})")
    print(f"Random Seed: {RANDOM_SEED}")
    print(f"\nSemantic Entropy Settings:")
    print(f"  - Num samples: {SE_NUM_SAMPLES}")
    print(f"  - Temperature: {SE_TEMPERATURE}")
    print(f"  - NLI Model: {NLI_MODEL}")
    print(f"  - Enabled: {COMPUTE_SEMANTIC_ENTROPY}")
    print(f"\nToken budgets:")
    print(f"  - Main generation: {MAX_NEW_TOKENS}")
    print(f"  - SE sampling:     {SE_MAX_NEW_TOKENS}")
    print(f"  - Two-pass:        {TWO_PASS_MAX_NEW_TOKENS}"
          f" (thinking disabled: {TWO_PASS_DISABLE_THINKING})")
    print("=" * 50)
