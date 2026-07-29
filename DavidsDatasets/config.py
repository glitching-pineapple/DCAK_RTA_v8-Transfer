# config.py - Model and experiment configuration
#
# Every experiment knob can be overridden via a DCAK_* environment variable,
# so sweeps never require editing this file:
#
#     DCAK_DATASET=gsm8k DCAK_MODEL_FAMILY=llama python3 main.py
#     DCAK_N_SAMPLES=150 DCAK_SEED=7 python3 main.py
#     DCAK_SPECIFIC_INDICES=258,301 python3 main.py
#
# Overrides are applied BEFORE the derived values below (USE_REASONING_FLOW,
# token budgets, SEMANTIC_ENTROPY_ACTIVE), so everything stays consistent.
# The in-file values remain the defaults.

import os as _os


def _env_str(name, default):
    return _os.environ.get(name, default)


def _env_int(name, default):
    v = _os.environ.get(name)
    return int(v) if v is not None else default


def _env_bool(name, default):
    v = _os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


def _env_int_list(name, default):
    v = _os.environ.get(name)
    if v is None:
        return default
    v = v.strip()
    return [int(x) for x in v.split(",") if x.strip()] if v else default


# ============== Model Configuration ==============
# Choose model family: "qwen", "qwen3", "llama", "llama4scout", "gemma", "gemma4", or "gptoss"
MODEL_FAMILY = _env_str("DCAK_MODEL_FAMILY", "qwen")

# Choose variant: "instruct" or "base"
MODEL_VARIANT = _env_str("DCAK_MODEL_VARIANT", "instruct")

# Dataset: "gsm8k", "mmlupro", "strategyqa", "medqa", "triviaqa", "legalbench"
DATASET = _env_str("DCAK_DATASET", "medqa")

# LegalBench subtask name (only used when DATASET == "legalbench").
# LegalBench is a suite of legal-reasoning tasks; each subtask is a separate
# HuggingFace config under `nguha/legalbench`. The default `hearsay` is a
# Yes/No binary-classification task — switch to e.g. "contract_qa",
# "consumer_contracts_qa", or "definition_classification" for other Yes/No
# subtasks. Non-binary subtasks are not currently wired up.
LEGALBENCH_TASK = _env_str("DCAK_LEGALBENCH_TASK", "consumer_contracts_qa")

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
    "llama4scout": {
        "instruct": "meta-llama/Llama-4-Scout-17B-16E-Instruct",
        "base": "meta-llama/Llama-4-Scout-17B-16E"
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
        "base": "google/gemma-4-31b",
    },
    "gptoss": {
        "instruct": "openai/gpt-oss-20b",
    }
}

# ============== Supply-chain / reproducibility pinning ==============
# Every from_pretrained / load_dataset call resolves through these pins.
# Commit SHAs captured from the HF Hub on 2026-07-06.
#
# Why: (a) reproducibility — an unpinned "latest" model or dataset can change
# under you mid-study, silently invalidating cross-run comparisons; and
# (b) supply chain — wherever trust_remote_code=True is genuinely required,
# a pinned SHA means a later compromise of the upstream repo cannot execute
# new code in your environment.
#
# None = "latest" (only for repos we couldn't resolve, e.g. gated gemma-4).
# Pin those as soon as you have access: the SHA is shown by
# `huggingface-cli repo info <name>` or the repo's commits page.
MODEL_REVISIONS = {
    "Qwen/Qwen2.5-7B-Instruct": "a09a35458c702b33eeacc393d103063234e8bc28",
    "Qwen/Qwen2.5-7B": "d149729398750b98c0af14eb82c78cfe92750796",
    "meta-llama/Llama-3.1-8B-Instruct": "0e9e39f249a16976918f6564b8830bc894c89659",
    "meta-llama/Llama-3.1-8B": "d04e592bb4f6aa9cfee91e2e20afa771667e1d4b",
    "meta-llama/Llama-4-Scout-17B-16E-Instruct": "92f3b1597a195b523d8d9e5700e57e4fbb8f20d3",
    "meta-llama/Llama-4-Scout-17B-16E": "14d516bdff6ac06cec40678529222f193386189c",
    "google/gemma-2-9b-it": "11c9b309abf73637e4b6f9a3fa1e92e615547819",
    "google/gemma-2-9b": "33c193028431c2fde6c6e51f29e6f17b60cbfac6",
    "Qwen/Qwen3-30B-A3B": "ad44e777bcd18fa416d9da3bd8f70d33ebb85d39",
    "openai/gpt-oss-20b": "6cee5e81ee83917806bbde320786a8fb61efebee",
    "google/gemma-4-31b-it": None,   # gated — pin once you have access
    "google/gemma-4-31b": None,      # gated — pin once you have access
}

DATASET_REVISIONS = {
    "openai/gsm8k": "740312add88f781978c0658806c59bc2815b9866",
    "TIGER-Lab/MMLU-Pro": "b189ec765aa7ed75c8acfea42df31fdae71f97be",
    "ChilleD/StrategyQA": "705562638fe1d8ca6bb98c66fc8f94d45fda8c83",
    "GBaker/MedQA-USMLE-4-options": "0fb93dd23a7339b6dcd27e241cb9b5eca62d4d18",
    "mandarjoshi/trivia_qa": "0f7faf33a3908546c6fd5b73a660e0f8ff173c2f",
    "nguha/legalbench": "daec8237410aa23e3faf4bc41ad8b3a7e1696826",
}

NLI_MODEL_REVISION = "7296194b9009373def4f7c5dad292651e4b5cf4e"   # microsoft/deberta-large-mnli
PRM_MODEL_REVISION = "0610740060112df12585d00a1c5f4624d2f59051"   # Qwen/Qwen2.5-Math-PRM-7B


def get_model_revision():
    """Pinned revision for the active model (None = latest, prints nothing)."""
    return MODEL_REVISIONS.get(get_model_name())


# ============== Experiment Parameters ==============
N_SAMPLES = _env_int("DCAK_N_SAMPLES", 10)     # Number of evaluation samples
RANDOM_SEED = _env_int("DCAK_SEED", 42)        # Random seed for reproducibility

# Override random sampling with explicit dataset row indices.
# - None  → random sampling (uses N_SAMPLES + RANDOM_SEED, original behavior).
# - list  → evaluate exactly these rows in order. N_SAMPLES is ignored.
# Useful for re-running a single sample to inspect its CoT, or to repro a
# specific failure: e.g. SPECIFIC_INDICES = [258] evaluates only row 258.
SPECIFIC_INDICES = _env_int_list("DCAK_SPECIFIC_INDICES", None)

# Reasoning flow = three-generation pipeline (<think>-aware Gen 1 + Gen 2
# self-rating + Gen 3 blinded critique) used for instruct-tuned reasoning
# models. The gemma4 *base* model has no chat template and no <think>
# scaffolding, so it bypasses the reasoning flow and runs the standard
# single-pass flow used by qwen/llama/gemma base+instruct.
USE_REASONING_FLOW = (
    MODEL_FAMILY in ("qwen3", "gptoss")
    or (MODEL_FAMILY == "gemma4" and MODEL_VARIANT == "instruct")
    or (MODEL_FAMILY == "llama4scout" and MODEL_VARIANT == "instruct")
)

# Qwen3 Gen 1 (reasoning-only) needs generous budget — thinking chain alone can exceed 4096 tokens
_MAX_NEW_TOKENS_BY_FAMILY = {"qwen": 1024, "qwen3": 8192, "llama": 1024, "llama4scout": 8192, "gemma": 1024, "gemma4": 8192, "gptoss": 8192}
MAX_NEW_TOKENS = _MAX_NEW_TOKENS_BY_FAMILY.get(MODEL_FAMILY, 1024)

# SE sampling budget — Qwen3 needs room for <think> block + Answer line
_SE_MAX_NEW_TOKENS_BY_FAMILY = {"qwen": 256, "qwen3": 4096, "llama": 256, "llama4scout": 4096, "gemma": 256, "gemma4": 4096, "gptoss": 4096}
SE_MAX_NEW_TOKENS = _SE_MAX_NEW_TOKENS_BY_FAMILY.get(MODEL_FAMILY, 256)

# Two-pass critique budget. Previously hard-coded to 512, which truncated
# Qwen3's <think> block before it could emit the "Confidence:" / "Correct:"
# lines on hard questions.
_TWO_PASS_MAX_NEW_TOKENS_BY_FAMILY = {"qwen": 1024, "qwen3": 4096, "llama": 1024, "llama4scout": 4096, "gemma": 1024, "gemma4": 4096, "gptoss": 4096}
TWO_PASS_MAX_NEW_TOKENS = _TWO_PASS_MAX_NEW_TOKENS_BY_FAMILY.get(MODEL_FAMILY, 1024)

# gemma4 base doesn't emit <think> blocks — use the smaller non-reasoning
# budgets to avoid wasting compute on tokens the base model will never use.
if MODEL_FAMILY == "gemma4" and not USE_REASONING_FLOW:
    MAX_NEW_TOKENS = 1024
    SE_MAX_NEW_TOKENS = 256
    TWO_PASS_MAX_NEW_TOKENS = 1024

# For reasoning-flow models, skip the <think> block on the critique pass so
# the entire budget goes to the critique + Confidence/Correct lines.
TWO_PASS_DISABLE_THINKING = USE_REASONING_FLOW

# ============== Throughput ==============
# Gen-1 batching: number of prompts decoded per forward pass in main.py.
#   1  = original serial behavior (default; byte-identical to old runs).
#   >1 = staged batched path: Gen 1 runs in left-padded batches, then Gen 2/3
#        run per-sample as before. Instruct families only — guarded families
#        (base variants, GPT-OSS) automatically fall back to serial inside
#        generate_with_logits_batched.
# Batched greedy decoding is numerically equivalent to serial up to kernel
# batching effects; run smoke_test_batched.py once per model to validate
# before trusting large runs. Memory scales with batch_size × max_new_tokens
# × vocab for the retained scores — start at 4-8.
GEN1_BATCH_SIZE = _env_int("DCAK_GEN1_BATCH_SIZE", 1)

# ============== Semantic Entropy Parameters ==============
# Based on Kuhn et al. (2023) "Semantic Uncertainty" paper

# Number of samples to draw for semantic entropy calculation
# Paper recommends 5-10 samples; more samples = better estimate but slower
# Temporarily set to 1 for speed during qwen3 debugging; restore to 5 for full runs
SE_NUM_SAMPLES = _env_int("DCAK_SE_NUM_SAMPLES", 1)

# Skip NLI (DeBERTa) clustering for speed during testing; set False for full SE runs
SKIP_NLI_CLUSTERING = _env_bool("DCAK_SKIP_NLI_CLUSTERING", True)

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
COMPUTE_SEMANTIC_ENTROPY = _env_bool("DCAK_COMPUTE_SEMANTIC_ENTROPY", True)

# Whether to compute answer-token logit entropy for MCQ datasets (mmlupro, medqa).
# Requires only 1 forward pass; set False to skip if not needed.
COMPUTE_ANSWER_TOKEN_ENTROPY = True

# ---- Effective SE status (single source of truth) ----
# COMPUTE_SEMANTIC_ENTROPY expresses intent; the debug flags above can veto
# it. Everything downstream (NLI model load, evaluation gate, summary prints)
# must consume THIS flag, not COMPUTE_SEMANTIC_ENTROPY. This prevents the
# failure mode where main.py loaded DeBERTa onto the GPU and printed
# "SE enabled" while evaluation.py silently skipped the computation.
SEMANTIC_ENTROPY_ACTIVE = (
    COMPUTE_SEMANTIC_ENTROPY
    and not SKIP_NLI_CLUSTERING
    and SE_NUM_SAMPLES >= 2   # SE over <2 samples is mathematically undefined
)

if COMPUTE_SEMANTIC_ENTROPY and not SEMANTIC_ENTROPY_ACTIVE:
    _se_veto_reasons = []
    if SKIP_NLI_CLUSTERING:
        _se_veto_reasons.append("SKIP_NLI_CLUSTERING=True")
    if SE_NUM_SAMPLES < 2:
        _se_veto_reasons.append(f"SE_NUM_SAMPLES={SE_NUM_SAMPLES} (needs >= 2)")
    # print, not warnings.warn: main.py installs a blanket warning filter
    # before importing config, which would silence this.
    print(
        "CONFIG WARNING: COMPUTE_SEMANTIC_ENTROPY=True but semantic entropy "
        "will NOT run this session: " + ", ".join(_se_veto_reasons)
        + ". Result files will contain no semantic_entropy columns."
    )


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
        "llama4scout": "Llama4-Scout-17B-16E",
        "gemma": "Gemma2-9B",
        "gemma4": "Gemma4-31B",
        "gptoss": "GPT-OSS-20B"
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
    print(f"  - Requested: {COMPUTE_SEMANTIC_ENTROPY}")
    print(f"  - Skip NLI clustering (debug): {SKIP_NLI_CLUSTERING}")
    print(f"  - ACTIVE THIS RUN: {SEMANTIC_ENTROPY_ACTIVE}")
    print(f"\nToken budgets:")
    print(f"  - Main generation: {MAX_NEW_TOKENS}")
    print(f"  - SE sampling:     {SE_MAX_NEW_TOKENS}")
    print(f"  - Two-pass:        {TWO_PASS_MAX_NEW_TOKENS}"
          f" (thinking disabled: {TWO_PASS_DISABLE_THINKING})")
    print("=" * 50)
