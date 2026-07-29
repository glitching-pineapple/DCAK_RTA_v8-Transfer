# main.py - Main entry point with semantic entropy support

import traceback
import warnings
# Silence HF/library deprecation noise only. The previous blanket
# filterwarnings('ignore') also muted our OWN diagnostics — notably the
# RuntimeWarning confidence.py raises when the emitted answer letter
# disagrees with the letter-probability argmax (a tokenizer-mapping bug
# signal that must stay visible).
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=UserWarning)

import numpy as np
import pandas as pd
from tqdm import tqdm

from config import (
    DATASET, N_SAMPLES, RANDOM_SEED, SPECIFIC_INDICES, get_model_label,
    SEMANTIC_ENTROPY_ACTIVE, COMPUTE_ANSWER_TOKEN_ENTROPY, NLI_MODEL, print_config
)
from model_utils import get_device, load_model_and_tokenizer
from data_utils import load_gsm8k, load_mmlupro, load_strategyqa, load_medqa, load_triviaqa, load_legalbench
from evaluation import evaluate_sample, evaluate_sample_quick
from visualization import (
    print_results_summary, plot_confidence_analysis, 
    calibration_analysis, print_auroc_comparison,
    semantic_entropy_analysis
)
from save_utils import save_results, IncrementalJSONLWriter


def set_seed(seed: int):
    """Seed every RNG the pipeline touches.

    Previously only np.random was seeded (for index selection), so any
    sampled generation — SE's do_sample=True path in particular — drew from
    torch's unseeded global RNG and differed run-to-run while the config
    banner advertised a seed.

    Note: this makes sampling reproducible for an identical sequence of
    generate() calls on the same hardware/software stack. Bitwise identity
    across GPUs/driver versions is not guaranteed (that would additionally
    need torch.use_deterministic_algorithms, at a real speed cost).
    """
    import random
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_dataset_by_name():
    """Load the dataset specified in config."""
    if DATASET == "gsm8k":
        return load_gsm8k()
    elif DATASET == "mmlupro":
        return load_mmlupro()
    elif DATASET == "strategyqa":
        return load_strategyqa()
    elif DATASET == "medqa":
        return load_medqa()
    elif DATASET == "triviaqa":
        return load_triviaqa()
    elif DATASET == "legalbench":
        return load_legalbench()
    else:
        raise ValueError(f"Unknown dataset: {DATASET}")


def main():
    # Print configuration
    print_config()

    # Seed ALL RNGs (python, numpy, torch CPU+CUDA) before any model work
    set_seed(RANDOM_SEED)

    # Setup
    device = get_device()
    
    # Load data
    print("\nLoading dataset...")
    dataset = load_dataset_by_name()
    
    # Load model on the device detected above (auto-sharded families ignore this)
    model, tokenizer = load_model_and_tokenizer(model_device=device)
    
    # Initialize semantic entropy calculator only when SE will actually run
    # (SEMANTIC_ENTROPY_ACTIVE folds in the debug vetoes) — otherwise DeBERTa
    # occupies GPU memory for a computation that is gated off.
    semantic_calculator = None
    if SEMANTIC_ENTROPY_ACTIVE:
        print("\nInitializing Semantic Entropy Calculator...")
        from config import NLI_MODEL_REVISION
        from semantic_entropy import SemanticEntropyCalculator
        semantic_calculator = SemanticEntropyCalculator(
            nli_model_name=NLI_MODEL,
            device=device,
            revision=NLI_MODEL_REVISION,
        )
    
    # Test on a single example first. When SPECIFIC_INDICES is set, use
    # the first one so the smoke-test exercises the same row(s) the main
    # loop will. Otherwise default to row 0 as before.
    test_idx = int(SPECIFIC_INDICES[0]) if SPECIFIC_INDICES else 0
    print("\n" + "=" * 50)
    print(f"TESTING ON SINGLE EXAMPLE (idx={test_idx})")
    print("=" * 50)

    result = evaluate_sample(
        model, tokenizer, dataset, test_idx,
        semantic_calculator=semantic_calculator,
        compute_semantic_entropy=SEMANTIC_ENTROPY_ACTIVE,
    )
    # Cache the smoke-test result: if test_idx is also in the evaluation set
    # (guaranteed when SPECIFIC_INDICES is set — the smoke test uses its first
    # element), reuse it instead of paying for a second identical evaluation.
    smoke_cache = {int(test_idx): result}
    
    print(f"\nQuestion: {result['question']}")
    print(f"Ground Truth: {result['ground_truth']}")
    print(f"Model Answer: {result['model_answer']}")
    print(f"Correct: {result['is_correct']}")
    
    print(f"\n--- Confidence Metrics ---")
    print(f"Mean log-prob (per-token): {result['seq_confidence_mean']:.4f}")
    print(f"Log-prob sum (length-confounded): {result['seq_log_prob_sum']:.4f}")
    print(f"Logit (min prob): {result['logit_confidence_min']:.4f}")
    print(f"Logit (geom mean): {result['logit_confidence_geom']:.4f}")
    
    if pd.notna(result.get('verbalized_confidence')):
        print(f"Verbalized confidence: {result['verbalized_confidence']:.0f}/10")
    else:
        print("Verbalized confidence: Not extracted")
    
    if result.get('more_likely_than_not') is not None:
        print(f"More likely than not: {result['more_likely_than_not']}")
    
    if COMPUTE_ANSWER_TOKEN_ENTROPY and result.get('answer_token_entropy') is not None:
        print(f"\n--- Answer Token Entropy ---")
        print(f"Entropy: {result['answer_token_entropy']:.4f} nats")
        print(f"Letter probs: {result['answer_letter_probs']}")
        print(f"Top letter: {result['top_answer_letter']}  |  Chosen raw prob: {result['chosen_answer_raw_prob']}")

    if SEMANTIC_ENTROPY_ACTIVE and 'semantic_entropy' in result:
        print(f"\n--- Semantic Entropy ---")
        print(f"SE (reasoning clusters):  {result['semantic_entropy']:.4f}  ({result['num_semantic_clusters']} clusters)")
        print(f"SE (answer clusters):     {result['semantic_entropy_answers']:.4f}  ({result['num_answer_clusters']} clusters)")
        print(f"Predictive entropy:       {result['predictive_entropy']:.4f}")
        print(f"Cluster sizes: {result['cluster_sizes']}")
        if result.get('sampled_answers'):
            print(f"Sampled answers preview: {result['sampled_answers'][:3]}")
    
    print(f"\n--- Full Response Preview ---")
    print(result['full_response'][:500] + "..." if len(result['full_response']) > 500 else result['full_response'])
    
    # Run on the chosen subset of the dataset. SPECIFIC_INDICES (if set in
    # config) wins over random sampling; otherwise pick N_SAMPLES rows at
    # random with the configured seed.
    if SPECIFIC_INDICES:
        sample_indices = [int(i) for i in SPECIFIC_INDICES]
        # Bounds check up front so a typo'd index fails loudly instead of
        # blowing up partway through the loop.
        out_of_range = [i for i in sample_indices if i < 0 or i >= len(dataset)]
        if out_of_range:
            raise IndexError(
                f"SPECIFIC_INDICES contains rows outside dataset range "
                f"[0, {len(dataset)}): {out_of_range}"
            )
        print(f"\n" + "=" * 50)
        print(f"RUNNING EVALUATION ON {len(sample_indices)} SPECIFIED INDICES: {sample_indices}")
        print("=" * 50)
    else:
        # Re-seed numpy immediately before index selection so the sampled
        # indices stay identical to historical runs (same seed → same rows)
        # regardless of how many numpy draws happened since set_seed().
        np.random.seed(RANDOM_SEED)
        sample_indices = np.random.choice(len(dataset), min(N_SAMPLES, len(dataset)), replace=False)
        print(f"\n" + "=" * 50)
        print(f"RUNNING EVALUATION ON {len(sample_indices)} RANDOM SAMPLES (seed={RANDOM_SEED})")
        print("=" * 50)
    
    results = []
    errors = []
    # Crash-safe row log: every completed row is on disk immediately, so an
    # OOM/crash at row N loses nothing. Also the mojibake-free source of truth
    # (JSONL round-trips raw model text exactly, unlike the CSV).
    row_log = IncrementalJSONLWriter()

    def _record_error(idx, e):
        # Do NOT silently drop failed rows: errors concentrate on long/hard
        # inputs (OOM, truncation edge cases), so dropping them quietly
        # biases accuracy/AUROC toward easy items. Record the failure with
        # its traceback so the denominator stays visible and auditable.
        traceback.print_exc()
        errors.append({
            "idx": int(idx),
            "error": f"{type(e).__name__}: {e}",
            "traceback": traceback.format_exc(),
        })

    from config import GEN1_BATCH_SIZE
    if GEN1_BATCH_SIZE > 1:
        # Staged batched path: Gen 1 decoded in left-padded batches (the
        # dominant per-sample cost), then extraction + Gen 2/3 per sample.
        # Chunked so each chunk's raw_scores are consumed and freed before
        # the next batch — memory stays bounded at one chunk.
        from config import USE_REASONING_FLOW
        from confidence import create_prompt, generate_with_logits_batched
        from evaluation import get_question_and_choices
        indices = [int(i) for i in sample_indices]
        for chunk_start in tqdm(range(0, len(indices), GEN1_BATCH_SIZE),
                                desc=f"Evaluating (batch={GEN1_BATCH_SIZE})"):
            chunk_idx = indices[chunk_start:chunk_start + GEN1_BATCH_SIZE]
            try:
                prompts = []
                for idx in chunk_idx:
                    q, c = get_question_and_choices(dataset[idx])
                    # Must mirror evaluate_sample's prompt settings exactly:
                    # reasoning-flow models get the no-confidence Gen-1 prompt.
                    prompts.append(create_prompt(
                        tokenizer, q, c,
                        include_confidence=not USE_REASONING_FLOW,
                    ))
                gen1_outputs = generate_with_logits_batched(
                    model, tokenizer, prompts, batch_size=GEN1_BATCH_SIZE,
                )
            except Exception as e:
                for idx in chunk_idx:
                    _record_error(idx, e)
                continue
            for k, idx in enumerate(chunk_idx):
                try:
                    if idx in smoke_cache:
                        result = smoke_cache[idx]
                    else:
                        result = evaluate_sample(
                            model, tokenizer, dataset, idx,
                            semantic_calculator=semantic_calculator,
                            compute_semantic_entropy=SEMANTIC_ENTROPY_ACTIVE,
                            gen1_precomputed=gen1_outputs[k],
                        )
                    results.append(result)
                    row_log.write_row(result)
                except Exception as e:
                    _record_error(idx, e)
            del gen1_outputs  # free this chunk's raw_scores before the next
    else:
        for idx in tqdm(sample_indices, desc="Evaluating"):
            try:
                if int(idx) in smoke_cache:
                    result = smoke_cache[int(idx)]
                else:
                    result = evaluate_sample(
                        model, tokenizer, dataset, idx,
                        semantic_calculator=semantic_calculator,
                        compute_semantic_entropy=SEMANTIC_ENTROPY_ACTIVE,
                    )
                results.append(result)
                row_log.write_row(result)
            except Exception as e:
                _record_error(idx, e)

    row_log.close()
    n_total = len(sample_indices)
    print(f"\nCompleted {len(results)}/{n_total} evaluations")
    if errors:
        failed_idx = [e["idx"] for e in errors]
        print(
            f"WARNING: {len(errors)}/{n_total} samples "
            f"({100 * len(errors) / n_total:.1f}%) FAILED and are excluded from "
            f"all metrics below. Failures are usually difficulty-correlated — "
            f"treat accuracy/AUROC as computed over a non-random (easier) "
            f"subsample.\nFailed indices: {failed_idx}"
        )
    if not results:
        print("All samples failed — nothing to analyze.")
        return
    
    # Expand answer_letter_probs dict into flat prob_A / prob_B / … columns
    if COMPUTE_ANSWER_TOKEN_ENTROPY and DATASET in ("mmlupro", "medqa"):
        for r in results:
            probs = r.pop("answer_letter_probs", None) or {}
            for letter, p in probs.items():
                r[f"prob_{letter}"] = p

    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Print summary
    print_results_summary(df)
    
    # Print AUROC comparison
    print_auroc_comparison(df)
    
    # Calibration analysis — per elicitation method. The merged column is
    # reported too, but the per-method ones are the attributable results.
    if 'verbalized_conf_source' in df.columns:
        print("\nverbalized_confidence source mix (two_pass vs single_pass fallback):")
        print(df['verbalized_conf_source'].value_counts(dropna=False).to_string())
    for conf_col in ('two_pass_confidence', 'single_pass_confidence', 'verbalized_confidence'):
        if conf_col in df.columns and df[conf_col].notna().any():
            calibration_analysis(df, conf_col)
    
    # Semantic entropy analysis
    if SEMANTIC_ENTROPY_ACTIVE:
        semantic_entropy_analysis(df)
    
    # Plot analysis
    try:
        plot_path = f'confidence_analysis_{get_model_label()}_{DATASET}.png'
        plot_confidence_analysis(df, save_path=plot_path)
    except Exception as e:
        print(f"Could not generate plots: {e}")
    
    # Save results (+ sidecar errors file when any samples failed)
    save_results(results, df, errors=errors)
    
    print("\n" + "=" * 50)
    print("DONE!")
    print("=" * 50)


def run_quick_test(n_samples: int = 5):
    """Quick test without semantic entropy for debugging."""
    print("Running quick test (no semantic entropy)...")

    set_seed(RANDOM_SEED)
    device = get_device()
    dataset = load_dataset_by_name()
    model, tokenizer = load_model_and_tokenizer()
    
    results = []
    for i in range(min(n_samples, len(dataset))):
        result = evaluate_sample_quick(model, tokenizer, dataset, i)
        results.append(result)
        print(f"Sample {i}: {result['is_correct']} | Answer: {result['model_answer']} | GT: {result['ground_truth']}")
    
    df = pd.DataFrame(results)
    print(f"\nAccuracy: {df['is_correct'].mean()*100:.1f}%")
    return df


if __name__ == "__main__":
    main()