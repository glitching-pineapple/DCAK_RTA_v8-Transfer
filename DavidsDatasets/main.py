# main.py - Main entry point with semantic entropy support

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from tqdm import tqdm

from config import (
    DATASET, N_SAMPLES, RANDOM_SEED, get_model_label,
    COMPUTE_SEMANTIC_ENTROPY, COMPUTE_ANSWER_TOKEN_ENTROPY, NLI_MODEL, print_config
)
from model_utils import get_device, load_model_and_tokenizer
from data_utils import load_gsm8k, load_mmlupro, load_strategyqa, load_medqa, load_triviaqa
from evaluation import evaluate_sample, evaluate_sample_quick
from visualization import (
    print_results_summary, plot_confidence_analysis, 
    calibration_analysis, print_auroc_comparison,
    semantic_entropy_analysis
)
from save_utils import save_results


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
    else:
        raise ValueError(f"Unknown dataset: {DATASET}")


def main():
    # Print configuration
    print_config()
    
    # Setup
    device = get_device()
    
    # Load data
    print("\nLoading dataset...")
    dataset = load_dataset_by_name()
    
    # Load model
    model, tokenizer = load_model_and_tokenizer()
    
    # Initialize semantic entropy calculator if enabled
    semantic_calculator = None
    if COMPUTE_SEMANTIC_ENTROPY:
        print("\nInitializing Semantic Entropy Calculator...")
        from semantic_entropy import SemanticEntropyCalculator
        semantic_calculator = SemanticEntropyCalculator(
            nli_model_name=NLI_MODEL,
            device=device,
        )
    
    # Test on a single example first
    print("\n" + "=" * 50)
    print("TESTING ON SINGLE EXAMPLE")
    print("=" * 50)
    
    result = evaluate_sample(
        model, tokenizer, dataset, 0,
        semantic_calculator=semantic_calculator,
        compute_semantic_entropy=COMPUTE_SEMANTIC_ENTROPY,
    )
    
    print(f"\nQuestion: {result['question']}")
    print(f"Ground Truth: {result['ground_truth']}")
    print(f"Model Answer: {result['model_answer']}")
    print(f"Correct: {result['is_correct']}")
    
    print(f"\n--- Confidence Metrics ---")
    print(f"Sequence log-prob: {result['seq_confidence_mean']:.4f}")
    print(f"Logit (min prob): {result['logit_confidence_min']:.4f}")
    print(f"Logit (geom mean): {result['logit_confidence_geom']:.4f}")
    
    if result.get('verbalized_confidence') is not None:
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

    if COMPUTE_SEMANTIC_ENTROPY and 'semantic_entropy' in result:
        print(f"\n--- Semantic Entropy ---")
        print(f"SE (reasoning clusters):  {result['semantic_entropy']:.4f}  ({result['num_semantic_clusters']} clusters)")
        print(f"SE (answer clusters):     {result['semantic_entropy_answers']:.4f}  ({result['num_answer_clusters']} clusters)")
        print(f"Predictive entropy:       {result['predictive_entropy']:.4f}")
        print(f"Cluster sizes: {result['cluster_sizes']}")
        if result.get('sampled_answers'):
            print(f"Sampled answers preview: {result['sampled_answers'][:3]}")
    
    print(f"\n--- Full Response Preview ---")
    print(result['full_response'][:500] + "..." if len(result['full_response']) > 500 else result['full_response'])
    
    # Run on sample of dataset
    print(f"\n" + "=" * 50)
    print(f"RUNNING EVALUATION ON {N_SAMPLES} SAMPLES")
    print("=" * 50)
    
    np.random.seed(RANDOM_SEED)
    sample_indices = np.random.choice(len(dataset), min(N_SAMPLES, len(dataset)), replace=False)
    
    results = []
    for idx in tqdm(sample_indices, desc="Evaluating"):
        try:
            result = evaluate_sample(
                model, tokenizer, dataset, idx,
                semantic_calculator=semantic_calculator,
                compute_semantic_entropy=COMPUTE_SEMANTIC_ENTROPY,
            )
            results.append(result)
        except Exception as e:
            print(f"\nError on sample {idx}: {e}")
            continue
    
    print(f"\nCompleted {len(results)} evaluations")
    
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
    
    # Calibration analysis
    if 'verbalized_confidence' in df.columns and df['verbalized_confidence'].notna().any():
        calibration_analysis(df, 'verbalized_confidence')
    
    # Semantic entropy analysis
    if COMPUTE_SEMANTIC_ENTROPY:
        semantic_entropy_analysis(df)
    
    # Plot analysis
    try:
        plot_path = f'confidence_analysis_{get_model_label()}_{DATASET}.png'
        plot_confidence_analysis(df, save_path=plot_path)
    except Exception as e:
        print(f"Could not generate plots: {e}")
    
    # Save results
    save_results(results, df)
    
    print("\n" + "=" * 50)
    print("DONE!")
    print("=" * 50)


def run_quick_test(n_samples: int = 5):
    """Quick test without semantic entropy for debugging."""
    print("Running quick test (no semantic entropy)...")
    
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