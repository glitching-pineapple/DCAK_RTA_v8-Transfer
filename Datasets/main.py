# main.py - Main entry point

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from tqdm import tqdm

from config import DATASET, N_SAMPLES, RANDOM_SEED, get_model_label
from model_utils import get_device, load_model_and_tokenizer
from data_utils import load_gsm8k, load_mmlupro, load_strategyqa
from evaluation import evaluate_sample
from visualization import print_results_summary, plot_confidence_analysis, calibration_analysis
from save_utils import save_results
from config import DATASET



def main():
    # Setup
    device = get_device()
    
    # Load data
    if DATASET == "gsm8k":
        dataset = load_gsm8k()
    elif DATASET == "mmlupro":
        dataset = load_mmlupro()
    elif DATASET == "strategyqa":
        dataset = load_strategyqa()
    
    # Load model
    model, tokenizer = load_model_and_tokenizer()
    
    # Test on a single example
    print("\nTesting on a single example...\n")
    result = evaluate_sample(model, tokenizer, dataset, 0)
    
    print(f"Question: {result['question']}")
    print(f"Ground Truth: {result['ground_truth']}")
    print(f"Model Answer: {result['model_answer']}")
    print(f"Correct: {result['is_correct']}")
    print(f"\n--- Confidence Metrics ---")
    #Incorrect name. Old Code below 
    #print(f"Logit-based (mean prob): {result['logit_confidence_mean']:.4f}")
    
    print(f"Sequence log-prob: {result['seq_confidence_mean']:.4f}") # New one, says "logit_confidence_mean" but that's only because the name isn't changed but calculation is. Acc calculates the sequence log prob
    print(f"Logit-based (min prob):  {result['logit_confidence_min']:.4f}")
    print(f"Logit-based (geom mean): {result['logit_confidence_geom']:.4f}")
  #  print(f"Verbalized confidence:   {result['verbalized_confidence']}")
    
    # Run on sample of dataset
    print(f"\nRunning evaluation on {150} samples...")
    np.random.seed(RANDOM_SEED)
    sample_indices = np.random.choice(len(dataset), 150, replace=False)
    
    results = []
    for idx in tqdm(sample_indices, desc="Evaluating"):
        try:
            result = evaluate_sample(model, tokenizer, dataset, idx)
            results.append(result)
        except Exception as e:
            print(f"Error on sample {idx}: {e}")
            continue
    
    print(f"\nCompleted {len(results)} evaluations")
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Print summary
    print_results_summary(df)
    
    # Plot analysis
   # plot_confidence_analysis(df, valid_df)
    
    # Calibration analysis
   # calibration_analysis(valid_df)
  
    # Save results
    save_results(results, df)
    
    print("\nDone!")


if __name__ == "__main__":
    main()
