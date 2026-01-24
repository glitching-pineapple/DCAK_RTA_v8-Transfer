# visualization.py - Plotting and analysis

import pandas as pd
import matplotlib.pyplot as plt
from config import MODEL_VARIANT, DATASET, get_model_label  # Update import

def print_results_summary(df: pd.DataFrame):
    """Print summary statistics."""
    print("=" * 50)
    print(f"RESULTS: {get_model_label()} on {DATASET.upper()}")  # Use get_model_label()
    print("=" * 50)
    
    # Overall accuracy
    accuracy = df['is_correct'].mean() * 100
    print(f"\nAccuracy: {accuracy:.1f}%")
    
    # Confidence statistics
    print(f"\n--- Logit-Based Confidence (Mean Prob) ---")
    print(f"Overall mean: {df['seq_confidence_mean'].mean():.4f}")
    print(f"Correct answers: {df[df['is_correct']]['seq_confidence_mean'].mean():.4f}")
    print(f"Wrong answers: {df[~df['is_correct']]['seq_confidence_mean'].mean():.4f}")
    
 #   print(f"\n--- Verbalized Confidence ---")
    #valid_verb = df[df['verbalized_confidence'].notna()]
  #  print(f"Overall mean: {valid_verb['verbalized_confidence'].mean():.1f}")
  #  print(f"Correct answers: {valid_verb[valid_verb['is_correct']]['verbalized_confidence'].mean():.1f}")
  #  print(f"Wrong answers: {valid_verb[~valid_verb['is_correct']]['verbalized_confidence'].mean():.1f}")
    
 #  return valid_verb


def plot_confidence_analysis(df: pd.DataFrame, valid_df: pd.DataFrame):
    """Generate confidence analysis plots."""
    # Calculate correlation
    valid_df = valid_df.copy()
    valid_df['logit_scaled'] = valid_df['seq_confidence_mean'] * 100
    correlation = valid_df['logit_scaled'].corr(valid_df['verbalized_confidence'])
    print(f"\nCorrelation between logit and verbalized confidence: {correlation:.3f}")
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Plot 1: Logit confidence by correctness
    ax1 = axes[0]
    correct_conf = df[df['is_correct']]['seq_confidence_mean']
    wrong_conf = df[~df['is_correct']]['seq_confidence_mean']
    ax1.boxplot([correct_conf, wrong_conf], labels=['Correct', 'Wrong'])
    ax1.set_ylabel('Mean Token Probability')
    ax1.set_title('Logit-Based Confidence')
    
    # Plot 2: Verbalized confidence by correctness
    ax2 = axes[1]
    correct_verb = valid_df[valid_df['is_correct']]['verbalized_confidence']
    wrong_verb = valid_df[~valid_df['is_correct']]['verbalized_confidence']
    ax2.boxplot([correct_verb, wrong_verb], labels=['Correct', 'Wrong'])
    ax2.set_ylabel('Self-Reported Confidence (0-100)')
    ax2.set_title('Verbalized Confidence')
    
    # Plot 3: Scatter plot of both confidence types
    ax3 = axes[2]
    colors = ['green' if c else 'red' for c in valid_df['is_correct']]
    ax3.scatter(valid_df['logit_scaled'], valid_df['verbalized_confidence'], 
                c=colors, alpha=0.6)
    ax3.set_xlabel('Logit Confidence (scaled 0-100)')
    ax3.set_ylabel('Verbalized Confidence')
    ax3.set_title(f'Correlation: {correlation:.3f}')
    ax3.plot([0, 100], [0, 100], 'k--', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'confidence_analysis_{MODEL_VARIANT}.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return correlation


def calibration_analysis(valid_df: pd.DataFrame):
    """Analyze calibration of verbalized confidence."""
    print("\n--- Calibration Analysis ---")
    
    valid_df = valid_df.copy()
    bins = [0, 20, 40, 60, 80, 100]
    valid_df['conf_bin'] = pd.cut(valid_df['verbalized_confidence'], bins=bins)
    
    calibration = valid_df.groupby('conf_bin').agg({
        'is_correct': ['mean', 'count']
    }).round(3)
    calibration.columns = ['Accuracy', 'Count']
    print("\nVerbalized Confidence Calibration:")
    print(calibration)
    
    return calibration
