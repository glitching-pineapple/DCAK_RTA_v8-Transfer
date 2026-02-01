# visualization.py - Plotting and analysis including semantic entropy

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from config import MODEL_VARIANT, DATASET, get_model_label


def print_results_summary(df: pd.DataFrame):
    """Print comprehensive summary statistics."""
    print("=" * 60)
    print(f"RESULTS: {get_model_label()} on {DATASET.upper()}")
    print("=" * 60)
    
    # Overall accuracy
    accuracy = df['is_correct'].mean() * 100
    print(f"\nOverall Accuracy: {accuracy:.1f}%")
    print(f"Total samples: {len(df)}")
    
    # Logit-based confidence
    print(f"\n--- Logit-Based Confidence (Sequence Log-Prob) ---")
    print(f"Overall mean: {df['seq_confidence_mean'].mean():.4f}")
    print(f"Correct answers: {df[df['is_correct']]['seq_confidence_mean'].mean():.4f}")
    print(f"Wrong answers: {df[~df['is_correct']]['seq_confidence_mean'].mean():.4f}")
    
    # Verbalized confidence
    if 'verbalized_confidence' in df.columns:
        valid_verb = df[df['verbalized_confidence'].notna()]
        if len(valid_verb) > 0:
            print(f"\n--- Verbalized Confidence (0-1 scale) ---")
            print(f"Extraction rate: {len(valid_verb)/len(df)*100:.1f}%")
            print(f"Overall mean: {valid_verb['verbalized_confidence'].mean():.3f}")
            print(f"Correct answers: {valid_verb[valid_verb['is_correct']]['verbalized_confidence'].mean():.3f}")
            print(f"Wrong answers: {valid_verb[~valid_verb['is_correct']]['verbalized_confidence'].mean():.3f}")
    
    # More likely than not
    if 'more_likely_than_not' in df.columns:
        valid_mln = df[df['more_likely_than_not'].notna()]
        if len(valid_mln) > 0:
            print(f"\n--- 'More Likely Than Not' Judgment ---")
            print(f"Extraction rate: {len(valid_mln)/len(df)*100:.1f}%")
            mln_correct = valid_mln[valid_mln['more_likely_than_not'] == True]
            if len(mln_correct) > 0:
                print(f"When model says 'Yes': {mln_correct['is_correct'].mean()*100:.1f}% actually correct")
            mln_incorrect = valid_mln[valid_mln['more_likely_than_not'] == False]
            if len(mln_incorrect) > 0:
                print(f"When model says 'No': {mln_incorrect['is_correct'].mean()*100:.1f}% actually correct")
    
    # Semantic entropy
    if 'semantic_entropy' in df.columns:
        valid_se = df[df['semantic_entropy'].notna() & (df['semantic_entropy'] != float('inf'))]
        if len(valid_se) > 0:
            print(f"\n--- Semantic Entropy ---")
            print(f"Overall mean: {valid_se['semantic_entropy'].mean():.4f}")
            print(f"Correct answers: {valid_se[valid_se['is_correct']]['semantic_entropy'].mean():.4f}")
            print(f"Wrong answers: {valid_se[~valid_se['is_correct']]['semantic_entropy'].mean():.4f}")
            print(f"Avg semantic clusters: {valid_se['num_semantic_clusters'].mean():.2f}")
            
            # Predictive entropy comparison
            if 'predictive_entropy' in valid_se.columns:
                print(f"\n--- Predictive Entropy (Baseline) ---")
                print(f"Overall mean: {valid_se['predictive_entropy'].mean():.4f}")
                print(f"Correct answers: {valid_se[valid_se['is_correct']]['predictive_entropy'].mean():.4f}")
                print(f"Wrong answers: {valid_se[~valid_se['is_correct']]['predictive_entropy'].mean():.4f}")
    
    return df


def compute_auroc(df: pd.DataFrame, score_col: str, higher_is_better: bool = False) -> float:
    """
    Compute AUROC for uncertainty prediction.
    
    For uncertainty measures where higher = more uncertain = less likely correct,
    set higher_is_better=False.
    
    Args:
        df: DataFrame with 'is_correct' column
        score_col: Column name for the uncertainty/confidence score
        higher_is_better: If True, higher scores predict correctness
        
    Returns:
        AUROC score (0.5 = random, 1.0 = perfect)
    """
    from sklearn.metrics import roc_auc_score
    
    valid = df[df[score_col].notna() & ~df[score_col].isin([float('inf'), float('-inf')])]
    if len(valid) < 2 or valid['is_correct'].nunique() < 2:
        return float('nan')
    
    scores = valid[score_col].values
    if not higher_is_better:
        scores = -scores  # Flip so higher = more confident
    
    return roc_auc_score(valid['is_correct'].astype(int), scores)


def print_auroc_comparison(df: pd.DataFrame):
    """Print AUROC comparison for all uncertainty measures."""
    print("\n" + "=" * 60)
    print("UNCERTAINTY MEASURE COMPARISON (AUROC)")
    print("Higher AUROC = better at predicting correctness")
    print("=" * 60)
    
    metrics = [
        ("seq_confidence_mean", "Sequence Log-Prob", True),
        ("logit_confidence_geom", "Geometric Mean Prob", True),
        ("verbalized_confidence", "Verbalized Confidence", True),
        ("semantic_entropy", "Semantic Entropy", False),
        ("predictive_entropy", "Predictive Entropy", False),
        ("predictive_entropy_normalized", "Pred. Entropy (Normalized)", False),
    ]
    
    results = []
    for col, name, higher_better in metrics:
        if col in df.columns:
            auroc = compute_auroc(df, col, higher_better)
            if not np.isnan(auroc):
                results.append((name, auroc))
                print(f"{name:35s}: {auroc:.4f}")
    
    return results


def plot_confidence_analysis(df: pd.DataFrame, save_path: str = None):
    """Generate comprehensive confidence analysis plots."""
    
    # Determine which plots to include
    has_verbalized = 'verbalized_confidence' in df.columns and df['verbalized_confidence'].notna().any()
    has_semantic = 'semantic_entropy' in df.columns and df['semantic_entropy'].notna().any()
    
    n_plots = 2 + int(has_verbalized) + int(has_semantic)
    fig, axes = plt.subplots(1, n_plots, figsize=(5*n_plots, 4))
    if n_plots == 1:
        axes = [axes]
    
    plot_idx = 0
    
    # Plot 1: Logit confidence by correctness
    ax = axes[plot_idx]
    correct_conf = df[df['is_correct']]['logit_confidence_geom'].dropna()
    wrong_conf = df[~df['is_correct']]['logit_confidence_geom'].dropna()
    if len(correct_conf) > 0 and len(wrong_conf) > 0:
        ax.boxplot([correct_conf, wrong_conf], labels=['Correct', 'Wrong'])
        ax.set_ylabel('Geometric Mean Token Probability')
        ax.set_title('Logit-Based Confidence')
    plot_idx += 1
    
    # Plot 2: Verbalized confidence by correctness
    if has_verbalized:
        ax = axes[plot_idx]
        valid_df = df[df['verbalized_confidence'].notna()]
        correct_verb = valid_df[valid_df['is_correct']]['verbalized_confidence']
        wrong_verb = valid_df[~valid_df['is_correct']]['verbalized_confidence']
        if len(correct_verb) > 0 and len(wrong_verb) > 0:
            ax.boxplot([correct_verb, wrong_verb], labels=['Correct', 'Wrong'])
            ax.set_ylabel('Self-Reported Confidence (0-1)')
            ax.set_title('Verbalized Confidence')
        plot_idx += 1
    
    # Plot 3: Semantic entropy by correctness
    if has_semantic:
        ax = axes[plot_idx]
        valid_se = df[df['semantic_entropy'].notna() & (df['semantic_entropy'] != float('inf'))]
        correct_se = valid_se[valid_se['is_correct']]['semantic_entropy']
        wrong_se = valid_se[~valid_se['is_correct']]['semantic_entropy']
        if len(correct_se) > 0 and len(wrong_se) > 0:
            ax.boxplot([correct_se, wrong_se], labels=['Correct', 'Wrong'])
            ax.set_ylabel('Semantic Entropy')
            ax.set_title('Semantic Entropy\n(Lower = More Certain)')
        plot_idx += 1
    
    # Plot 4: Scatter plot comparing measures
    ax = axes[plot_idx]
    if has_semantic and has_verbalized:
        valid = df[df['semantic_entropy'].notna() & 
                   df['verbalized_confidence'].notna() &
                   (df['semantic_entropy'] != float('inf'))]
        colors = ['green' if c else 'red' for c in valid['is_correct']]
        ax.scatter(valid['semantic_entropy'], valid['verbalized_confidence'], 
                   c=colors, alpha=0.6, s=50)
        ax.set_xlabel('Semantic Entropy')
        ax.set_ylabel('Verbalized Confidence')
        ax.set_title('Semantic vs Verbalized\n(Green=Correct)')
    elif has_semantic:
        valid = df[df['semantic_entropy'].notna() & (df['semantic_entropy'] != float('inf'))]
        colors = ['green' if c else 'red' for c in valid['is_correct']]
        ax.scatter(valid['predictive_entropy'], valid['semantic_entropy'],
                   c=colors, alpha=0.6, s=50)
        ax.set_xlabel('Predictive Entropy')
        ax.set_ylabel('Semantic Entropy')
        ax.set_title('Predictive vs Semantic Entropy')
        ax.plot([0, ax.get_xlim()[1]], [0, ax.get_xlim()[1]], 'k--', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()
    return fig


def calibration_analysis(df: pd.DataFrame, confidence_col: str = 'verbalized_confidence'):
    """Analyze calibration of confidence measure."""
    print(f"\n--- Calibration Analysis ({confidence_col}) ---")
    
    valid_df = df[df[confidence_col].notna()].copy()
    if len(valid_df) == 0:
        print("No valid confidence values found.")
        return None
    
    # Create confidence bins (0-1 scale)
    bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    valid_df['conf_bin'] = pd.cut(valid_df[confidence_col], bins=bins)
    
    calibration = valid_df.groupby('conf_bin', observed=True).agg({
        'is_correct': ['mean', 'count']
    }).round(3)
    calibration.columns = ['Accuracy', 'Count']
    
    print(f"\n{confidence_col} Calibration:")
    print(calibration)
    
    # Compute Expected Calibration Error
    total = calibration['Count'].sum()
    if total > 0:
        bin_midpoints = [0.1, 0.3, 0.5, 0.7, 0.9]
        ece = 0
        for (_, row), midpoint in zip(calibration.iterrows(), bin_midpoints):
            if row['Count'] > 0:
                expected_acc = midpoint  # Already 0-1
                actual_acc = row['Accuracy']
                ece += (row['Count'] / total) * abs(actual_acc - expected_acc)
        print(f"\nExpected Calibration Error: {ece:.4f}")
    
    return calibration


def semantic_entropy_analysis(df: pd.DataFrame):
    """Detailed analysis of semantic entropy results."""
    if 'semantic_entropy' not in df.columns:
        print("No semantic entropy data available.")
        return
    
    valid = df[df['semantic_entropy'].notna() & (df['semantic_entropy'] != float('inf'))]
    if len(valid) == 0:
        print("No valid semantic entropy values.")
        return
    
    print("\n" + "=" * 60)
    print("SEMANTIC ENTROPY DETAILED ANALYSIS")
    print("=" * 60)
    
    # Cluster analysis
    print("\n--- Semantic Clustering Stats ---")
    print(f"Average clusters per question: {valid['num_semantic_clusters'].mean():.2f}")
    print(f"Max clusters: {valid['num_semantic_clusters'].max()}")
    print(f"Min clusters: {valid['num_semantic_clusters'].min()}")
    
    # Correlation with correctness
    print("\n--- Entropy by Cluster Count ---")
    for n in sorted(valid['num_semantic_clusters'].unique()):
        subset = valid[valid['num_semantic_clusters'] == n]
        if len(subset) > 0:
            acc = subset['is_correct'].mean() * 100
            print(f"  {n} clusters: {len(subset)} samples, {acc:.1f}% accuracy")
    
    # Compare SE vs PE
    if 'predictive_entropy' in valid.columns:
        from scipy import stats
        corr, p = stats.pearsonr(valid['semantic_entropy'], valid['predictive_entropy'])
        print(f"\nCorrelation (SE vs PE): {corr:.3f} (p={p:.4f})")
