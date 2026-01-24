from config import DATASET
from data_utils import extract_ground_truth, extract_model_answer
from confidence import (
    generate_with_logits,
    compute_confidence_metrics,
    get_verbalized_confidence,
    create_prompt
)

def evaluate_sample(model, tokenizer, dataset, idx: int) -> dict:
    """Evaluate a single sample."""
    sample = dataset[idx]
    
    # Get question and choices based on dataset
    if DATASET == "gsm8k":
        question = sample['question']
        choices = None
    elif DATASET == "mmlupro":
        question = sample['question']
        choices = sample['options']  # List of choices
    elif DATASET == "strategyqa":
        question = sample['question']
        choices = None
    
    ground_truth = extract_ground_truth(sample, DATASET)
    
    # Generate answer with logits
    prompt = create_prompt(tokenizer, question, choices)
    response, token_probs, tokens = generate_with_logits(model, tokenizer, prompt)
    
    # Extract model's answer
    model_answer = extract_model_answer(response, DATASET)
    is_correct = (model_answer == ground_truth) if model_answer else False
    
    # Compute logit-based confidence
    confidence_metrics = compute_confidence_metrics(token_probs)
    
    # Get verbalized confidence
   
    
    return {
        "idx": idx,
        "question": question[:100] + "...",
        "ground_truth": ground_truth,
        "model_answer": model_answer,
        "is_correct": is_correct,
        
        "seq_confidence_mean": confidence_metrics["log_prob_sum"], #New above says confidence mean but acc contains sequence log prob just kept name for compatibility w old code for now
        "logit_confidence_min": confidence_metrics["min_prob"],
        "logit_confidence_geom": confidence_metrics["geom_mean"],
        #"verbalized_confidence": verbalized_conf,
        "full_response": response,
    }