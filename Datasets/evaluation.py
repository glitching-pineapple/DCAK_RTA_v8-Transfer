# evaluation.py - Sample evaluation with semantic entropy

from typing import Dict, Optional
from config import DATASET, SE_NUM_SAMPLES, SE_TEMPERATURE
from data_utils import extract_ground_truth, extract_model_answer, check_triviaqa_correct
from confidence import (
    generate_with_logits,
    compute_confidence_metrics,
    extract_verbalized_confidence,
    extract_more_likely_than_not,
    create_prompt,
    create_simple_prompt,
    get_verbalized_confidence_separate,
)


def evaluate_sample(
    model, 
    tokenizer, 
    dataset, 
    idx: int,
    semantic_calculator=None,
    compute_semantic_entropy: bool = True,
) -> Dict:
    """
    Evaluate a single sample with multiple confidence measures.
    
    Includes:
    - Logit-based confidence metrics
    - Verbalized confidence (from CoT response, 0-1 scale)
    - Semantic entropy (if calculator provided)
    """
    sample = dataset[idx]
    
    # Get question and choices based on dataset
    if DATASET == "gsm8k":
        question = sample['question']
        choices = None
    elif DATASET == "mmlupro":
        question = sample['question']
        choices = sample['options']
    elif DATASET == "strategyqa":
        question = sample['question']
        choices = None
    elif DATASET == "medqa":
        question = sample['question']
        # bigbio/med_qa may use 'options' or store choices differently
        choices = sample.get('options', sample.get('choices', []))
    elif DATASET == "triviaqa":
        question = sample['question']
        choices = None
    else:
        question = sample.get('question', str(sample))
        choices = sample.get('options', None)
    
    ground_truth = extract_ground_truth(sample, DATASET)
    
    # Generate main answer with CoT prompt (includes verbalized confidence)
    prompt = create_prompt(tokenizer, question, choices)
    response, token_probs, tokens = generate_with_logits(model, tokenizer, prompt)
    
    # Extract model's answer
    model_answer = extract_model_answer(response, DATASET)
    
    # Check correctness (TriviaQA uses fuzzy matching for multiple aliases)
    if DATASET == "triviaqa":
        is_correct = check_triviaqa_correct(model_answer, sample)
    else:
        is_correct = (model_answer == ground_truth) if model_answer else False
    
    # Compute logit-based confidence
    confidence_metrics = compute_confidence_metrics(token_probs)
    
    # Extract verbalized confidence from CoT response (0-1 scale)
    verbalized_conf = extract_verbalized_confidence(response, DATASET)
    
    # Extract "Correct" judgment
    more_likely = extract_more_likely_than_not(response)
    
    # If verbalized confidence not found in response, try separate query
    if verbalized_conf is None and model_answer:
        verbalized_conf = get_verbalized_confidence_separate(
            model, tokenizer, question, model_answer
        )
    
    # Build result dictionary
    result = {
        "idx": idx,
        "question": question[:200] + "..." if len(question) > 200 else question,
        "ground_truth": ground_truth,
        "model_answer": model_answer,
        "is_correct": is_correct,
        
        # Logit-based metrics
        "seq_confidence_mean": confidence_metrics["log_prob_sum"],
        "logit_confidence_min": confidence_metrics["min_prob"],
        "logit_confidence_geom": confidence_metrics["geom_mean"],
        "logit_confidence_mean_prob": confidence_metrics["mean_prob"],
        
        # Verbalized confidence (0-1 scale)
        "verbalized_confidence": verbalized_conf,
        "more_likely_than_not": more_likely,
        
        # Full response for inspection
        "full_response": response,
    }
    
    # Compute semantic entropy if calculator provided
    if compute_semantic_entropy and semantic_calculator is not None:
        se_results = compute_semantic_entropy_for_question(
            model, tokenizer, semantic_calculator,
            question, choices, DATASET
        )
        result.update({
            "semantic_entropy": se_results["semantic_entropy"],
            "predictive_entropy": se_results["predictive_entropy"],
            "predictive_entropy_normalized": se_results["predictive_entropy_normalized"],
            "num_semantic_clusters": se_results["num_clusters"],
            "cluster_sizes": se_results["cluster_sizes"],
            "sampled_answers": se_results.get("extracted_answers", []),
        })
    
    return result


def compute_semantic_entropy_for_question(
    model,
    tokenizer,
    semantic_calculator,
    question: str,
    choices: list,
    dataset: str,
) -> Dict:
    """
    Compute semantic entropy by sampling multiple answers.
    """
    from semantic_entropy import sample_answers_with_probs
    
    # Use simpler prompt for sampling
    prompt = create_simple_prompt(tokenizer, question, choices)
    
    # Sample multiple answers
    answers, log_probs, lengths = sample_answers_with_probs(
        model, tokenizer, prompt,
        num_samples=SE_NUM_SAMPLES,
        max_new_tokens=256,
        temperature=SE_TEMPERATURE,
    )
    
    # Extract just the answer portion from each response.
    # CRITICAL: Do NOT fall back to ans[:100] — that gives CoT prefixes
    # which all look similar, collapsing everything into 1 cluster (SE ≈ 0).
    # Instead, use a unique sentinel so failed extractions form their own clusters.
    extracted_answers = []
    extraction_failures = 0
    for i, ans in enumerate(answers):
        extracted = extract_model_answer(ans, dataset)
        if extracted:
            extracted_answers.append(extracted)
        else:
            # Use a unique sentinel per failed extraction so NLI doesn't
            # spuriously cluster them together
            extracted_answers.append(f"[EXTRACTION_FAILED_{i}]")
            extraction_failures += 1
    
    if extraction_failures > 0:
        import warnings
        warnings.warn(
            f"SE answer extraction failed for {extraction_failures}/{len(answers)} "
            f"samples on dataset={dataset}. These will form separate clusters."
        )
    
    # Compute semantic entropy
    se_results = semantic_calculator.compute_semantic_entropy(
        context=question,
        answers=extracted_answers,
        log_probs=log_probs,
        length_normalize=True,
        answer_lengths=lengths,
    )
    
    se_results["extracted_answers"] = extracted_answers
    se_results["raw_answers"] = answers
    
    return se_results


def evaluate_sample_quick(
    model, 
    tokenizer, 
    dataset, 
    idx: int,
) -> Dict:
    """
    Quick evaluation without semantic entropy (faster for debugging).
    """
    return evaluate_sample(
        model, tokenizer, dataset, idx,
        semantic_calculator=None,
        compute_semantic_entropy=False,
    )
