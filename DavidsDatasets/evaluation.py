# evaluation.py - Sample evaluation with semantic entropy

import re
from typing import Dict, Optional
from config import DATASET, SE_NUM_SAMPLES, SE_TEMPERATURE, COMPUTE_ANSWER_TOKEN_ENTROPY, MODEL_FAMILY
from data_utils import extract_ground_truth, extract_model_answer, extract_model_answer_strict, extract_reasoning, check_triviaqa_correct
from confidence import (
    generate_with_logits,
    compute_confidence_metrics,
    extract_answer_token_entropy,
    extract_verbalized_confidence,
    extract_more_likely_than_not,
    create_prompt,
    create_simple_prompt,
    get_verbalized_confidence_separate,
    get_two_pass_confidence,
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
    - Verbalized confidence (from CoT response, 1-10 scale)
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
        raw_options = sample.get('options', sample.get('choices', {}))
        if isinstance(raw_options, dict):
            choices = [raw_options[k] for k in sorted(raw_options.keys())]
        elif isinstance(raw_options, list):
            choices = raw_options
        else:
            choices = []
    elif DATASET == "triviaqa":
        question = sample['question']
        choices = None
    else:
        question = sample.get('question', str(sample))
        choices = sample.get('options', None)
    
    ground_truth = extract_ground_truth(sample, DATASET)
    #print ("David, " + ground_truth)
    
    # Generate main answer with CoT prompt (includes verbalized confidence)
    prompt = create_prompt(tokenizer, question, choices)
    response, token_probs, tokens, raw_scores = generate_with_logits(model, tokenizer, prompt)

    # Qwen3 emits <think>...</think> before the answer — strip it before any parsing
    if MODEL_FAMILY == "qwen3":
        response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()

    # Extract model's answer
    model_answer = extract_model_answer(response, DATASET)
    
    # Check correctness (TriviaQA uses fuzzy matching for multiple aliases)
    if DATASET == "triviaqa":
        is_correct = check_triviaqa_correct(model_answer, sample)
    else:
        is_correct = (model_answer == ground_truth) if model_answer else False
    
    # Compute logit-based confidence
    confidence_metrics = compute_confidence_metrics(token_probs)

    # Compute answer-token logit entropy for MCQ datasets (single forward pass)
    if COMPUTE_ANSWER_TOKEN_ENTROPY and DATASET in ("mmlupro", "medqa"):
        ate_results = extract_answer_token_entropy(tokens, raw_scores, tokenizer, DATASET)
    else:
        ate_results = {
            "answer_token_entropy": None,
            "answer_letter_probs": None,
            "top_answer_letter": None,
            "chosen_answer_raw_prob": None,
        }
    
    # Extract single-pass verbalized confidence from CoT response (1-10 scale)
    single_pass_conf = extract_verbalized_confidence(response, DATASET)
    
    # Extract "Correct" judgment from single-pass
    single_pass_correct = extract_more_likely_than_not(response)
    
    # If single-pass confidence not found in response, try separate query
    if single_pass_conf is None and model_answer:
        single_pass_conf = get_verbalized_confidence_separate(
            model, tokenizer, question, model_answer
        )
    
    # Two-pass confidence: separate critique-then-rate call
    two_pass_results = {"two_pass_confidence": None, "two_pass_correct": None, "two_pass_critique": ""}
    if model_answer:
        two_pass_results = get_two_pass_confidence(
            model, tokenizer, question, model_answer, response, choices
        )
    
    # Use two-pass as the primary verbalized confidence
    verbalized_conf = two_pass_results["two_pass_confidence"]
    more_likely = two_pass_results["two_pass_correct"]
    
    # Fall back to single-pass if two-pass extraction failed
    if verbalized_conf is None:
        verbalized_conf = single_pass_conf
    if more_likely is None:
        more_likely = single_pass_correct
    
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
        
        # Verbalized confidence (primary = two-pass, 1-10 scale)
        "verbalized_confidence": verbalized_conf,
        "more_likely_than_not": more_likely,
        
        # Single-pass confidence (for comparison)
        "single_pass_confidence": single_pass_conf,
        "single_pass_correct": single_pass_correct,
        
        # Two-pass critique (for inspection)
        "two_pass_critique": two_pass_results["two_pass_critique"],
        
        # Full response for inspection
        "full_response": response,

        # Answer-token logit entropy (MCQ only; None for gsm8k/strategyqa/triviaqa)
        "answer_token_entropy": ate_results["answer_token_entropy"],
        "answer_letter_probs": ate_results["answer_letter_probs"],
        "top_answer_letter": ate_results["top_answer_letter"],
        "chosen_answer_raw_prob": ate_results["chosen_answer_raw_prob"],
    }
    
    # Compute semantic entropy if calculator provided
    if compute_semantic_entropy and semantic_calculator is not None:
        se_results = compute_semantic_entropy_for_question(
            model, tokenizer, semantic_calculator,
            question, choices, DATASET
        )
        result.update({
            "semantic_entropy": se_results["semantic_entropy"],
            "semantic_entropy_answers": se_results["semantic_entropy_answers"],
            "predictive_entropy": se_results["predictive_entropy"],
            "predictive_entropy_normalized": se_results["predictive_entropy_normalized"],
            "num_semantic_clusters": se_results["num_clusters"],
            "num_answer_clusters": se_results["num_answer_clusters"],
            "cluster_sizes": se_results["cluster_sizes"],
            "sampled_answers": se_results.get("extracted_answers", []),
            "se_extraction_failure_rate": se_results.get("se_extraction_failure_rate", 0.0),
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
    # STRICT mode: only accept answers found via the "Answer:" line (Priority 1).
    # The fallback extractors (Priority 2/3) grab intermediate CoT numbers
    # (e.g., "3" from a computation step instead of the final "36"), which
    # inflates semantic cluster counts and corrupts SE.
    extracted_answers = []
    valid_raw_answers = []
    valid_log_probs = []
    valid_lengths = []
    extraction_failures = 0

    for i, ans in enumerate(answers):
        extracted = extract_model_answer_strict(ans, dataset)
        if extracted:
            extracted_answers.append(extracted)
            valid_raw_answers.append(ans)
            valid_log_probs.append(log_probs[i])
            valid_lengths.append(lengths[i])
        else:
            extraction_failures += 1
    
    se_extraction_failure_rate = extraction_failures / len(answers) if answers else 0.0
    
    if extraction_failures > 0:
        import warnings
        warnings.warn(
            f"SE strict extraction failed for {extraction_failures}/{len(answers)} "
            f"samples on dataset={dataset}. These are excluded from SE computation."
        )
    
    # Need at least 2 valid answers for meaningful entropy
    if len(extracted_answers) < 2:
        return {
            "semantic_entropy": float('inf'),
            "semantic_entropy_answers": float('inf'),
            "predictive_entropy": float('inf'),
            "predictive_entropy_normalized": float('inf'),
            "num_clusters": 0,
            "num_answer_clusters": 0,
            "cluster_sizes": [],
            "extracted_answers": extracted_answers,
            "raw_answers": answers,
            "se_extraction_failure_rate": se_extraction_failure_rate,
        }
    
    # Cluster on full CoT reasoning chains so that "A via different reasoning"
    # counts as a separate semantic cluster. Answer strings are clustered
    # separately for the secondary semantic_entropy_answers column.
    se_results = semantic_calculator.compute_semantic_entropy(
        context=question,
        answers=extracted_answers,
        log_probs=valid_log_probs,
        length_normalize=True,
        answer_lengths=valid_lengths,
        clustering_answers=valid_raw_answers,
    )

    se_results["extracted_answers"] = extracted_answers
    se_results["raw_answers"] = answers
    se_results["se_extraction_failure_rate"] = se_extraction_failure_rate
    
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