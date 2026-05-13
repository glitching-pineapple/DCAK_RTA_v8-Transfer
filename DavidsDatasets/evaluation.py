# evaluation.py - Sample evaluation with semantic entropy

import math
import re
from typing import Dict, Optional

_QWEN3_THINK_RE = re.compile(r'<think>.*?</think>', re.DOTALL)
from config import DATASET, SE_NUM_SAMPLES, SE_TEMPERATURE, SE_MAX_NEW_TOKENS, COMPUTE_ANSWER_TOKEN_ENTROPY, MODEL_FAMILY, SKIP_NLI_CLUSTERING
from data_utils import extract_ground_truth, extract_model_answer, extract_model_answer_strict, extract_reasoning, check_triviaqa_correct, answers_match
from confidence import (
    generate_with_logits,
    compute_confidence_metrics,
    extract_answer_token_entropy,
    extract_verbalized_confidence,
    extract_more_likely_than_not,
    create_prompt,
    create_simple_prompt,
    get_verbalized_confidence_separate,
    get_gen2_confidence,
    get_two_pass_confidence,
    get_forced_answer,
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
    elif DATASET == "legalbench":
        # LegalBench subtasks store the prompt in `text`; some variants may
        # also expose `question`. Fall back so different subtasks still load.
        question = sample.get('text', sample.get('question', str(sample)))
        choices = None
    else:
        question = sample.get('question', str(sample))
        choices = sample.get('options', None)
    
    ground_truth = extract_ground_truth(sample, DATASET)

    token_probs: list = []  # populated by whichever branch runs below
    _empty_two_pass = {
        "two_pass_confidence": None,
        "two_pass_correct": None,
        "two_pass_critique": "",
        "two_pass_finish_reason": None,
        "two_pass_was_truncated": None,
    }
    was_forced = False
    forced_response = None
    if MODEL_FAMILY in ("qwen3", "gemma4", "gptoss"):
        # --- qwen3 three-generation flow ---
        # Gen 1: reasoning + final answer only (no confidence rubric in prompt)
        prompt = create_prompt(tokenizer, question, choices, include_confidence=False)
        response_raw, token_probs, tokens, raw_scores, main_meta = generate_with_logits(model, tokenizer, prompt)
        # Stripped form: <think>...</think> removed. Used for answer-letter
        # extraction so the regex doesn't match a draft answer the model
        # wrote inside its scratchpad.
        response = _QWEN3_THINK_RE.sub('', response_raw).strip()
        # The actual chain of thought lives inside <think>...</think>. The
        # critic and self-rater need it — without it, critics confabulate
        # on easy questions and abdicate on hard ones. Pull the think
        # content out and pass that as the reasoning to evaluate. Fall back
        # to the stripped response if no think block was emitted.
        _think_match = re.search(r'<think>(.*?)</think>', response_raw, re.DOTALL)
        reasoning_for_critique = (
            _think_match.group(1).strip() if _think_match else response
        )

        # Prefer the strict "Answer:" line extractor. If the main response
        # committed via a clean Answer line, trust it — even when the EOS
        # token wasn't emitted before max_new_tokens (very common for Gemma4
        # at 8192-token budgets). Only fall back to forcing when the model
        # never produced a clean Answer line. was_forced reflects "did the
        # main response fail to commit?", not "did EOS happen?".
        strict_answer = extract_model_answer_strict(response, DATASET)
        if strict_answer is not None:
            model_answer = strict_answer
        else:
            was_forced = True
            forced_answer, forced_response = get_forced_answer(
                model, tokenizer, question, response, DATASET, choices
            )
            if forced_answer is not None:
                model_answer = forced_answer
            else:
                model_answer = extract_model_answer(response, DATASET)

        # Gen 2: model told it's its own work; returns verbalized confidence + MLN
        single_pass_conf = None
        single_pass_correct = None
        if model_answer:
            gen2 = get_gen2_confidence(
                model, tokenizer, question, reasoning_for_critique, model_answer, choices
            )
            single_pass_conf = gen2["gen2_confidence"]
            single_pass_correct = gen2["gen2_correct"]

        # Gen 3: blinded two-pass critique; includes Gen 2 scores as context
        two_pass_results = dict(_empty_two_pass)
        if model_answer:
            two_pass_results = get_two_pass_confidence(
                model, tokenizer, question, model_answer, reasoning_for_critique, choices,
                gen2_confidence=single_pass_conf,
                gen2_correct=single_pass_correct,
            )
    else:
        # --- standard single-pass flow for all other model families ---
        prompt = create_prompt(tokenizer, question, choices, include_confidence=True)
        response, token_probs, tokens, raw_scores, main_meta = generate_with_logits(model, tokenizer, prompt)

        # Same Priority-1-driven gate as the reasoning-model branch: trust
        # the main response only when it produced a clean Answer line; else
        # force. was_forced flags rows where the main response failed to
        # commit, regardless of whether the forced call itself succeeded.
        strict_answer = extract_model_answer_strict(response, DATASET)
        if strict_answer is not None:
            model_answer = strict_answer
        else:
            was_forced = True
            forced_answer, forced_response = get_forced_answer(
                model, tokenizer, question, response, DATASET, choices
            )
            if forced_answer is not None:
                model_answer = forced_answer
            else:
                model_answer = extract_model_answer(response, DATASET)

        single_pass_conf = extract_verbalized_confidence(response, DATASET)
        single_pass_correct = extract_more_likely_than_not(response)

        if single_pass_conf is None and model_answer:
            single_pass_conf = get_verbalized_confidence_separate(
                model, tokenizer, question, model_answer
            )

        two_pass_results = dict(_empty_two_pass)
        if model_answer:
            two_pass_results = get_two_pass_confidence(
                model, tokenizer, question, model_answer, response, choices
            )

    # Check correctness via the per-dataset comparator. answers_match handles:
    # - gsm8k numeric equivalence ("6.00" == "6", "1,000" == "1000")
    # - mmlupro/medqa case + wrapper normalization ("(A)" == "A", "a" == "A")
    # - strategyqa case + trailing punctuation ("No." == "No")
    # - triviaqa alias-aware fuzzy match (delegates to check_triviaqa_correct)
    is_correct = answers_match(model_answer, ground_truth, DATASET, sample)

    # Logit-based confidence (computed from Gen 1 token probabilities)
    confidence_metrics = compute_confidence_metrics(token_probs)

    # Answer-token logit entropy for MCQ datasets (single forward pass)
    if COMPUTE_ANSWER_TOKEN_ENTROPY and DATASET in ("mmlupro", "medqa"):
        ate_results = extract_answer_token_entropy(tokens, raw_scores, tokenizer, DATASET)
    else:
        ate_results = {
            "answer_token_entropy": None,
            "answer_letter_probs": None,
            "top_answer_letter": None,
            "chosen_letter": None,
            "chosen_answer_raw_prob": None,
        }

    # Primary verbalized confidence = two-pass; fall back to single-pass (Gen 2 for qwen3)
    verbalized_conf = two_pass_results["two_pass_confidence"]
    more_likely = two_pass_results["two_pass_correct"]
    if verbalized_conf is None:
        verbalized_conf = single_pass_conf
    if more_likely is None:
        more_likely = single_pass_correct

    # Hard-failure policy for rows where the model never produced a parseable
    # answer letter (typically: math questions where the CoT exhausts the
    # token budget mid-derivation, or where the final value doesn't map to
    # any MCQ option). The row is automatically incorrect — no Gen 2/Gen 3
    # was run (gated above by `if model_answer:`), and we explicitly NaN out
    # every verbalized-confidence field so partial signals don't pollute
    # downstream calibration analysis. The `answer_extraction_failed` column
    # makes these rows trivially filterable.
    answer_extraction_failed = (model_answer is None) or (model_answer == "")
    if answer_extraction_failed:
        is_correct = False
        verbalized_conf = math.nan
        more_likely = None
        single_pass_conf = math.nan
        single_pass_correct = None
    
    # Build result dictionary
    result = {
        "idx": idx,
        "question": question[:200] + "..." if len(question) > 200 else question,
        "ground_truth": ground_truth,
        "model_answer": model_answer,
        "is_correct": is_correct,
        # True when the model failed to produce a parseable answer letter
        # (typically math CoT that ran out of tokens). Rows with this flag
        # have NaN verbalized_confidence/single_pass_confidence and should
        # be excluded from calibration analyses, not treated as low-confidence
        # data points.
        "answer_extraction_failed": answer_extraction_failed,
        
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
        "two_pass_finish_reason": two_pass_results["two_pass_finish_reason"],
        "two_pass_was_truncated": two_pass_results["two_pass_was_truncated"],

        # Main-pass generation diagnostics (truncation here means the CoT
        # itself was cut off, not just the critique)
        "main_pass_finish_reason": main_meta["finish_reason"],
        "main_pass_was_truncated": main_meta["was_truncated"],
        "was_forced": was_forced,
        "forced_answer_response": forced_response,

        # Full response for inspection
        "full_response": response,

        # Answer-token logit entropy (MCQ only; None for gsm8k/strategyqa/triviaqa)
        "answer_token_entropy": ate_results["answer_token_entropy"],
        "answer_letter_probs": ate_results["answer_letter_probs"],
        "top_answer_letter": ate_results["top_answer_letter"],
        # The letter actually emitted in the response text. Disagreement with
        # top_answer_letter signals a tokenizer/letter-id mapping issue.
        "chosen_letter": ate_results["chosen_letter"],
        "chosen_answer_raw_prob": ate_results["chosen_answer_raw_prob"],
    }
    
    # Compute semantic entropy unless NLI clustering is disabled or no calculator provided
    if compute_semantic_entropy and semantic_calculator is not None and not SKIP_NLI_CLUSTERING:
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
        max_new_tokens=SE_MAX_NEW_TOKENS,
        temperature=SE_TEMPERATURE,
    )

    # Strip Qwen3 <think>...</think> blocks before extraction
    if MODEL_FAMILY in ("qwen3", "gemma4", "gptoss"):
        answers = [_QWEN3_THINK_RE.sub('', a).strip() for a in answers]

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
            "semantic_entropy": float('nan'),
            "semantic_entropy_answers": float('nan'),
            "predictive_entropy": float('nan'),
            "predictive_entropy_normalized": float('nan'),
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