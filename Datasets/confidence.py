# confidence.py - Logit-based and verbalized confidence functions

import re
import torch
import numpy as np
from typing import Optional, Tuple, List
from config import MODEL_VARIANT, MAX_NEW_TOKENS  # Keep as is, still need MODEL_VARIANT for chat template


def generate_with_logits(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = MAX_NEW_TOKENS,
    temperature: float = 1.0,
) -> Tuple[str, List[float], List[str]]:
    """
    Generate response and capture token-level probabilities.
    
    Returns:
        - generated_text: The model's response
        - token_probs: Probability of each generated token
        - tokens: The actual tokens generated
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_length = inputs.input_ids.shape[1]
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=False,
            return_dict_in_generate=True,
            output_scores=True,
            pad_token_id=tokenizer.pad_token_id,
        )
    
    # Extract generated tokens (excluding prompt)
    generated_ids = outputs.sequences[0, input_length:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    # Calculate probabilities for each generated token
    token_probs = []
    tokens = []
    
    for i, score in enumerate(outputs.scores):
        probs = torch.softmax(score[0], dim=-1)
        token_id = generated_ids[i].item()
        token_prob = probs[token_id].item()
        token_probs.append(token_prob)
        tokens.append(tokenizer.decode([token_id]))
    
    return generated_text, token_probs, tokens


def compute_confidence_metrics(token_probs: List[float]) -> dict:
    """Compute various confidence metrics from token probabilities."""
    if not token_probs:
        return { # Remove this "mean_prob": 0, 
            "min_prob": 0, 
            "geom_mean": 0, 
            
            # Old one "log_prob_sum": 0
            "log_prob_sum": -float("inf")} #Added cause apparently the above one makes an empty output look perfectly confident
    
    # Old: probs = np.array(token_probs)
    probs = np.array(token_probs, dtype=np.float64) #Added the dtype for precision
    
    return {
        # Remove this "mean_prob": float(np.mean(probs)),
        "min_prob": float(np.min(probs)),
        "geom_mean": float(np.exp(np.mean(np.log(probs + 1e-10)))),
        # Do we need this still? "sequence_prob": float(np.prod(probs)),
        "log_prob_sum": float(np.sum(np.log(probs + 1e-10))),
    }


def get_verbalized_confidence(
    model,
    tokenizer,
    question: str,
    answer: str
) -> Optional[float]:
    """
    Ask the model how confident it is in its answer.
    Returns a confidence score between 0-100.
    """
    confidence_prompt = f"""You solved the following math problem:

Question: {question}

Your answer: {answer}

On a scale from 0 to 10, how confident are you that your answer is correct? 
Respond with ONLY a number between 0 and 10, nothing else."""
    
    if MODEL_VARIANT == "instruct":
        messages = [{"role": "user", "content": confidence_prompt}]
        formatted_prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    else:
        formatted_prompt = confidence_prompt + "\n\nConfidence:"
    
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=10,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    
    response = tokenizer.decode(
        outputs[0, inputs.input_ids.shape[1]:], 
        skip_special_tokens=True
    ).strip()
    
    match = re.search(r'(\d+(?:\.\d+)?)', response)
    if match:
        conf = float(match.group(1))
        return min(100, max(0, conf))
    return None


def create_prompt(tokenizer, question: str, choices: list = None) -> str:
    """Create prompt based on dataset type."""
    from config import MODEL_VARIANT, DATASET
    
    if DATASET == "gsm8k":
        base_prompt = f"""Solve the following math problem step by step. 
At the end, provide your final answer as a single number after "The answer is: ".

Question: {question}

Solution:"""

    elif DATASET == "mmlupro":
        # Format choices as A, B, C, D...
        choices_text = "\n".join([f"{chr(65+i)}. {c}" for i, c in enumerate(choices)])
        base_prompt = f"""

Question: {question}

{choices_text}
First, provide your answer (A, B, C, D, etc.).
Then, rate your confidence on a scale from 1 to 10
Next, determine if you think it is more likely than not that your answer is correct
Format your response as:
Answer: [LETTER]
Confidence: [NUMBER 1-10]
More than not: Yes/No

"""

    elif DATASET == "strategyqa":
        base_prompt = f"""Answer the following yes/no question. Think step by step, then provide your final answer as "Yes" or "No" after "The answer is: ".

Question: {question}

Solution:"""
    
    if MODEL_VARIANT == "instruct":
        messages = [{"role": "user", "content": base_prompt}]
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    else:
        return base_prompt
