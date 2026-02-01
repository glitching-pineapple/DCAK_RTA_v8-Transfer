# confidence.py - Logit-based and verbalized confidence functions with CoT

import re
import torch
import numpy as np
from typing import Optional, Tuple, List, Dict
from config import MODEL_VARIANT, MAX_NEW_TOKENS


def generate_with_logits(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = MAX_NEW_TOKENS,
    temperature: float = 1.0,
    do_sample: bool = False,
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
            do_sample=do_sample,
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
        return {
            "min_prob": 0, 
            "geom_mean": 0, 
            "log_prob_sum": -float("inf"),
            "mean_prob": 0,
        }
    
    probs = np.array(token_probs, dtype=np.float64)
    
    return {
        "min_prob": float(np.min(probs)),
        "geom_mean": float(np.exp(np.mean(np.log(probs + 1e-10)))),
        "log_prob_sum": float(np.sum(np.log(probs + 1e-10))),
        "mean_prob": float(np.mean(probs)),
    }


def extract_verbalized_confidence(response: str, dataset: str) -> Optional[float]:
    """
    Extract verbalized confidence from the model's response.
    
    Returns confidence as decimal 0-1.
    Handles optional markdown bold (**Confidence:**).
    """
    # Handle optional markdown bold ** around "Confidence" and after ":"
    # Matches: Confidence: 0.85, **Confidence:** 0.72, Confidence:** 0.91
    pattern = r'\*{0,2}[Cc]onfidence\*{0,2}:\*{0,2}\s*(1\.?\d*|0?\.\d+|0)'
    
    match = re.search(pattern, response)
    if match:
        conf = float(match.group(1))
        return min(1.0, max(0.0, conf))
    
    return None


def extract_more_likely_than_not(response: str) -> Optional[bool]:
    """
    Extract the 'Correct' judgment from response.
    Handles optional markdown bold (**Correct:**).
    """
    # Handle optional markdown bold ** around keywords and after ":"
    patterns = [
        r'\*{0,2}[Cc]orrect\*{0,2}:\*{0,2}\s*(Yes|No)',
        r'\*{0,2}[Mm]ore\s+(?:likely\s+)?than\s+not\*{0,2}[:\s]*(Yes|No)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            answer = match.group(1).lower()
            return answer == 'yes'
    
    return None


def create_prompt(tokenizer, question: str, choices: list = None) -> str:
    """
    Create prompt with Chain-of-Thought reasoning and verbalized confidence.
    
    The prompt structure asks the model to:
    1. Think through the problem step by step
    2. Provide a final answer (JUST the value, no sentence)
    3. Rate confidence as precise decimal 0-1
    4. State if answer is more likely correct than not
    """
    from config import MODEL_VARIANT, DATASET
    
    if DATASET == "gsm8k":
        base_prompt = f"""Solve the following math problem. Think through it step by step, then provide your final answer and confidence.

Question: {question}

Instructions:
1. Show your reasoning step by step
2. State ONLY the final numerical answer after "Answer:" (just the number, no words)
3. Rate your confidence as a precise decimal from 0 to 1 (e.g., 0.72, 0.38, 0.91) after "Confidence:"
4. State if you think your answer is more likely correct than not after "Correct:" (Yes or No)

Example format:
Answer: 42
Confidence: 0.83
Correct: Yes

Solution:
Let me work through this step by step.
"""

    elif DATASET == "mmlupro":
        choices_text = "\n".join([f"{chr(65+i)}. {c}" for i, c in enumerate(choices)])
        base_prompt = f"""Answer the following multiple choice question. Think through it step by step, then provide your answer and confidence.

Question: {question}

{choices_text}

Instructions:
1. Analyze each option carefully
2. Explain your reasoning step by step
3. State ONLY the answer letter after "Answer:" (just the letter, e.g., A)
4. Rate your confidence as a precise decimal from 0 to 1 (e.g., 0.72, 0.38, 0.91) after "Confidence:"
5. State if you think your answer is more likely correct than not after "Correct:" (Yes or No)

Example format:
Answer: B
Confidence: 0.76
Correct: Yes

Solution:
Let me analyze each option step by step.
"""

    elif DATASET == "strategyqa":
        base_prompt = f"""Answer the following yes/no question. Think through it step by step, then provide your answer and confidence.

Question: {question}

Instructions:
1. Consider relevant facts and reasoning
2. Explain your thinking step by step
3. State ONLY your answer after "Answer:" (just Yes or No)
4. Rate your confidence as a precise decimal from 0 to 1 (e.g., 0.72, 0.38, 0.91) after "Confidence:"
5. State if you think your answer is more likely correct than not after "Correct:" (Yes or No)

Example format:
Answer: Yes
Confidence: 0.64
Correct: Yes

Solution:
Let me think through this step by step.
"""

    elif DATASET == "medqa":
        choices_text = "\n".join([f"{chr(65+i)}. {c}" for i, c in enumerate(choices)])
        base_prompt = f"""Answer the following medical question. Think through it step by step, then provide your answer and confidence.

Question: {question}

{choices_text}

Instructions:
1. Identify key clinical findings and relevant history
2. Consider the pathophysiology involved
3. Evaluate each answer choice against the clinical presentation
4. State ONLY the answer letter after "Answer:" (just the letter, e.g., A)
5. Rate your confidence as a precise decimal from 0 to 1 (e.g., 0.72, 0.38, 0.91) after "Confidence:"
6. State if you think your answer is more likely correct than not after "Correct:" (Yes or No)

Example format:
Answer: C
Confidence: 0.67
Correct: Yes

Solution:
Let me analyze this step by step.
"""

    elif DATASET == "triviaqa":
        base_prompt = f"""Answer the following trivia question. Think through it step by step, then provide your answer and confidence.

Question: {question}

Instructions:
1. Consider what you know about this topic
2. Think through related facts that might help
3. State ONLY your answer after "Answer:" (just the answer, no extra words)
4. Rate your confidence as a precise decimal from 0 to 1 (e.g., 0.72, 0.38, 0.91) after "Confidence:"
5. State if you think your answer is more likely correct than not after "Correct:" (Yes or No)

Example format:
Answer: Paris
Confidence: 0.84
Correct: Yes

Solution:
Let me think through this step by step.
"""
    
    else:
        base_prompt = f"""Answer the following question with step-by-step reasoning.

Question: {question}

Provide your reasoning, then:
Answer: [your answer, just the value]
Confidence: [decimal 0-1, e.g., 0.72]
Correct: [Yes/No]

Solution:
"""
    
    if MODEL_VARIANT == "instruct":
        messages = [{"role": "user", "content": base_prompt}]
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    else:
        return base_prompt


def create_simple_prompt(tokenizer, question: str, choices: list = None) -> str:
    """
    Create a simpler prompt for answer sampling (used in semantic entropy).
    Asks for JUST the answer to make extraction reliable.
    """
    from config import MODEL_VARIANT, DATASET
    
    if DATASET == "gsm8k":
        base_prompt = f"""Solve step by step, then give your final numerical answer.

Question: {question}

Think step by step, then write your final answer as JUST a number after "Answer:".
Solution:"""

    elif DATASET == "mmlupro":
        choices_text = "\n".join([f"{chr(65+i)}. {c}" for i, c in enumerate(choices)])
        base_prompt = f"""Question: {question}

{choices_text}

Think step by step, then write JUST the answer letter after "Answer:".
Solution:"""

    elif DATASET == "strategyqa":
        base_prompt = f"""Question: {question}

Think step by step, then write JUST Yes or No after "Answer:".
Solution:"""

    elif DATASET == "medqa":
        choices_text = "\n".join([f"{chr(65+i)}. {c}" for i, c in enumerate(choices)])
        base_prompt = f"""Question: {question}

{choices_text}

Think through the clinical presentation step by step, then write JUST the answer letter after "Answer:".
Solution:"""

    elif DATASET == "triviaqa":
        base_prompt = f"""Question: {question}

Think step by step, then write JUST the answer after "Answer:".
Solution:"""
    
    else:
        base_prompt = f"""Question: {question}

Think step by step, then write your answer after "Answer:".
Solution:"""
    
    if MODEL_VARIANT == "instruct":
        messages = [{"role": "user", "content": base_prompt}]
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    else:
        return base_prompt


def get_verbalized_confidence_separate(
    model,
    tokenizer,
    question: str,
    answer: str
) -> Optional[float]:
    """
    Ask the model separately how confident it is in its answer.
    This is a fallback if confidence isn't in the main response.
    
    Returns confidence as decimal 0-1.
    """
    confidence_prompt = f"""You solved the following problem:

Question: {question}

Your answer: {answer}

How confident are you that your answer is correct?
Respond with ONLY a precise decimal number between 0 and 1 (e.g., 0.72, 0.38, 0.91), nothing else."""
    
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
    
    match = re.search(r'(1\.?\d*|0?\.\d+|0)', response)
    if match:
        conf = float(match.group(1))
        return min(1.0, max(0.0, conf))
    return None
