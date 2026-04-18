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
) -> Tuple[str, List[float], List[str], list]:
    """
    Generate response and capture token-level probabilities.

    Returns:
        - generated_text: The model's response
        - token_probs: Probability of each generated token
        - tokens: The actual tokens generated
        - raw_scores: Per-step vocab logit tensors, shape (1, vocab_size) each
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

    generated_ids = outputs.sequences[0, input_length:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    # Keep raw per-step distributions for answer token entropy
    raw_scores = list(outputs.scores)

    token_probs = []
    tokens = []

    for i, score in enumerate(outputs.scores):
        probs = torch.softmax(score[0], dim=-1)
        token_id = generated_ids[i].item()
        token_prob = probs[token_id].item()
        token_probs.append(token_prob)
        tokens.append(tokenizer.decode([token_id]))

    return generated_text, token_probs, tokens, raw_scores


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


def extract_answer_token_entropy(
    tokens: List[str],
    raw_scores: list,
    tokenizer,
    dataset: str,
) -> Dict:
    """
    Compute Shannon entropy over answer-letter logits at the answer decision token.

    At the exact position where the model emits the answer letter (A/B/C/…),
    we read the full vocab distribution, extract probabilities for valid answer
    letters, renormalize, and compute entropy.  Requires only the single forward
    pass already performed by generate_with_logits.

    Returns dict with keys:
        answer_token_entropy  – float (nats); nan if answer position not found
        answer_letter_probs   – {letter: renorm_prob} or None
        top_answer_letter     – str or None
        chosen_answer_raw_prob – raw (pre-renorm) prob of the emitted letter
    """
    MCQ_DATASETS = {"mmlupro", "medqa"}
    null_result = {
        "answer_token_entropy": None,
        "answer_letter_probs": None,
        "top_answer_letter": None,
        "chosen_answer_raw_prob": None,
    }

    if dataset not in MCQ_DATASETS:
        return null_result

    valid_letters = list("ABCDEFGHIJ") if dataset == "mmlupro" else list("ABCDE")

    # Build letter -> set of token IDs mapping.
    # Some tokenizers encode "A" and " A" as different IDs; we collect both.
    letter_to_token_ids: Dict[str, set] = {}
    for letter in valid_letters:
        ids: set = set()
        for form in (letter, f" {letter}"):
            encoded = tokenizer.encode(form, add_special_tokens=False)
            if len(encoded) == 1:
                ids.add(encoded[0])
        letter_to_token_ids[letter] = ids

    # Locate the last "Answer:" marker in the token stream, then find the
    # first answer-letter token that follows it.
    accumulated = ""
    marker_end_char = None
    import re as _re
    for tok in tokens:
        accumulated += tok
        m = _re_answer_marker.search(accumulated)
        if m:
            marker_end_char = m.end()  # keep updating → last occurrence

    if marker_end_char is None:
        return {**null_result, "answer_token_entropy": float("nan")}

    # Walk tokens to find which index starts at/after marker_end_char
    char_count = 0
    search_from = len(tokens)  # fallback: no letter found
    for pos, tok in enumerate(tokens):
        char_count += len(tok)
        if char_count >= marker_end_char:
            search_from = pos + 1
            break

    answer_pos = None
    chosen_letter = None
    for pos in range(search_from, len(tokens)):
        tok_clean = tokens[pos].strip().upper()
        if tok_clean in valid_letters:
            answer_pos = pos
            chosen_letter = tok_clean
            break
        # Stop if a non-whitespace, non-trivial token intervenes
        if tokens[pos].strip() and tokens[pos].strip() not in (":", "-", ".", "*", "\n", "#"):
            break

    if answer_pos is None or answer_pos >= len(raw_scores):
        return {**null_result, "answer_token_entropy": float("nan")}

    # Softmax over full vocab at the answer decision step
    logits = raw_scores[answer_pos]
    if logits.dim() == 2:
        logits = logits[0]  # (1, vocab_size) -> (vocab_size,)
    probs = torch.softmax(logits.float(), dim=-1)

    # Extract and sum probabilities for each valid letter (all token-ID forms)
    letter_probs: Dict[str, float] = {}
    for letter in valid_letters:
        p = sum(probs[tid].item() for tid in letter_to_token_ids[letter] if tid < probs.shape[0])
        letter_probs[letter] = p

    chosen_answer_raw_prob = letter_probs.get(chosen_letter, 0.0)

    total = sum(letter_probs.values())
    if total <= 0:
        return {**null_result, "answer_token_entropy": float("nan")}

    renorm = {l: p / total for l, p in letter_probs.items()}
    entropy = float(max(0.0, -sum(p * np.log(p + 1e-10) for p in renorm.values())))
    top_letter = max(renorm, key=renorm.get)

    return {
        "answer_token_entropy": entropy,
        "answer_letter_probs": {l: round(renorm[l], 4) for l in valid_letters},
        "top_answer_letter": top_letter,
        "chosen_answer_raw_prob": round(chosen_answer_raw_prob, 6),
    }


# Pre-compiled pattern reused across calls
_re_answer_marker = re.compile(r"[Aa]nswer\s*:", re.IGNORECASE)


def extract_verbalized_confidence(response: str, dataset: str) -> Optional[float]:
    """
    Extract verbalized confidence from the model's response.
    
    Returns confidence as integer 1-10.
    Handles:
    - Standard integer: "Confidence: 7"
    - With /10: "Confidence: 8/10"
    - Markdown bold: "**Confidence:** 9"
    - Approximate language: "Confidence: about 6"
    - Space around colon: "Confidence : 8"
    - Legacy decimal format: "Confidence: 0.85" → auto-converted to 1-10
    - Legacy percentage: "Confidence: 85%" → auto-converted to 1-10
    """
    # Strip markdown bold for easier matching
    cleaned = response.replace('*', '')
    
    # Primary pattern: "Confidence" followed by optional filler then a number
    # Handles: "Confidence: 8", "Confidence: 7/10", "Confidence: about 9"
    pattern = r'[Cc]onfidence\s*:\s*(?:approximately|about|around|~|roughly)?\s*(\d+(?:\.\d+)?)\s*(?:/10|%)?'
    
    matches = re.findall(pattern, cleaned)
    if matches:
        conf = float(matches[-1])  # Take the LAST match
        # Normalize legacy formats to 1-10 scale
        if conf > 10:
            conf = conf / 10.0
        elif conf <= 1.0 and '.' in str(matches[-1]):
            conf = conf * 10.0
        return min(10.0, max(1.0, round(conf)))
    
    return None


def extract_more_likely_than_not(response: str) -> Optional[bool]:
    """
    Extract the 'Correct' judgment from response.
    Handles optional markdown bold (**Correct:**).
    """
    # Handle optional markdown bold ** around keywords and after ":"
    patterns = [
        r'^\s*\*{0,2}[Cc]orrect\*{0,2}:\*{0,2}\s*(Yes|No)',
        r'^\s*\*{0,2}[Mm]ore\s+(?:likely\s+)?than\s+not\*{0,2}[:\s]*(Yes|No)',
    ]
    
    last_match = None
    for pattern in patterns:
        matches = re.findall(pattern, response, re.IGNORECASE | re.MULTILINE)
        if matches:
            last_match = matches[-1]
    if last_match is not None:
        return last_match.lower() == 'yes'
    return None


def create_prompt(tokenizer, question: str, choices: list = None) -> str:
    """
    Create prompt with Chain-of-Thought reasoning and verbalized confidence.
    
    The prompt structure asks the model to:
    1. Think through the problem step by step
    2. Provide a final answer (JUST the value, no sentence)
    3. Rate confidence as a decimal from 1-10 (Ex. 7.2)
    4. State if answer is more likely correct than not
    """
    from config import MODEL_VARIANT, DATASET

#for confidence #commented out to test rubric based verbalized conf. evaluation

    
#     if DATASET == "gsm8k":
#         base_prompt = f"""Solve the following math problem. Think through it step by step, then provide your final answer and confidence.

# Question: {question}

# Instructions:
# 1. Show your reasoning step by step
# 2. State ONLY the final numerical answer after "Answer:" (just the number, no words)
# 3. Rate your confidence as an integer from 1 to 10 (where 1 = very uncertain, 10 = very certain) after "Confidence:"
# 4. State if you think your answer is more likely correct than not after "Correct:" (Yes or No)

# Example format:
# Answer: 42
# Confidence: 8
# Correct: Yes

# Solution:
# Let me work through this step by step.
# """

    if DATASET == "gsm8k":
        base_prompt = f"""Solve the following math problem. Think through it step by step, then provide your final answer and confidence.

Question: {question}

First, explicitly reason through the question step by step to arrive at an answer. Then, evaluate how likely your answer is to be correct by selecting EXACTLY ONE of these confidence classes:

1 = "Almost no chance" (0-10% likely correct)
2 = "Highly unlikely" (10-20% likely correct)
3 = "Chances are slight" (20-30% likely correct)
4 = "Unlikely" (30-40% likely correct)
5 = "Less than even" (40-50% likely correct)
6 = "Better than even" (50-60% likely correct)
7 = "Likely" (60-70% likely correct)
8 = "Very good chance" (70-80% likely correct)
9 = "Highly likely" (80-90% likely correct)
10 = "Almost certain" (90-100% likely correct)

After your step-by-step solution, you MUST end with EXACTLY this format (replace <YOUR_NUMBER> with your computed answer):
Answer: <YOUR_NUMBER>
Confidence: <1-10>
Correct: Yes or No

The Confidence number MUST match the class you selected — for example, if you select "Better than even" you MUST write Confidence: 6, not any other number.

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
4. Rate your confidence as an integer from 1 to 10 (where 1 = very uncertain, 10 = very certain) after "Confidence:"
5. State if you think your answer is more likely correct than not after "Correct:" (Yes or No)

Example format:
Answer: B
Confidence: 7
Correct: Yes

Solution:
Let me analyze each option step by step.
"""

    elif DATASET == "strategyqa":
        base_prompt = f"""Answer the following yes/no question. Think through it step by step, then provide your answer and confidence.

Question: {question}

First, consider relevant facts and reasoning, and explain your thinking step by step to arrive at an answer. Then, thoroughly assess your confidence in that answer by evaluating your thinking process so far. Finally, classify your confidence into one of the following classes based on how likely your answer is to be correct:

- "Almost no chance" (0.0–0.1)
- "Highly unlikely" (0.1–0.2)
- "Chances are slight" (0.2–0.3)
- "Unlikely" (0.3–0.4)
- "Less than even" (0.4–0.5)
- "Better than even" (0.5–0.6)
- "Likely" (0.6–0.7)
- "Very good chance" (0.7–0.8)
- "Highly likely" (0.8–0.9)
- "Almost certain" (0.9–1.0)

Each category reflects the probability that your answer is correct. After your reasoning, you MUST end your response with EXACTLY these three lines and nothing after them:

Answer: Yes
Confidence: 6
Correct: Yes

Solution:
Let me think through this step by step.
"""

    elif DATASET == "medqa":
        choices_text = "\n".join([f"{chr(65+i)}. {c}" for i, c in enumerate(choices)])
        base_prompt =    f"""Solve the following medical question. Think through it step by step, then provide your final answer and confidence. 

Question: {question}
{choices_text}

First, explicitly reason through the question step by step to arrive at an answer. Then, evaluate how likely your answer is to be correct by selecting EXACTLY ONE of these confidence classes:

1 = "Almost no chance" (0-10% likely correct)
2 = "Highly unlikely" (10-20% likely correct)
3 = "Chances are slight" (20-30% likely correct)
4 = "Unlikely" (30-40% likely correct)
5 = "Less than even" (40-50% likely correct)
6 = "Better than even" (50-60% likely correct)
7 = "Likely" (60-70% likely correct)
8 = "Very good chance" (70-80% likely correct)
9 = "Highly likely" (80-90% likely correct)
10 = "Almost certain" (90-100% likely correct)


After your step-by-step solution, you MUST end with EXACTLY this format (replace <YOUR_FINAL_ANSWER> with your answer letter):
Answer: <YOUR_FINAL_ANSWER>
Confidence: <1-10>
Correct: Yes or No

The Confidence number MUST match the class you selected — for example, if you select "Better than even" you MUST write Confidence: 6, not any other number.

Solution:
Let me work through this step by step.

"""



    elif DATASET == "triviaqa":
        base_prompt = f"""Answer the following trivia question. Think through it step by step, then provide your answer and confidence.

Question: {question}

First, consider what you know about this topic and think through related facts that might help, step by step, to arrive at an answer. Then, thoroughly assess your confidence in that answer by evaluating your thinking process so far. Finally, classify your confidence into one of the following classes based on how likely your answer is to be correct:

- "Almost no chance" (0.0–0.1)
- "Highly unlikely" (0.1–0.2)
- "Chances are slight" (0.2–0.3)
- "Unlikely" (0.3–0.4)
- "Less than even" (0.4–0.5)
- "Better than even" (0.5–0.6)
- "Likely" (0.6–0.7)
- "Very good chance" (0.7–0.8)
- "Highly likely" (0.8–0.9)
- "Almost certain" (0.9–1.0)

Each category reflects the probability that your answer is correct. You MUST end your response with EXACTLY this format (these three lines are required):
Answer: [just the answer, no extra words]
Confidence: [1-10]
Correct: [Yes/No]

Example format:
Answer: Paris
Confidence: 8
Correct: Yes

Solution:
Let me think through this step by step.
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

#just for the answer
    if DATASET == "gsm8k":
        base_prompt = f"""Solve step by step, then give your final numerical answer.

Question: {question}

Think step by step. You MUST end your response with your final answer on its own line in exactly this format:
Answer: [number]
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
        # choices must be a list of strings by this point (normalized in evaluation.py)
        if not choices or not isinstance(choices[0], str) or len(choices[0]) <= 1:
            raise ValueError(f"medqa choices look wrong — got: {choices}. "
                             f"Check evaluation.py options extraction.")
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
    
    Returns confidence as integer 1-10.
    """
    confidence_prompt = f"""You solved the following problem:

Question: {question}

Your answer: {answer}

How confident are you that your answer is correct?
Respond with ONLY a single integer from 1 to 10 (where 1 = very uncertain, 10 = very certain), nothing else."""
    
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
        # Normalize legacy formats
        if conf <= 1.0 and '.' in match.group(1):
            conf = conf * 10.0
        elif conf > 10:
            conf = conf / 10.0
        return min(10.0, max(1.0, round(conf)))
    return None

def get_two_pass_confidence(
    model,
    tokenizer,
    question: str,
    answer: str,
    reasoning: str,
    choices: list = None,
) -> Dict:
    """
    Two-pass verbalized confidence: show the model its own reasoning and
    answer in a SEPARATE generation, then ask it to critique before rating.
    This breaks the self-reinforcement loop where the model generates an
    answer and immediately rates itself highly because it just produced a
    coherent chain of thought.  By forcing a fresh pass that explicitly
    asks "look for errors", the model is more likely to notice mistakes
    and assign a lower, more calibrated confidence.
    Returns a dict with:
        - two_pass_confidence: float 1-10
        - two_pass_correct: bool or None
        - two_pass_critique: str (the full critique response)
    """
    from config import MODEL_VARIANT, DATASET
    # Truncate reasoning to avoid blowing context on very long CoT
    reasoning_trimmed = reasoning[:2000] if len(reasoning) > 2000 else reasoning
    # Build the choices text if applicable
    choices_text = ""
    if choices:
        choices_text = "\n".join([f"{chr(65+i)}. {c}" for i, c in enumerate(choices)])
        choices_text = f"\nAnswer choices:\n{choices_text}\n"
    critique_prompt = f"""You are reviewing a solution to the following problem. Your job is to check the reasoning for errors before rating your confidence.
Question: {question}
{choices_text}
Proposed solution:
{reasoning_trimmed}
Final answer given: {answer}
Instructions:
1. Re-read the solution step by step. For each step, check whether the logic and arithmetic are correct.
2. Identify any specific errors, unsupported assumptions, or steps where the reasoning is shaky.
3. If you find errors, explain them briefly.
4. Based on your review, rate your confidence that the final answer "{answer}" is correct using EXACTLY ONE of these levels:
1 = "Almost no chance correct" (0-10%)
2 = "Highly unlikely correct" (10-20%)
3 = "Slight chance correct" (20-30%)
4 = "Unlikely correct" (30-40%)
5 = "Less than even" (40-50%)
6 = "Better than even" (50-60%)
7 = "Likely correct" (60-70%)
8 = "Very good chance correct" (70-80%)
9 = "Highly likely correct" (80-90%)
10 = "Almost certain correct" (90-100%)
You MUST end your response with exactly:
Confidence: <1-10>
Correct: Yes or No"""
    if MODEL_VARIANT == "instruct":
        messages = [{"role": "user", "content": critique_prompt}]
        formatted_prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    else:
        formatted_prompt = critique_prompt + "\n\nReview:"
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    critique_response = tokenizer.decode(
        outputs[0, inputs.input_ids.shape[1]:],
        skip_special_tokens=True,
    ).strip()
    # Extract confidence from critique
    conf = extract_verbalized_confidence(critique_response, DATASET)
    # Extract correct judgment from critique
    correct_judgment = extract_more_likely_than_not(critique_response)
    return {
        "two_pass_confidence": conf,
        "two_pass_correct": correct_judgment,
        "two_pass_critique": critique_response,
    }


if __name__ == "__main__":
    """Quick smoke-test for extract_answer_token_entropy.

    Loads Qwen2.5-7B-Instruct, runs one greedy forward pass on a MedQA-style
    question, and prints the answer-letter probability dict + entropy so you
    can verify the signal looks sane before running the full pipeline.

    Expected confident output:   {A: ~0.90, B: ~0.05, ...}  entropy ≈ 0.3
    Expected uncertain output:   {A: ~0.35, B: ~0.30, ...}  entropy ≈ 1.3
    If entropy is near ln(5) ≈ 1.61 for every sample the raw_scores are wrong.
    """
    import sys
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name = "Qwen/Qwen2.5-7B-Instruct"
    print(f"Loading {model_name} …")
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    mdl = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True
    )

    question = (
        "A 45-year-old man presents with chest pain radiating to the left arm, "
        "diaphoresis, and shortness of breath for 30 minutes. "
        "Which of the following is the most likely diagnosis?"
    )
    choices = [
        "Stable angina",
        "Acute myocardial infarction",
        "Pulmonary embolism",
        "Aortic dissection",
        "Pericarditis",
    ]
    choices_text = "\n".join(f"{chr(65+i)}. {c}" for i, c in enumerate(choices))
    prompt_text = (
        f"Answer the following medical question.\n\nQuestion: {question}\n\n"
        f"{choices_text}\n\nAnswer:"
    )
    messages = [{"role": "user", "content": prompt_text}]
    formatted = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    response, token_probs, tokens, raw_scores = generate_with_logits(
        mdl, tok, formatted, max_new_tokens=256, do_sample=False
    )

    print("\n--- Generated response ---")
    print(response[:500])

    result = extract_answer_token_entropy(tokens, raw_scores, tok, dataset="medqa")
    print("\n--- Answer token entropy ---")
    print(f"  Letter probs : {result['answer_letter_probs']}")
    print(f"  Entropy      : {result['answer_token_entropy']:.4f}" if result["answer_token_entropy"] is not None else "  Entropy: None")
    print(f"  Top letter   : {result['top_answer_letter']}")
    print(f"  Chosen raw p : {result['chosen_answer_raw_prob']}")
    sys.exit(0)