# confidence.py - Logit-based and verbalized confidence functions with CoT
# print("Hello World")

import re
import torch
import numpy as np
from typing import Optional, Tuple, List, Dict
from config import (
    MODEL_VARIANT,
    MODEL_FAMILY,
    MAX_NEW_TOKENS,
    TWO_PASS_MAX_NEW_TOKENS,
    TWO_PASS_DISABLE_THINKING,
)

# GPT-OSS harmony format: committed answer lives after this delimiter.
# Defined here (not imported from evaluation.py) to avoid a circular import.
_HARMONY_FINAL_DELIM = "assistantfinal"
# Harmony analysis-channel marker: text starting with this is not a real answer.
# No \b — the channel name "analysis" runs directly into the next word ("analysisWe…").
_ANALYSIS_MARKER_RE = re.compile(r'^analysis', re.IGNORECASE)


def _detect_truncation(
    generated_text: str,
    generated_ids,
    tokenizer,
    expect_confidence_markers: bool = False,
) -> Dict:
    """Classify why generation stopped and whether the output is structurally complete.

    finish_reason is 'eos' if the last generated token is the EOS token,
    else 'length' (i.e. hit max_new_tokens). was_truncated is True if
    finish_reason is 'length', or — when expect_confidence_markers is True —
    if the text is missing </think> (when <think> was opened) or the
    "Confidence:" / "Correct:" markers the two-pass prompt requires.

    Empty-eos responses (base model emitting EOS immediately due to an OOD
    instruction prompt) are NOT classified as truncated — the model simply
    didn't generate anything, which is distinct from hitting the token budget.
    Flagging them truncated produces the contradictory combination
    (finish_reason=eos, was_truncated=True) that appears in the CSV when the
    two-pass critique prompt is sent to a base model.
    """
    last_id = int(generated_ids[-1]) if len(generated_ids) > 0 else -1
    eos_id = tokenizer.eos_token_id
    finish_reason = "eos" if eos_id is not None and last_id == eos_id else "length"

    was_truncated = (finish_reason == "length")
    # Only check structural completeness for non-empty responses. An empty eos
    # means the model exited immediately (OOD prompt) — not a truncation.
    if expect_confidence_markers and generated_text:
        think_opened = "<think>" in generated_text
        think_closed = (not think_opened) or ("</think>" in generated_text)
        has_confidence = bool(re.search(r"[Cc]onfidence\s*:", generated_text))
        has_correct = bool(re.search(r"[Cc]orrect\s*:", generated_text))
        if not (think_closed and has_confidence and has_correct):
            was_truncated = True

    return {"finish_reason": finish_reason, "was_truncated": was_truncated}


def _format_choices(choices: list) -> str:
    return "\n".join(f"{chr(65+i)}. {c}" for i, c in enumerate(choices))


_CONF_RUBRIC = """
Then, thoroughly assess your confidence in that answer by evaluating your thinking process so far. Finally, classify your confidence into one of the following classes based on how likely your answer is to be correct, by selecting EXACTLY ONE:

- 1 = "Almost no chance" (0-10% likely correct)
- 2 = "Highly unlikely" (10-20% likely correct)
- 3 = "Chances are slight" (20-30% likely correct)
- 4 = "Unlikely" (30-40% likely correct)
- 5 = "Less than even" (40-50% likely correct)
- 6 = "Better than even" (50-60% likely correct)
- 7 = "Likely" (60-70% likely correct)
- 8 = "Very good chance" (70-80% likely correct)
- 9 = "Highly likely" (80-90% likely correct)
- 10 = "Almost certain" (90-100% likely correct)

"""


def generate_with_logits(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = MAX_NEW_TOKENS,
    temperature: float = 1.0,
    do_sample: bool = False,
    repetition_penalty: float = None,
    no_repeat_ngram_size: int = None,
    stop_strings: list = None,
) -> Tuple[str, List[float], List[str], list, Dict]:
    """
    Generate response and capture token-level probabilities.

    Returns:
        - generated_text: The model's response
        - token_probs: Probability of each generated token
        - tokens: The actual tokens generated
        - raw_scores: Per-step vocab logit tensors, shape (1, vocab_size) each
        - meta: dict with "finish_reason" ("eos"|"length") and "was_truncated" (bool)

    Repetition-loop handling:
        Two model classes collapse into verbatim repetition loops on hard
        prompts — emitting the same line dozens-to-hundreds of times until they
        exhaust max_new_tokens (finish_reason="length"), never committing to a
        final answer:

          1. Base (non-instruct) models. On an instruction prompt they either
             loop verbatim or answer and then keep generating new questions.
          2. GPT-OSS-20B, which — despite being instruct-tuned — loops inside
             its Harmony "analysis" (reasoning) channel. On TriviaQA this hit
             ~16% of items (24/149), all finish_reason="length", with single
             lines repeated up to ~1000×; the answer never reached the "final"
             channel, so model_answer leaked the truncated analysis text. GSM8K
             (deterministic numeric target) was unaffected (0/150).

        Guard policy follows one principle: apply the minimal constraint that
        prevents non-termination, preferring constraints that are INERT on
        well-behaved generations (no_repeat_ngram_size, stop_strings) over
        repetition_penalty, which reshapes the distribution at every step and
        so perturbs the very token probabilities the confidence study measures.

          - no_repeat_ngram_size=3 → BOTH base and GPT-OSS. It only fires when a
            3-gram would repeat, so on a non-looping greedy generation the
            output (and its scores) are bit-identical to unconstrained decoding.
            GPT-OSS's loops are exactly repeated 3-grams, so this alone kills
            them; that lets GPT-OSS skip repetition_penalty entirely, keeping
            its non-looping rows untouched.
          - repetition_penalty → DISABLED for all models. Although originally
            applied to base only, empirical comparison showed it penalizes the
            evaluation format tokens (Answer:, Confidence:, Correct:) that
            appear in the prompt, causing the base model to avoid the structured
            output entirely and generate free-form prose instead. Net effect:
            loop rows dropped from ~12 to ~0, but extractable answers dropped
            from ~28 to ~5. The ngram ban alone is sufficient for base loops.
          - stop_strings → BASE ONLY. Targets the base over-generation failure
            (restating "Question:"/"Solution:" blocks); those markers don't fit
            GPT-OSS's reasoning loops and could clip a legitimate analysis
            channel.

        Every other instruct run (Qwen, Gemma) is left completely untouched
        (guards off), so existing instruct results remain reproducible
        byte-for-byte. Selective re-runs are therefore possible for GPT-OSS:
        only the looped rows change; clean rows are unaffected by ngram=3.

        When the guards are active they warp `outputs.scores`, which would
        corrupt the logit-confidence metrics AND the MCQ answer-token-entropy.
        So on the guarded path we re-derive both `token_probs` and `raw_scores`
        from a clean teacher-forced forward pass (raw, unwarped logits).
    """
    # Anti-loop ngram ban: base (any family) + GPT-OSS. Inert on clean rows.
    _needs_ngram_guard = (MODEL_VARIANT == "base") or (MODEL_FAMILY == "gptoss")
    # repetition_penalty disabled: penalizes format tokens (Answer:/Confidence:)
    # that appear in the prompt, breaking structured output for base models.
    _needs_rep_penalty = False

    # Resolve guard defaults (None = auto). See docstring for the policy.
    if repetition_penalty is None:
        repetition_penalty = 1.2 if _needs_rep_penalty else 1.0
    if no_repeat_ngram_size is None:
        no_repeat_ngram_size = 3 if _needs_ngram_guard else 0
    # Stop sequences target base over-generation only (not GPT-OSS loops).
    if stop_strings is None and MODEL_VARIANT == "base":
        stop_strings = ["\nQuestion:", "\nAnswer the following", "\nSolution:"]

    use_penalty = bool(repetition_penalty) and repetition_penalty != 1.0
    use_ngram = bool(no_repeat_ngram_size) and no_repeat_ngram_size > 0
    guards_active = use_penalty or use_ngram or bool(stop_strings)

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_length = inputs.input_ids.shape[1]

    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        return_dict_in_generate=True,
        output_scores=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    # temperature only matters when sampling (and warns under greedy otherwise).
    if do_sample:
        gen_kwargs["temperature"] = temperature
    if use_penalty:
        gen_kwargs["repetition_penalty"] = repetition_penalty
    if use_ngram:
        gen_kwargs["no_repeat_ngram_size"] = no_repeat_ngram_size
    if stop_strings:
        gen_kwargs["stop_strings"] = stop_strings
        gen_kwargs["tokenizer"] = tokenizer

    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)

    generated_ids = outputs.sequences[0, input_length:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    meta = _detect_truncation(generated_text, generated_ids, tokenizer)

    token_probs = []
    tokens = []

    if guards_active and generated_ids.numel() > 0:
        # Clean forward pass over the full sequence → unwarped per-token logits.
        with torch.no_grad():
            clean_logits = model(outputs.sequences).logits[0]  # (seq_len, vocab)
        raw_scores = []
        for i in range(generated_ids.shape[0]):
            # logits at position p predict token p+1; the i-th generated token
            # sits at absolute index (input_length + i).
            row = clean_logits[input_length + i - 1]
            raw_scores.append(row.unsqueeze(0))  # keep (1, vocab) shape
            probs = torch.softmax(row.float(), dim=-1)
            token_id = generated_ids[i].item()
            token_probs.append(probs[token_id].item())
            tokens.append(tokenizer.decode([token_id]))
    else:
        # Instruct (and any guards-off) path — identical to the original.
        raw_scores = list(outputs.scores)
        for i, score in enumerate(outputs.scores):
            probs = torch.softmax(score[0], dim=-1)
            token_id = generated_ids[i].item()
            token_prob = probs[token_id].item()
            token_probs.append(token_prob)
            tokens.append(tokenizer.decode([token_id]))

    return generated_text, token_probs, tokens, raw_scores, meta


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
        "chosen_letter": None,
        "chosen_answer_raw_prob": None,
    }

    if dataset not in MCQ_DATASETS:
        return null_result

    valid_letters = list("ABCDEFGHIJ") if dataset == "mmlupro" else list("ABCDE")

    # Build letter -> set of token IDs mapping by scanning the vocab.
    # Cached on the tokenizer so the O(vocab_size) decode loop runs once.
    letter_to_token_ids = _get_letter_token_ids(tokenizer, valid_letters)

    # Locate the last "Answer:" marker in the token stream, then find the
    # first answer-letter token that follows it.
    accumulated = ""
    marker_end_char = None
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

    # Sanity check: under greedy decoding, the letter the model actually emitted
    # (chosen_letter, found by walking the decoded text) should match the argmax
    # of the renormalized letter distribution (top_letter). When these disagree
    # it usually means the letter→token-id map missed the actually-emitted token
    # form — surface a warning rather than silently writing inconsistent CSV rows.
    if chosen_letter is not None and top_letter != chosen_letter:
        import warnings as _warnings
        _warnings.warn(
            f"extract_answer_token_entropy: emitted letter {chosen_letter!r} differs "
            f"from prob-distribution top {top_letter!r} (renorm={dict((l, round(p, 4)) for l, p in renorm.items())}). "
            f"Likely tokenizer/letter-id mismatch; the prob_X columns for this row may be unreliable.",
            RuntimeWarning,
        )

    return {
        "answer_token_entropy": entropy,
        "answer_letter_probs": {l: round(renorm[l], 4) for l in valid_letters},
        "top_answer_letter": top_letter,
        "chosen_letter": chosen_letter,
        "chosen_answer_raw_prob": round(chosen_answer_raw_prob, 6),
    }


# Pre-compiled patterns reused across calls
_re_answer_marker = re.compile(r"[Aa]nswer\s*:", re.IGNORECASE)
_re_think_block = re.compile(r'<think>.*?</think>', re.DOTALL)


# Cache of letter -> {token_id, ...} mappings, keyed per (tokenizer instance, letter set).
# Built once per tokenizer because vocab enumeration is O(vocab_size) decodes (~150k for Qwen).
_LETTER_TOKEN_IDS_ATTR = "_dcak_letter_token_ids_cache"


def _build_letter_token_ids(tokenizer, valid_letters: List[str]) -> Dict[str, set]:
    """Build a complete letter -> set-of-token-ids map by enumerating the vocab.

    For each token in the vocabulary, decode it to a string and check whether
    its stripped/uppercased form is exactly one of the valid letters. This
    catches every form the model might actually emit at the answer position
    ("E", " E", "E\\n", "E.", " E ", etc.) — including merged BPE tokens that
    bare `tokenizer.encode("E")` would miss. Without this, when the model
    emits a merged form like "E\\n" as a single token, that token's id falls
    outside letter_to_token_ids["E"], and renormalization amplifies whatever
    residual probability mass remains in the other letters' sets — typically
    showing up as a +1 column shift in the prob_X CSV columns.
    """
    valid_set = set(valid_letters)
    out: Dict[str, set] = {l: set() for l in valid_letters}
    vocab_size = getattr(tokenizer, "vocab_size", None)
    if not vocab_size:
        try:
            vocab_size = len(tokenizer.get_vocab())
        except Exception:
            return out
    for tid in range(vocab_size):
        try:
            decoded = tokenizer.decode([tid])
        except Exception:
            continue
        stripped = decoded.strip().upper()
        if stripped in valid_set:
            out[stripped].add(tid)
    return out


def _get_letter_token_ids(tokenizer, valid_letters: List[str]) -> Dict[str, set]:
    cache = getattr(tokenizer, _LETTER_TOKEN_IDS_ATTR, None)
    if cache is None:
        cache = {}
        try:
            setattr(tokenizer, _LETTER_TOKEN_IDS_ATTR, cache)
        except (AttributeError, TypeError):
            pass  # tokenizer doesn't allow attributes — just rebuild each time
    cache_key = tuple(valid_letters)
    if cache_key not in cache:
        cache[cache_key] = _build_letter_token_ids(tokenizer, valid_letters)
    return cache[cache_key]


def _truncate_to_first_block(response: str) -> str:
    """
    Cut a response after its FIRST completed Answer/Confidence/Correct block.

    Base models often answer correctly and then keep generating — restating the
    template or hallucinating new questions and answering those. Extractors that
    scan the whole response (or take the LAST match) then grab the continuation's
    answer/confidence or a literal "<YOUR_ANSWER>" placeholder, marking a correct
    answer wrong. Restricting every extractor to the first block fixes that;
    within the block last-match behaviour is preserved so genuine mid-solution
    self-correction still works.
    """
    if not response:
        return response
    m = re.search(r'\*{0,2}[Cc]orrect\*{0,2}\s*:\s*(?:Yes|No)\b', response, re.IGNORECASE)
    if m:
        return response[:m.end()]
    m2 = re.search(r'\n\s*(?:Question\s*:|Answer the following|Solution\s*:)',
                   response, re.IGNORECASE)
    if m2:
        return response[:m2.start()]
    return response


def extract_verbalized_confidence(response: str, dataset: str) -> Optional[float]:
    """
    Extract verbalized confidence from the model's response.

    Returns confidence as integer 1-10.
    Handles:
    - Explicit colon:        "Confidence: 7",  "Confidence: 8/10"
    - Word before colon:     "Confidence level: 9",  "Confidence score: 8"
    - No colon (own line):   "Confidence 9"  (base-model standalone format)
    - Prose form:            "my confidence is 9",  "confidence level is about 7 out of 10"
    - Markdown bold:         "**Confidence:** 9"
    - Approximate language:  "Confidence: about 6"
    - Legacy decimal:        "Confidence: 0.85"  → auto-converted to 1-10 scale
    - Legacy percentage:     "Confidence: 85%"   → auto-converted to 1-10 scale

    Three patterns are tried in priority order; the LAST match within each
    pattern wins (so a model that self-corrects mid-response still commits to
    its final rating).
    """
    # Only look at the model's first completed block (ignore any continuation).
    # Strip markdown bold for easier matching.
    cleaned = _truncate_to_first_block(response).replace('*', '')

    _FILLER = r'(?:approximately|about|around|only|just|~|roughly|nearly|almost)?'
    _SUFFIX  = r'(?:/10|out\s+of\s+10|%)?'

    # P1: explicit colon, optionally one filler word between "Confidence" and ":"
    # Covers: "Confidence: 8", "Confidence level: 9", "Confidence score: 7/10"
    p1 = (r'[Cc]onfidence(?:\s+\w+)?\s*:\s*'
          + _FILLER + r'\s*(\d+(?:\.\d+)?)\s*' + _SUFFIX)

    # P2: no colon, number on its own line — base-model structured-output pattern
    # Covers: "Confidence 9\n", "Confidence 10\n"
    p2 = r'(?m)^[Cc]onfidence\s+(\d+)\s*' + _SUFFIX + r'\s*$'

    # P3: prose form — "confidence is 9", "confidence level is about 7 out of 10"
    # The _SUFFIX captures the denominator so we grab the NUMERATOR (e.g. 7, not 10).
    p3 = (r'[Cc]onfidence(?:\s+\w+)?\s+'
          r'(?:is|of|at)\s+'
          + _FILLER + r'\s*(\d+(?:\.\d+)?)\s*' + _SUFFIX)

    for pattern in (p1, p2, p3):
        matches = re.findall(pattern, cleaned)
        if matches:
            conf = float(matches[-1])
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
    # Only look at the model's first completed block (ignore any continuation).
    response = _truncate_to_first_block(response)
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


def create_prompt(tokenizer, question: str, choices: list = None, include_confidence: bool = True) -> str:
    """
    Create prompt with Chain-of-Thought reasoning and optionally verbalized confidence.

    When include_confidence=False (qwen3 Gen 1), the prompt asks only for thorough
    reasoning and a final answer — no confidence rubric or Confidence/Correct output.
    Confidence is elicited separately in Gen 2 to avoid token-limit truncation.

    The prompt is built in two parts:
      - instruction_body: clean instructions + format example that explicitly says
        "end your response with…" so the Answer line is the LAST thing the model
        emits. Sent as-is to instruct/reasoning models via the chat template.
      - base_primer: a trailing "Solution:\\nLet me … step by step." line that acts
        as a next-token continuation primer for BASE models (which don't follow
        instructions and need to be primed mid-sentence). Appended ONLY for base
        models. Instruct models would otherwise interpret the primer as part of
        the user instruction and get stuck in a loop trying to reconcile
        "end with Answer:" vs. "end with Solution: Let me think…".
    """
    from config import MODEL_VARIANT, DATASET

    if DATASET == "gsm8k":
        primer_verb = "work through"
        if include_confidence:
            instruction_body = f"""Solve the following math problem. Think through it step by step, then provide your final answer and confidence.

Question: {question}

First, explicitly reason through the question step by step to arrive at an answer.{_CONF_RUBRIC}After your step-by-step solution, end your response with EXACTLY these three lines (replace <YOUR_NUMBER> with your computed answer):
Answer: <YOUR_NUMBER>
Confidence: <1-10>
Correct: Yes or No

The Confidence number MUST match the class you selected — for example, if you select "Better than even" you MUST write Confidence: 6, not any other number."""
        else:
            instruction_body = f"""Solve the following math problem. Think through it step by step, then provide your final answer.

Question: {question}

Reason through the problem step by step. End your response with a single line in this format:
Answer: <YOUR_NUMBER>"""

    elif DATASET == "mmlupro":
        primer_verb = "analyze each option"
        choices_text = _format_choices(choices)
        if include_confidence:
            instruction_body = f"""Answer the following multiple choice question. Think through it step by step, then provide your answer and confidence.

Question: {question}

{choices_text}

First, analyze each option carefully and explain your reasoning step by step to arrive at an answer.{_CONF_RUBRIC}After your step-by-step analysis, end your response with EXACTLY these three lines (replace <YOUR_LETTER> with your chosen answer letter):
Answer: <YOUR_LETTER>
Confidence: <1-10>
Correct: Yes or No

The Confidence number MUST match the class you selected — for example, if you select "Better than even" you MUST write Confidence: 6, not any other number."""
        else:
            instruction_body = f"""Answer the following multiple choice question. Think through it step by step, then provide your answer.

Question: {question}

{choices_text}

Analyze each option carefully and explain your reasoning step by step. End your response with a single line in this format (just the letter, e.g. B):
Answer: <YOUR_LETTER>"""

    elif DATASET == "strategyqa":
        primer_verb = "think through"
        if include_confidence:
            instruction_body = f"""Answer the following yes/no question. Think through it step by step, then provide your answer and confidence.

Question: {question}

First, consider relevant facts and reasoning, and explain your thinking step by step to arrive at an answer.{_CONF_RUBRIC}After your reasoning, end your response with EXACTLY these three lines (replace <YOUR_ANSWER> with Yes or No):
Answer: <YOUR_ANSWER>
Confidence: <1-10>
Correct: Yes or No

The Confidence number MUST match the class you selected — for example, if you select "Better than even" you MUST write Confidence: 6, not any other number."""
        else:
            instruction_body = f"""Answer the following yes/no question. Think through it step by step, then provide your answer.

Question: {question}

Consider relevant facts and reasoning, and explain your thinking step by step. End your response with a single line in this format:
Answer: Yes
or
Answer: No"""

    elif DATASET == "medqa":
        primer_verb = "work through"
        choices_text = _format_choices(choices)
        if include_confidence:
            instruction_body = f"""Solve the following medical question. Think through it step by step, then provide your final answer and confidence.

Question: {question}
{choices_text}

First, explicitly reason through the question step by step to arrive at an answer.{_CONF_RUBRIC}After your step-by-step solution, end your response with EXACTLY these three lines (replace <YOUR_FINAL_ANSWER> with your answer letter):
Answer: <YOUR_FINAL_ANSWER>
Confidence: <1-10>
Correct: Yes or No

The Confidence number MUST match the class you selected — for example, if you select "Better than even" you MUST write Confidence: 6, not any other number."""
        else:
            instruction_body = f"""Solve the following medical question. Think through it step by step, then provide your final answer.

Question: {question}
{choices_text}

Reason through the clinical presentation step by step. End your response with a single line in this format:
Answer: <YOUR_FINAL_ANSWER>"""

    elif DATASET == "legalbench":
        primer_verb = "think through"
        if include_confidence:
            instruction_body = f"""Answer the following legal-reasoning yes/no question. Think through it step by step, then provide your answer and confidence.

Question: {question}

First, consider the relevant legal rules and facts, and explain your reasoning step by step to arrive at an answer.{_CONF_RUBRIC}After your reasoning, end your response with EXACTLY these three lines (replace <YOUR_ANSWER> with Yes or No):
Answer: <YOUR_ANSWER>
Confidence: <1-10>
Correct: Yes or No

The Confidence number MUST match the class you selected — for example, if you select "Better than even" you MUST write Confidence: 6, not any other number."""
        else:
            instruction_body = f"""Answer the following legal-reasoning yes/no question. Think through it step by step, then provide your answer.

Question: {question}

Consider the relevant legal rules and facts, and explain your reasoning step by step. End your response with a single line in this format:
Answer: Yes
or
Answer: No"""

    elif DATASET == "triviaqa":
        primer_verb = "think through"
        if include_confidence:
            instruction_body = f"""Answer the following trivia question. Think through it step by step, then provide your answer and confidence.

Question: {question}

First, consider what you know about this topic and think through related facts that might help, step by step, to arrive at an answer.{_CONF_RUBRIC}After your reasoning, end your response with EXACTLY these three lines (replace <YOUR_ANSWER> with the answer, no extra words):
Answer: <YOUR_ANSWER>
Confidence: <1-10>
Correct: Yes or No

The Confidence number MUST match the class you selected — for example, if you select "Better than even" you MUST write Confidence: 6, not any other number."""
        else:
            instruction_body = f"""Answer the following trivia question. Think through it step by step, then provide your answer.

Question: {question}

Consider what you know and reason through related facts step by step. End your response with a single line in this format (just the answer, no extra words):
Answer: <YOUR_ANSWER>"""

    else:
        # Fallback for unknown datasets — keep behavior generic
        primer_verb = "think through"
        instruction_body = f"""Answer the following question. Think through it step by step.

Question: {question}

End your response with a single line in this format:
Answer: <YOUR_ANSWER>"""

    if MODEL_VARIANT == "instruct":
        messages = [{"role": "user", "content": instruction_body}]
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    else:
        # Base models don't follow instructions; prime them mid-sentence so they
        # continue naturally into the reasoning and eventually emit "Answer: X".
        base_primer = f"\n\nSolution:\nLet me {primer_verb} this step by step.\n\n"
        return instruction_body + base_primer


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
        choices_text = _format_choices(choices)
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
        choices_text = _format_choices(choices)
        base_prompt = f"""Question: {question}

{choices_text}

Think through the clinical presentation step by step, then write JUST the answer letter after "Answer:".
Solution:"""

    elif DATASET == "triviaqa":
        base_prompt = f"""Question: {question}

Think step by step, then write JUST the answer after "Answer:".
Solution:"""

    elif DATASET == "legalbench":
        base_prompt = f"""Question: {question}

Think through the legal reasoning step by step, then write JUST Yes or No after "Answer:".
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

    Base models receive a minimal Q&A-style prompt that matches their
    pretraining distribution. The instruction-style prompt causes them to
    emit EOS immediately (it's OOD), which the regex then misses — or,
    if the rubric text happens to end with "10", returns 10.0 for every row.
    """
    from model_utils import generate_simple_response
    from config import MODEL_VARIANT

    if MODEL_VARIANT == "base":
        # Minimal Q&A format — matches pretraining. Prompt already ends with
        # the answer trigger so base_suffix is empty.
        confidence_prompt = f"Q: {question}\nA: {answer}\nConfidence (1-10):"
        base_suffix = ""
    else:
        confidence_prompt = f"""You solved the following problem:

Question: {question}

Your answer: {answer}

How confident are you that your answer is correct?
Respond with ONLY a single integer from 1 to 10 (where 1 = very uncertain, 10 = very certain), nothing else."""
        base_suffix = "\n\nConfidence:"

    response = generate_simple_response(
        model, tokenizer, confidence_prompt, max_new_tokens=10, base_suffix=base_suffix
    )

    match = re.search(r'(\d+(?:\.\d+)?)', response)
    if match:
        conf = float(match.group(1))
        if conf <= 1.0 and '.' in match.group(1):
            conf = conf * 10.0
        elif conf > 10:
            conf = conf / 10.0
        return min(10.0, max(1.0, round(conf)))
    return None


def get_correct_separate_base(
    model,
    tokenizer,
    question: str,
    answer: str,
) -> Optional[bool]:
    """
    Ask a base model whether its answer is correct by comparing logits.

    Logit-comparison rather than generation: one forward pass, compare the
    raw logit for ' Yes' vs ' No' at the first output position. Always
    returns True or False — never None (unless tokenization fails to produce
    distinct Yes/No token IDs, which is handled by falling back to None).

    Why not generation: greedy continuation after "A:" is unreliable for
    Llama-3.1-8B-base — the model's preferred token is not consistently
    'Yes' or 'No' even for clearly correct/incorrect answers (§30.1). The
    generative version left single_pass_correct blank (e.g. idx 3101, 8936,
    7574, 16129). Logit comparison bypasses generation entirely.

    Only called for base models — instruct models produce "Correct: Yes/No"
    inline in the main response, extracted by extract_more_likely_than_not.
    """
    # Truncate verbose mid-sentence answers to avoid strong continuation
    # signals that skew the logit distribution toward text completion.
    _ans = answer
    if len(_ans) > 150:
        _last_period = _ans[:150].rfind('.')
        _ans = _ans[:_last_period + 1] if _last_period != -1 else _ans[:150]

    prompt = f"Q: {question}\nA: {_ans}\nQ: Is this answer correct? A:"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # Token IDs for " Yes" and " No" (leading space = how they appear mid-text)
    yes_ids = [
        tokenizer.encode(s, add_special_tokens=False)[0]
        for s in (" Yes", "Yes", " yes", "yes")
        if tokenizer.encode(s, add_special_tokens=False)
    ]
    no_ids = [
        tokenizer.encode(s, add_special_tokens=False)[0]
        for s in (" No", "No", " no", "no")
        if tokenizer.encode(s, add_special_tokens=False)
    ]
    if not yes_ids or not no_ids:
        return None  # tokenizer gave nothing useful

    with torch.no_grad():
        logits = model(**inputs).logits[0, -1, :]  # shape: [vocab_size]

    yes_logit = max(logits[i].item() for i in yes_ids)
    no_logit = max(logits[i].item() for i in no_ids)
    return yes_logit > no_logit


def get_gen2_confidence(
    model,
    tokenizer,
    question: str,
    reasoning: str,
    answer: str,
    choices: list = None,
) -> Dict:
    """
    Gen 2 verbalized confidence for qwen3 (own-work-aware).

    The model is explicitly told this is ITS OWN reasoning chain and asked
    to rate confidence and more-likely-than-not based solely on that reasoning.
    This is a short, separate generation so it cannot be truncated by a long
    think block the way the inline confidence request can.

    Returns:
        gen2_confidence  – float 1-10 or None
        gen2_correct     – bool or None
        gen2_response    – raw response string
    """
    from config import MODEL_VARIANT, DATASET

    # 3000 chars: Gen 2 is own-work-aware and benefits from seeing more detail
    reasoning_trimmed = reasoning[:3000] if len(reasoning) > 3000 else reasoning
    choices_text = ""
    if choices:
        choices_text = _format_choices(choices)
        choices_text = f"\nAnswer choices:\n{choices_text}\n"

    prompt = f"""The following is YOUR OWN reasoning chain and final answer that YOU previously produced for the question below. This work is entirely yours.

Question: {question}
{choices_text}
YOUR reasoning chain:
{reasoning_trimmed}

YOUR final answer: {answer}

Based on YOUR reasoning chain and thought process above, how confident are you that YOUR answer is correct?

Select EXACTLY ONE confidence level:

- 1 = "Almost no chance" (0-10% likely correct)
- 2 = "Highly unlikely" (10-20% likely correct)
- 3 = "Chances are slight" (20-30% likely correct)
- 4 = "Unlikely" (30-40% likely correct)
- 5 = "Less than even" (40-50% likely correct)
- 6 = "Better than even" (50-60% likely correct)
- 7 = "Likely" (60-70% likely correct)
- 8 = "Very good chance" (70-80% likely correct)
- 9 = "Highly likely" (80-90% likely correct)
- 10 = "Almost certain" (90-100% likely correct)

Do NOT write any explanation. Your entire visible response must consist of ONLY these two lines:
Confidence: <1-10>
Correct: Yes or No"""

    from model_utils import generate_simple_response
    # Same fix as the two-pass critique: a Qwen3 thinking model spends its
    # entire budget inside <think> on a 512-token call, so the Confidence:/
    # Correct: lines never appear and extraction silently returns None.
    # Use the bigger TWO_PASS_MAX_NEW_TOKENS budget and skip thinking on
    # Qwen3 — Gen 2 is just a self-rating, it doesn't need extended reasoning.
    enable_thinking = False if TWO_PASS_DISABLE_THINKING else None
    response = generate_simple_response(
        model, tokenizer, prompt,
        max_new_tokens=TWO_PASS_MAX_NEW_TOKENS,
        base_suffix="\n\nAssessment:",
        enable_thinking=enable_thinking,
    )

    # Strip think blocks before extraction so internal reasoning doesn't interfere
    response_clean = _re_think_block.sub('', response).strip()
    conf = extract_verbalized_confidence(response_clean, DATASET)
    correct = extract_more_likely_than_not(response_clean)

    return {
        "gen2_confidence": conf,
        "gen2_correct": correct,
        "gen2_response": response_clean,
    }


def get_two_pass_confidence(
    model,
    tokenizer,
    question: str,
    answer: str,
    reasoning: str,
    choices: list = None,
    gen2_confidence: Optional[float] = None,
    gen2_correct: Optional[bool] = None,
) -> Dict:
    """
    Gen 3 (two-pass) verbalized confidence: present the reasoning, answer, and
    optionally a pre-assigned verbalized score as anonymous third-party work, then
    ask the model to critique and independently rate confidence.

    For qwen3, gen2_confidence and gen2_correct are passed in from Gen 2 so the
    critique prompt includes the self-reported score — but the model is NOT told
    the work is its own, reducing self-serving bias.

    For other model families, gen2_confidence/gen2_correct are None and the prompt
    mirrors the original two-pass structure.

    Returns:
        two_pass_confidence – float 1-10 or None
        two_pass_correct    – bool or None
        two_pass_critique   – raw critique response string
    """
    from config import MODEL_VARIANT, DATASET

    # 2000 chars: blinded reviewer gets a summary; longer would bloat the critique prompt
    reasoning_trimmed = reasoning[:2000] if len(reasoning) > 2000 else reasoning
    choices_text = ""
    if choices:
        choices_text = _format_choices(choices)
        choices_text = f"\nAnswer choices:\n{choices_text}\n"

    # When Gen 2 scores are available, include them as context so the critique can
    # agree or push back on the self-reported confidence — without revealing authorship.
    if gen2_confidence is not None:
        gen2_correct_str = "Yes" if gen2_correct else ("No" if gen2_correct is not None else "unknown")
        assigned_score_block = f"""
The respondent also self-assessed their answer and assigned:
  Verbalized confidence: {gen2_confidence}/10
  More likely correct than not: {gen2_correct_str}
"""
    else:
        assigned_score_block = ""

    critique_prompt = f"""You are reviewing a solution submitted by someone else to the following problem. Your job is to check the reasoning for errors and independently assess how likely the final answer is correct.

REQUIRED OUTPUT FORMAT — your response MUST end with these two lines, exactly:
Confidence: <integer 1-10>
Correct: <Yes or No>

Question: {question}
{choices_text}
Submitted solution:
{reasoning_trimmed}

Final answer given: {answer}{assigned_score_block}
Instructions:
1. Re-read the solution step by step. For each step, check whether the logic and arithmetic are correct.
2. Identify any specific errors, unsupported assumptions, or steps where the reasoning is shaky.
3. If you find errors, explain them briefly.
4. Based on your independent review, rate your confidence that the final answer "{answer}" is correct by selecting EXACTLY ONE of these classes:

- 1 = "Almost no chance" (0-10% likely correct)
- 2 = "Highly unlikely" (10-20% likely correct)
- 3 = "Chances are slight" (20-30% likely correct)
- 4 = "Unlikely" (30-40% likely correct)
- 5 = "Less than even" (40-50% likely correct)
- 6 = "Better than even" (50-60% likely correct)
- 7 = "Likely" (60-70% likely correct)
- 8 = "Very good chance" (70-80% likely correct)
- 9 = "Highly likely" (80-90% likely correct)
- 10 = "Almost certain" (90-100% likely correct)

You MUST end your response with exactly:
Confidence: <1-10>
Correct: Yes or No"""
    if MODEL_VARIANT == "instruct":
        messages = [{"role": "user", "content": critique_prompt}]
        # Qwen3 chat templates accept `enable_thinking`; pass it when we
        # want to skip the <think> block so the critique budget is spent on
        # the critique itself. Other tokenizers ignore unknown kwargs in
        # most transformers versions, but guard with try/except to be safe.
        template_kwargs = {"tokenize": False, "add_generation_prompt": True}
        if TWO_PASS_DISABLE_THINKING:
            template_kwargs["enable_thinking"] = False
        try:
            formatted_prompt = tokenizer.apply_chat_template(messages, **template_kwargs)
        except TypeError:
            template_kwargs.pop("enable_thinking", None)
            formatted_prompt = tokenizer.apply_chat_template(messages, **template_kwargs)
    elif MODEL_FAMILY == "llama":
        # Llama-3.1-8B-base emits EOS immediately on the dense instruction-style
        # critique prompt: the model is too small and the rubric text is OOD,
        # so no valid continuation tokens survive and EOS wins (§25/§26 mechanism).
        # Qwen/Gemma base models are large enough to pattern-complete the full
        # critique prompt and produce valid critiques (confirmed by CSV evidence).
        # Use a minimal native Q&A format for Llama base only.
        formatted_prompt = (
            f"Q: {question}\n"
            f"A: {answer}\n"
            f"Q: Is this answer correct? Rate confidence 1-10.\n"
            f"A:"
        )
    else:
        formatted_prompt = critique_prompt + "\n\nReview:"
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
    _two_pass_gen_kwargs = dict(
        max_new_tokens=TWO_PASS_MAX_NEW_TOKENS,
        do_sample=False,
        return_dict_in_generate=True,
        pad_token_id=tokenizer.pad_token_id,
    )
    if MODEL_FAMILY == "llama" and MODEL_VARIANT == "base":
        # Short native Q&A output — cap tokens and skip ngram guard (short prompt,
        # no loop risk; ngram guard over-bans on compact Q&A context).
        _two_pass_gen_kwargs["max_new_tokens"] = 128
    elif MODEL_FAMILY == "gptoss" or MODEL_VARIANT == "base":
        # Anti-loop guard for gptoss (dense assistant format loops without it)
        # and all non-Llama base models (Qwen/Gemma base can pattern-complete the
        # critique but loop without the ngram guard on max_new_tokens budgets).
        _two_pass_gen_kwargs["no_repeat_ngram_size"] = 3
    with torch.no_grad():
        outputs = model.generate(**inputs, **_two_pass_gen_kwargs)
    generated_ids = outputs.sequences[0, inputs.input_ids.shape[1]:]
    critique_response = tokenizer.decode(
        generated_ids, skip_special_tokens=True
    ).strip()
    meta = _detect_truncation(
        critique_response, generated_ids, tokenizer, expect_confidence_markers=True
    )
    # Strip harmony envelope before extraction. GPT-OSS writes analysis in a
    # pre-"assistantfinal" channel; _truncate_to_first_block can cut at a
    # mid-sentence "Correct: Yes" in that channel and then
    # extract_more_likely_than_not (which requires a line-start ^) returns None.
    critique_for_extraction = critique_response
    _crit_lower = critique_response.lower()
    _crit_idx = _crit_lower.rfind(_HARMONY_FINAL_DELIM)
    if _crit_idx != -1:
        critique_for_extraction = critique_response[_crit_idx + len(_HARMONY_FINAL_DELIM):].strip()
    conf = extract_verbalized_confidence(critique_for_extraction, DATASET)
    correct_judgment = extract_more_likely_than_not(critique_for_extraction)

    return {
        "two_pass_confidence": conf,
        "two_pass_correct": correct_judgment,
        "two_pass_critique": critique_response,
        "two_pass_finish_reason": meta["finish_reason"],
        "two_pass_was_truncated": meta["was_truncated"],
    }


def _truncate_countdown_loop(response: str) -> str:
    """Cut before a base-model probability-countdown loop.

    A base model that knows the answer but can't commit to the Answer: format
    sometimes enters a countdown loop: "I'll say there's a 99% chance X" →
    "98%" → "97%"… The percentages change each iteration so
    no_repeat_ngram_size=3 doesn't catch it (every line's tokens differ), but
    the response exhausts max_new_tokens without ever writing Answer:.

    Strategy: find lines containing both a confidence trigger phrase and a
    percentage. If 3+ such lines appear within a 30-line span, truncate at the
    first one so the forced pass gets the clean initial reasoning instead.

    Handles concatenated-word variants the base model produces mid-loop
    (e.g. "Iwill say", "I willsaya") via \\s* between tokens.
    """
    _COUNTDOWN_LINE_RE = re.compile(
        r"(?:i\s*(?:'ll|will)\s*say|there\s*(?:'s|is))[^\n]{0,80}\d{1,3}\s*%",
        re.IGNORECASE,
    )
    lines = response.split('\n')
    matched = [i for i, ln in enumerate(lines) if _COUNTDOWN_LINE_RE.search(ln)]
    if len(matched) >= 3 and (matched[-1] - matched[0]) <= 30:
        return '\n'.join(lines[:matched[0]]).rstrip()
    return response


def get_forced_answer(
    model,
    tokenizer,
    question: str,
    reasoning: str,
    dataset: str,
    choices: list = None,
) -> Tuple[Optional[str], str]:
    """
    Force a final answer when the main pass was truncated by the token budget.

    Shows the model its (likely incomplete) reasoning and asks for ONLY the
    answer in the dataset's expected format — no further reasoning. Used
    instead of relying on extract_model_answer's Priority-3 fallback (last
    standalone letter/number in response), which on a truncated qwen3
    <think> block returns whatever letter happens to appear last in the
    chain of thought, not a real commitment.

    Returns:
        (forced_answer, forced_response) — forced_answer parsed via the
        same extract_model_answer used on the main pass; None if the
        forced call still failed to produce one.
    """
    from data_utils import extract_model_answer
    from model_utils import generate_simple_response

    # Strip probability-countdown loops before clipping so the forced pass
    # sees the clean initial reasoning rather than the loop that exhausted
    # max_new_tokens. Inert on responses without a countdown pattern.
    reasoning_delooped = _truncate_countdown_loop(reasoning)
    # 3000 chars: enough for the model to recall its train of thought without
    # bloating a forced-answer prompt that should be quick.
    reasoning_clip = reasoning_delooped[:3000] if len(reasoning_delooped) > 3000 else reasoning_delooped

    # Each branch builds the *body* of the prompt — no trailing template line.
    # The literal "Answer: " is appended via base_suffix below so that BASE
    # models complete from that point with just the answer value rather than
    # regurgitating template placeholders like "<number>" or "<0-10>".
    if dataset == "mmlupro":
        choices_text = _format_choices(choices)
        prompt = f"""You were working on this multiple choice question but ran out of thinking time and did NOT commit to a final answer.

Question: {question}

{choices_text}

Your reasoning so far (likely incomplete):
{reasoning_clip}

Based on your reasoning above — even if it is incomplete or inconclusive — commit to your best-guess answer letter NOW. Output only the single letter (A through J)."""
        max_tokens = 8
    elif dataset == "medqa":
        choices_text = _format_choices(choices)
        prompt = f"""You were working on this medical question but ran out of thinking time and did NOT commit to a final answer.

Question: {question}
{choices_text}

Your reasoning so far (likely incomplete):
{reasoning_clip}

Based on your reasoning above — even if it is incomplete or inconclusive — commit to your best-guess answer letter NOW. Output only the single letter (A through E)."""
        max_tokens = 8
    elif dataset == "gsm8k":
        prompt = f"""You were solving this math problem but ran out of thinking time and did NOT commit to a final answer.

Question: {question}

Your work so far (likely incomplete):
{reasoning_clip}

Based on your work above, commit to your best-guess numerical answer NOW. Output only the number."""
        max_tokens = 16
    elif dataset == "strategyqa":
        prompt = f"""You were answering this yes/no question but ran out of thinking time and did NOT commit to a final answer.

Question: {question}

Your reasoning so far (likely incomplete):
{reasoning_clip}

Based on your reasoning above, commit to a final answer NOW. Output only the word Yes or No."""
        max_tokens = 8
    elif dataset == "triviaqa":
        prompt = f"""You were answering this trivia question but ran out of thinking time and did NOT commit to a final answer.

Question: {question}

Your reasoning so far (likely incomplete):
{reasoning_clip}

Based on your reasoning above, commit to your best-guess answer NOW. Output only the answer, no extra words."""
        max_tokens = 32
    elif dataset == "legalbench":
        prompt = f"""You were answering this legal-reasoning yes/no question but ran out of thinking time and did NOT commit to a final answer.

Question: {question}

Your reasoning so far (likely incomplete):
{reasoning_clip}

Based on your reasoning above, commit to a final answer NOW. Output only the word Yes or No."""
        max_tokens = 8
    else:
        return None, ""

    # For BASE models: the instruction-style prompts built above are outside
    # their training distribution — they generate EOS immediately because they
    # have never seen text like "commit to your best-guess answer NOW. Output
    # only..." followed by a factual completion. Override to a minimal Q&A
    # format that matches their pretraining distribution.
    from config import MODEL_VARIANT, MODEL_FAMILY
    _forced_base_suffix = "\n\nAnswer: "  # instruct path ignores base_suffix
    if MODEL_VARIANT == "base" or MODEL_FAMILY == "gptoss":
        if dataset == "triviaqa":
            prompt = f"Q: {question}\nA:"
        elif dataset in ("mmlupro", "medqa"):
            prompt = f"Q: {question}\n{_format_choices(choices)}\nAnswer:"
        elif dataset == "gsm8k":
            prompt = f"Problem: {question}\nAnswer:"
        elif dataset in ("strategyqa", "legalbench"):
            prompt = f"Q: {question}\nAnswer (Yes or No):"
        _forced_base_suffix = ""  # prompt already ends with the answer trigger

    # loop_guard=False: the forced budget is ≤ 32 tokens — impossible to loop.
    forced_response = generate_simple_response(
        model, tokenizer, prompt,
        max_new_tokens=max_tokens,
        base_suffix=_forced_base_suffix,
        loop_guard=False,
    )

    # Strip qwen3 think blocks if the forced call also produced one
    forced_response_clean = _re_think_block.sub('', forced_response).strip()
    # Strip harmony envelope: GPT-OSS writes analysis before "assistantfinal";
    # without stripping, the analysis text leaks into the answer slot when the
    # forced call hits max_new_tokens before reaching the committed final section.
    _forced_lower = forced_response_clean.lower()
    _forced_idx = _forced_lower.rfind(_HARMONY_FINAL_DELIM)
    if _forced_idx != -1:
        forced_response_clean = forced_response_clean[_forced_idx + len(_HARMONY_FINAL_DELIM):].strip()

    # Truncate base-model Q&A continuations: base models completing "Q: ...\nA:"
    # often continue with "\nQ: ..." new questions. Extraction already anchors to
    # the first line, but this keeps forced_answer_response clean in stored output.
    _qa_cont = forced_response_clean.find('\nQ:')
    if _qa_cont != -1:
        forced_response_clean = forced_response_clean[:_qa_cont]

    # For base models the response is just the completion AFTER "Answer: " —
    # i.e. it starts with the answer value directly, no "Answer:" prefix. Try
    # extract_model_answer first; if that fails (no anchored "Answer:" line),
    # treat the cleaned response as the bare answer and re-parse.
    forced_answer = extract_model_answer(forced_response_clean, dataset)
    if forced_answer is None and forced_response_clean:
        # Prepend "Answer: " so the same extractor that handles instruct-path
        # responses can find it. This works because the base completion is just
        # the answer value on the first line.
        forced_answer = extract_model_answer(
            f"Answer: {forced_response_clean}", dataset
        )
    # Reject harmony analysis-channel text that leaked into the answer slot
    # (GPT-OSS truncated before "assistantfinal" → forced_response_clean starts
    # with "analysis"; extractor accepted it via the "Answer: {text}" path).
    if forced_answer is not None and _ANALYSIS_MARKER_RE.match(str(forced_answer)):
        forced_answer = None

    return forced_answer, forced_response_clean


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
    choices_text = _format_choices(choices)
    prompt_text = (
        f"Answer the following medical question.\n\nQuestion: {question}\n\n"
        f"{choices_text}\n\nAnswer:"
    )
    messages = [{"role": "user", "content": prompt_text}]
    formatted = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    response, token_probs, tokens, raw_scores, _meta = generate_with_logits(
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