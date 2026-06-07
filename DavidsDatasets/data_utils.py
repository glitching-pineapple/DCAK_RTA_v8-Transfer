# data_utils.py - Dataset loading and answer extraction

import math
import re
import unicodedata
from typing import Optional
from datasets import load_dataset


def load_gsm8k():
    dataset = load_dataset("openai/gsm8k", "main", split="test")
    print(f"Loaded GSM8K: {len(dataset)} test examples")
    return dataset


def load_mmlupro():
    ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
    print(f"Loaded MMLU-Pro: {len(ds)} test examples")
    return ds


def load_strategyqa():
    ds = load_dataset("ChilleD/StrategyQA", split="test")
    print(f"Loaded StrategyQA: {len(ds)} test examples")
    return ds


def load_medqa():
    """Load MedQA dataset (US medical licensing exam style)."""
    ds = load_dataset("GBaker/MedQA-USMLE-4-options", split="test")
    print(f"Loaded MedQA: {len(ds)} test examples")
    return ds


def load_triviaqa():
    """Load TriviaQA dataset (questions + answers only, skip large document files).

    Uses the canonical namespaced repo `mandarjoshi/trivia_qa`. Newer
    huggingface_hub versions reject the bare-name form `trivia_qa`, so we no
    longer fall back to it — that fallback used to mask the real loader error
    with an HfUriError about repo-id format.
    """
    try:
        ds = load_dataset(
            "mandarjoshi/trivia_qa", "rc.nocontext",
            split="validation", trust_remote_code=True,
        )
    except TypeError:
        # Older `datasets` versions don't accept trust_remote_code kwarg
        ds = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split="validation")
    print(f"Loaded TriviaQA: {len(ds)} validation examples")
    return ds


def load_legalbench():
    """Load LegalBench dataset (Yes/No legal-reasoning subtask selected via LEGALBENCH_TASK)."""
    from config import LEGALBENCH_TASK
    ds = load_dataset("nguha/legalbench", LEGALBENCH_TASK, split="test")
    print(f"Loaded LegalBench[{LEGALBENCH_TASK}]: {len(ds)} test examples")
    return ds


def extract_ground_truth(sample: dict, dataset: str) -> Optional[str]:
    """Extract ground truth based on dataset type."""
    if dataset == "gsm8k":
        match = re.search(r'####\s*([\d,]+)', sample['answer'])
        if match:
            return match.group(1).replace(',', '')
        return None
    
    elif dataset == "mmlupro":
        return sample['answer']
    
    elif dataset == "strategyqa":
        return "Yes" if sample['answer'] else "No"
    
    elif dataset == "medqa":
        # Some versions of the dataset return the letter directly ("A"),
        # others return an integer index. Try `answer_idx` first, then
        # fall back to `answer`.
        for key in ("answer_idx", "answer"):
            if key in sample:
                ans = sample[key]
                if isinstance(ans, int):
                    return chr(65 + ans)
                return str(ans).upper()
        return None
        
    elif dataset == "triviaqa":
        if 'answer' in sample:
            answers = sample['answer']
            if isinstance(answers, dict):
                if 'value' in answers:
                    return answers['value']
                if 'aliases' in answers and answers['aliases']:
                    return answers['aliases'][0]
            elif isinstance(answers, list) and answers:
                return answers[0]
            return str(answers)
        return None

    elif dataset == "legalbench":
        # LegalBench Yes/No subtasks store the label in `answer` as a string.
        # Normalize to "Yes"/"No" capitalization; pass other values through.
        ans = sample.get('answer')
        if ans is None:
            return None
        s = str(ans).strip()
        if s.lower() == 'yes':
            return 'Yes'
        if s.lower() == 'no':
            return 'No'
        return s

    return None


# GPT-OSS uses OpenAI's "harmony" response format, which interleaves an
# analysis channel and a final-response channel without using <think>...</think>
# tags. The literal token "assistantfinal" delimits the start of the
# committed final response. If we run extraction over the full text, we end up
# matching "Answer:" patterns inside the analysis channel (or eating the whole
# blob), so we strip everything before the LAST "assistantfinal" first. No-op
# for any response that doesn't contain the delimiter.
_HARMONY_FINAL_DELIM = "assistantfinal"


def _strip_harmony_envelope(response: str) -> str:
    """Return only the post-`assistantfinal` portion if present; else pass through.

    Case-insensitive: handles assistantFinal, AssistantFinal, ASSISTANTFINAL, etc.
    GPT-OSS occasionally capitalises the token differently across generation runs.
    """
    if not response:
        return response
    idx = response.lower().rfind(_HARMONY_FINAL_DELIM)
    if idx == -1:
        return response
    return response[idx + len(_HARMONY_FINAL_DELIM):]


def _truncate_to_first_block(response: str) -> str:
    """
    Cut a response after its FIRST completed Answer/Confidence/Correct block.

    Base models often answer correctly and then keep generating — restating the
    template or hallucinating new questions and answering those. Scanning the
    whole response (or taking the LAST "Answer:" match) then grabs the
    continuation's answer or a literal "<YOUR_ANSWER>" placeholder, marking a
    correct answer wrong. Restricting extraction to the first block fixes that;
    within the block last-match behaviour is preserved so mid-solution
    self-correction still works. (Kept self-contained here rather than imported
    from confidence.py to avoid a circular import.)
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


# Refusal / abstention patterns (handoff §19.3). Kept deliberately TIGHT: we
# would rather miss a refusal than mislabel a real answer as one. These match
# the model declining to commit — "please provide the terms", "I can't
# determine … without", "I don't have access" — the texture seen when a model
# (e.g. Gemma2-9B-instruct on TriviaQA) emits an empty "Answer:" line followed
# by a low confidence rating instead of an answer.
_REFUSAL_PATTERNS = [
    r'please provide',
    r'provide the (?:terms|text|document|passage|question|details)',
    r"i need the (?:terms|text|document|passage|context|question)",
    r"i (?:don'?t|do not) have (?:access|enough information|the information)",
    r"i (?:can'?t|cannot|am unable to|could not) (?:determine|answer|find|provide|recall|access|look)",
    r"(?:can'?t|cannot|unable to|couldn'?t) [^.\n]{0,50}? without",
    r"without (?:more|additional|further|the|specific) (?:information|context|details|terms|access)",
    r"need(?: to)? (?:consult|look .* up|access) (?:a |an |the )?(?:reliable )?(?:source|database|encyclopedia)",
    r"struggling to (?:pinpoint|recall|determine|identify|find|name|answer)",
    r"(?:don'?t|do not|can'?t|cannot) (?:recall|remember|know) (?:the|which|what|who|any)",
]
_REFUSAL_RE = re.compile("|".join(_REFUSAL_PATTERNS), re.IGNORECASE)


def is_refusal_response(response: str, extracted_answer) -> bool:
    """Conservative refusal/abstention detector (handoff §19.3).

    Returns True only when BOTH:
      1. extraction produced no usable answer (``extracted_answer`` is None/empty
         — in the live pipeline this is the post-forcing answer, so it already
         encodes "main AND forced both failed"), and
      2. the response text reads like an abstention (matches a refusal pattern).

    If any answer parsed, this is never a refusal. Callers should EXCLUDE
    refusals from accuracy/calibration (or bucket them separately) rather than
    force a coin-flip Yes/No, which would inject noise into the calibration
    signal the study measures.
    """
    if extracted_answer is not None and str(extracted_answer).strip():
        return False
    if not isinstance(response, str) or not response.strip():
        return False
    text = _strip_harmony_envelope(response)
    # Only scan the tail of long responses. Mid-reasoning phrases like "I can't
    # find anything online" or "I don't remember who" fire the patterns even when
    # the model was genuinely attempting to answer. Genuine abstentions appear at
    # or near the end of the response. We use a character-based window (last 350
    # chars) for responses longer than 400 chars, because line-based splitting
    # gives the full text when the response is a single long paragraph — exactly
    # the case where base-model false positives appear.
    if len(text) > 400:
        tail = text[-350:]
    else:
        tail = text
    return bool(_REFUSAL_RE.search(tail))


def extract_model_answer(response: str, dataset: str) -> Optional[str]:
    """
    Extract model answer based on dataset type.

    Handles common model output patterns including:
    - Clean answers: "Answer: 42"
    - Sentence answers: "Answer: The total is 42 dollars."
    - Markdown bold: "**Answer:** 42"
    - Dollar signs and commas: "Answer: $65,960"
    - GPT-OSS harmony format: "<analysis>...assistantfinalAnswer: 42"
    """
    response = _strip_harmony_envelope(response)
    response = _truncate_to_first_block(response)
    
    if dataset == "gsm8k":
        # Priority 1: "Answer:" anchored at start of line (so "in this answer:"
        # mid-CoT doesn't hijack the match); take the LAST match (final commit).
        answer_matches = re.findall(
            r'(?m)^[^a-zA-Z\n]*[Aa]nswer[^a-zA-Z\n]*:\s*([^\n]+)', response
        )
        if answer_matches:
            answer_text = answer_matches[-1]
            # Remove anything after rubric phrases
            answer_text = re.split(
                r'[Cc]onfidence|Almost|Highly|Very good|Likely|Unlikely|Better than|Less than|Chances',
                answer_text
            )[0]
            # Extract the number
            num_match = re.search(r'\$?([\d,]+(?:\.\d+)?)', answer_text)
            if num_match:
                return num_match.group(1).replace(',', '')

        # Priority 2: Common phrasing patterns (kept — legitimate alt commitments)
        patterns = [
            r'[Tt]he answer is:?\s*\$?([\d,]+(?:\.\d+)?)',
            r'[Ff]inal answer:?\s*\$?([\d,]+(?:\.\d+)?)',
            r'=\s*\$?([\d,]+(?:\.\d+)?)\s*$',
            r'####\s*([\d,]+(?:\.\d+)?)',
        ]
        for pattern in patterns:
            match = re.search(pattern, response)
            if match:
                return match.group(1).replace(',', '')

        # NOTE: the previous Priority-3 "last number in response" fallback was
        # removed. On base-model forced calls that echo template fragments like
        # "Confidence: <0-10>", it returned "10" — a confident-looking but
        # fabricated answer. Returning None instead correctly routes the row to
        # answer_extraction_failed=True so it drops out of calibration analysis.
        return None
    
    elif dataset == "mmlupro":
        # Priority 1: "Answer:" line (anchored to start of line; last match wins).
        # Handles: "Answer: B", "Answer: The answer is B.", "**Answer:** B"
        matches = re.findall(
            r'(?m)^[^a-zA-Z\n]*[Aa]nswer[^a-zA-Z\n]*:\s*(.+?)\s*$', response
        )
        if matches:
            letter_match = re.search(r'\(?([A-J])\)?', matches[-1])
            if letter_match:
                return letter_match.group(1).upper()
        
        # Priority 2: Common phrasing
        patterns = [
            r'[Tt]he answer is:?\s*\(?([A-J])\)?',
            r'[Ff]inal answer:?\s*\(?([A-J])\)?',
            r'\b([A-J])\s*(?:is correct|is the answer)',
            r'(?:correct answer is|answer is)\s*\(?([A-J])\)?',
        ]
        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                return match.group(1).upper()
        
        # Priority 3: Last standalone letter
        letters = re.findall(r'\b([A-J])\b', response)
        if letters:
            return letters[-1].upper()
        return None
    
    elif dataset == "strategyqa":
        # Priority 1: "Answer:" line (anchored to start of line; last match wins).
        matches = re.findall(
            r'(?m)^[^a-zA-Z\n]*[Aa]nswer[^a-zA-Z\n]*:\s*(.+?)\s*$', response
        )
        if matches:
            yn_match = re.search(r'\b(Yes|No)\b', matches[-1], re.IGNORECASE)
            if yn_match:
                return yn_match.group(1).capitalize()
        
        # Priority 2: Common phrasing
        patterns = [
            r'[Tt]he answer is:?\s*(Yes|No)',
            r'[Ff]inal answer:?\s*(Yes|No)',
        ]
        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                return match.group(1).capitalize()
        
        # Priority 3: Fallback
        last_lines = "\n".join(response.strip().split("\n")[-3:]).lower()
        if re.search(r'\byes\b', last_lines):
            return "Yes"
        if re.search(r'\bno\b', last_lines):
            return "No"
        return None
    
    elif dataset == "medqa":
        # Priority 1: "Answer:" line (anchored to start of line; last match wins).
        matches = re.findall(
            r'(?m)^[^a-zA-Z\n]*[Aa]nswer[^a-zA-Z\n]*:\s*(.+?)\s*$', response
        )
        if matches:
            letter_match = re.search(r'\(?([A-E])\)?', matches[-1])
            if letter_match:
                return letter_match.group(1).upper()
        
        # Priority 2: Common phrasing
        patterns = [
            r'[Tt]he answer is:?\s*\(?([A-E])\)?',
            r'[Cc]orrect answer:?\s*\(?([A-E])\)?',
            r'\b([A-E])\s*(?:is correct|is the answer)',
        ]
        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                return match.group(1).upper()
        
        # Priority 3: Last standalone letter A-E
        letters = re.findall(r'\b([A-E])\b', response)
        if letters:
            return letters[-1].upper()
        return None
    
    elif dataset == "triviaqa":
        # Priority 1: "Answer:" anchored at start of line, last match wins. Also
        # trim trailing rubric markers in case the answer line is followed by
        # them on the same line (rare but possible).
        matches = re.findall(
            r'(?m)^[^a-zA-Z\n]*[Aa]nswer[^a-zA-Z\n]*:\s*(.+?)\s*$', response
        )
        if matches:
            answer = re.split(r'\*{0,2}[Cc]onfidence|\*{0,2}[Cc]orrect', matches[-1])[0]
            # Cut before model self-commentary that follows on the same line
            # after the committed answer (e.g. "Gregory Peck. I am confident
            # that this is the correct answer because..."). These start with a
            # common sentence-opener after a period. Doesn't affect answers that
            # contain legitimate "." (e.g. "J.K. Rowling") because those aren't
            # followed by "I /My /So /This /It /In ".
            answer = re.split(r'\.\s+(?:I[\s\']|My\s|So\s|This\s|It\s|In\s)', answer)[0]
            answer = answer.strip().rstrip('.')
            # Bare numbers are confidence scores, not trivia answers.
            if answer and not re.match(r'^\d+\.?\d*$', answer):
                return answer

        # Priority 1.5: "Answer:" appearing mid-line after a commit-phrase
        # separator (comma, period, ellipsis). Catches base-model patterns like:
        #   "So overall, Answer: Henry II"
        #   "...my reasoning... Answer: Isle of skye"
        # The separator requirement blocks mid-CoT uses like "in this answer:"
        # (no preceding comma/period). The length cap + prose-opener filter guard
        # against accidentally capturing a continuation clause as the answer.
        matches_15 = re.findall(r'[,;.…]\s*[Aa]nswer\s*:\s*(.+?)(?:\n|$)', response)
        if matches_15:
            answer = re.split(r'\*{0,2}[Cc]onfidence|\*{0,2}[Cc]orrect', matches_15[-1])[0]
            answer = answer.strip().rstrip('.')
            _PROSE_OPENERS = r'^(?:we |i |it |if |so |but |and |or |that |which |when |there |is |are |was |were )'
            if (answer
                    and len(answer) <= 120
                    and not re.match(r'^\d+\.?\d*$', answer)
                    and not re.match(_PROSE_OPENERS, answer, re.IGNORECASE)):
                return answer

        # Priority 2: Common phrasing. Colon is REQUIRED after "final answer"
        # to prevent "my final answer would be... Answer: X" from over-capturing
        # the trailing clause instead of the committed answer after "Answer:".
        # "My answer is:" catches base-model commit phrases that use first-person
        # rather than the structured Answer: line (e.g. idx 16129: "My answer
        # is: strong winds").
        patterns = [
            r'[Tt]he answer is:?\s*(.+?)(?:\n|$)',
            r'[Ff]inal [Aa]nswer:\s*(.+?)(?:\n|$)',
            r'[Mm]y (?:final )?[Aa]nswer is:?\s*(.+?)(?:\n|$)',
        ]
        for pattern in patterns:
            match = re.search(pattern, response)
            if match:
                # rstrip('.') before strip quotes so "answer".'s trailing " is
                # exposed and removed (e.g. My answer is: "strong winds".)
                ans = match.group(1).strip().rstrip('.').strip('"\'')
                # Cut before sentence-level self-commentary after the answer
                # (e.g. "Jasper Fforde. I am 70% confident..." → "Jasper Fforde").
                ans = re.split(r'\.\s+(?:I[\s\']|My\s|So\s|This\s|It\s|In\s)', ans)[0].strip().rstrip('.')
                # "my answer is correct/right" — the match captured a meta-word,
                # not the actual trivia answer; skip and keep trying patterns.
                if re.match(r'^(?:correct|incorrect|right|wrong|true|false|unknown|unsure)$', ans, re.I):
                    continue
                if ans and not re.match(r'^\d+\.?\d*$', ans):
                    return ans
        return None

    elif dataset == "legalbench":
        # Priority 1: "Answer:" line (anchored to start of line; last match wins).
        matches = re.findall(
            r'(?m)^[^a-zA-Z\n]*[Aa]nswer[^a-zA-Z\n]*:\s*(.+?)\s*$', response
        )
        if matches:
            yn_match = re.search(r'\b(Yes|No)\b', matches[-1], re.IGNORECASE)
            if yn_match:
                return yn_match.group(1).capitalize()

        # Priority 2: Common phrasing
        patterns = [
            r'[Tt]he answer is:?\s*(Yes|No)',
            r'[Ff]inal answer:?\s*(Yes|No)',
        ]
        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                return match.group(1).capitalize()

        # Priority 3: Last 3 lines fallback
        last_lines = "\n".join(response.strip().split("\n")[-3:]).lower()
        if re.search(r'\byes\b', last_lines):
            return "Yes"
        if re.search(r'\bno\b', last_lines):
            return "No"
        return None

    return None


def extract_model_answer_strict(response: str, dataset: str) -> Optional[str]:
    """
    Strict answer extraction for SE samples — only accepts answers from
    an explicit "Answer:" line (Priority 1). Does NOT fall back to
    "last number in response" or other heuristics, because those grab
    intermediate CoT reasoning numbers and inflate semantic clusters.

    Priority 1: explicit "Answer:" line
    Priority 2: last 2 lines of response (still strict — avoids mid-CoT numbers)

    Returns None if no clean answer can be found.
    """
    # See note above _strip_harmony_envelope — GPT-OSS responses can hide the
    # committed answer behind an "assistantfinal" delimiter that must be
    # peeled off before any "Answer:" pattern is reliably anchorable.
    response = _strip_harmony_envelope(response)
    response = _truncate_to_first_block(response)
    cleaned = response.replace('*', '')

    if dataset == "gsm8k":
        # Priority 1: explicit "Answer:" line — anchored to start of line so
        # mid-CoT phrases like "in this answer:" don't hijack the match, and
        # findall + [-1] so the model's FINAL commit wins over earlier draft
        # answers it may have written.
        matches = re.findall(
            r'(?m)^[^a-zA-Z\n]*[Aa]nswer[^a-zA-Z\n]*:\s*([^\n]+)', cleaned
        )
        if matches:
            answer_text = matches[-1]
            answer_text = re.split(
                r'[Cc]onfidence|Almost|Highly|Very good|Likely|Unlikely|Better than|Less than|Chances',
                answer_text
            )[0]
            # Capture integers and decimals (e.g. 47.25)
            num_match = re.search(r'\$?([\d,]+(?:\.\d+)?)', answer_text)
            if num_match:
                return num_match.group(1).replace(',', '')
        # Priority 2: last 2 lines — only accept a line that is purely a number
        last_lines = [l.strip() for l in cleaned.strip().splitlines() if l.strip()][-2:]
        for line in reversed(last_lines):
            num_match = re.fullmatch(r'\$?([\d,]+(?:\.\d+)?)', line)
            if num_match:
                return num_match.group(1).replace(',', '')
        return None
    
    elif dataset == "mmlupro":
        matches = re.findall(
            r'(?m)^[^a-zA-Z\n]*[Aa]nswer[^a-zA-Z\n]*:\s*(.+?)\s*$', cleaned
        )
        if matches:
            letter_match = re.search(r'\(?([A-J])\)?', matches[-1])
            if letter_match:
                return letter_match.group(1).upper()
        return None

    elif dataset == "strategyqa":
        matches = re.findall(
            r'(?m)^[^a-zA-Z\n]*[Aa]nswer[^a-zA-Z\n]*:\s*(.+?)\s*$', cleaned
        )
        if matches:
            yn_match = re.search(r'\b(Yes|No)\b', matches[-1], re.IGNORECASE)
            if yn_match:
                return yn_match.group(1).capitalize()
        return None

    elif dataset == "medqa":
        matches = re.findall(
            r'(?m)^[^a-zA-Z\n]*[Aa]nswer[^a-zA-Z\n]*:\s*(.+?)\s*$', cleaned
        )
        if matches:
            letter_match = re.search(r'\(?([A-E])\)?', matches[-1])
            if letter_match:
                return letter_match.group(1).upper()
        return None

    elif dataset == "triviaqa":
        # Anchored, last-match wins, then trim trailing rubric markers if the
        # response continued past the answer.
        matches = re.findall(
            r'(?m)^[^a-zA-Z\n]*[Aa]nswer[^a-zA-Z\n]*:\s*(.+?)\s*$', cleaned
        )
        if matches:
            answer = re.split(r'[Cc]onfidence|[Cc]orrect', matches[-1])[0]
            answer = answer.strip().rstrip('.')
            if answer:
                return answer
        return None

    elif dataset == "legalbench":
        matches = re.findall(
            r'(?m)^[^a-zA-Z\n]*[Aa]nswer[^a-zA-Z\n]*:\s*(.+?)\s*$', cleaned
        )
        if matches:
            yn_match = re.search(r'\b(Yes|No)\b', matches[-1], re.IGNORECASE)
            if yn_match:
                return yn_match.group(1).capitalize()
        return None

    return None


def extract_reasoning(response: str) -> str:
    """
    Extract the reasoning chain from a raw model response by taking
    everything before the final "Answer:" line.
    """
    parts = re.split(r'\n?[Aa]nswer\s*:', response)
    return parts[0].strip() if len(parts) > 1 else response.strip()


def _trivia_norm_nfkd(text: str) -> str:
    """NFKD + strip combining marks → ASCII.

    Handles accented-letter artifacts produced by GPT-OSS, e.g. double-encoded
    UTF-8/Latin-1: "JÃºpiter" → encode latin-1 → decode utf-8 → "Júpiter" →
    NFKD → "Jupiter".
    """
    try:
        text = text.encode('latin-1').decode('utf-8')
    except (UnicodeEncodeError, UnicodeDecodeError):
        pass
    out = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('ascii')
    return re.sub(r'\s+', ' ', out).strip().lower()


def _trivia_norm_space(text: str) -> str:
    """Treat every non-ASCII character cluster as a word separator.

    Handles the narrow-no-break-space artifact: GPT-OSS emits U+202F between
    word components (e.g. "USS Missouri"). The CSV stores/reads those bytes
    mangled as â¯ (two non-ASCII chars). Replacing every non-ASCII cluster with
    a regular space recovers "USS Missouri".
    """
    try:
        text = text.encode('latin-1').decode('utf-8')
    except (UnicodeEncodeError, UnicodeDecodeError):
        pass
    out = re.sub(r'[^\x00-\x7f]+', ' ', text)
    return re.sub(r'\s+', ' ', out).strip().lower()


def _trivia_compact(text: str) -> str:
    """Strip every non-alphanumeric character — last-resort comparator.

    Collapses "G Neisen Au" → "gneisenau" and "moby d ick" → "mobydick" so
    that space-fragmented artifacts still match their aliases.
    """
    return re.sub(r'[^a-z0-9]', '', text.lower())


def check_triviaqa_correct(model_answer: str, sample: dict) -> bool:
    """Special correctness check for TriviaQA (multiple acceptable answers).

    Three-tier comparison to handle GPT-OSS Unicode artifacts:

    Tier 1 — raw lowercase: catches plain-ASCII matches.
    Tier 2 — NFKD normalization (+ encoding-fix attempt): handles double-encoded
             accented characters like "JÃºpiter" → "Jupiter".
    Tier 3 — space normalization + compact: handles narrow-no-break-space artifacts
             like "USSâ¯Missouri" (→ "USS Missouri") and letter-splitting artifacts
             like "Gâ¯Neisenâ¯Au" (→ compact "gneisenau").
    """
    if model_answer is None:
        return False

    model_lower = model_answer.lower().strip()
    model_nfkd = _trivia_norm_nfkd(model_answer)
    model_spaced = _trivia_norm_space(model_answer)
    model_compact = _trivia_compact(model_spaced)

    acceptable = []
    if 'answer' in sample:
        answers = sample['answer']
        if isinstance(answers, dict):
            if 'value' in answers:
                acceptable.append(answers['value'].lower())
            if 'aliases' in answers:
                acceptable.extend([a.lower() for a in answers['aliases']])
            if 'normalized_aliases' in answers:
                acceptable.extend([a.lower() for a in answers['normalized_aliases']])
        elif isinstance(answers, list):
            acceptable.extend([a.lower() for a in answers])

    for acc in acceptable:
        acc_nfkd = _trivia_norm_nfkd(acc)
        acc_spaced = _trivia_norm_space(acc)
        acc_compact = _trivia_compact(acc_spaced)

        # Tier 1: raw
        if model_lower == acc or model_lower in acc or acc in model_lower:
            return True
        # Tier 2: NFKD (handles accented/double-encoded chars)
        if model_nfkd == acc_nfkd or model_nfkd in acc_nfkd or acc_nfkd in model_nfkd:
            return True
        # Tier 3a: space-normalized (handles â¯ space artifacts)
        if model_spaced == acc_spaced or model_spaced in acc_spaced or acc_spaced in model_spaced:
            return True
        # Tier 3b: compact (handles letter-splitting artifacts like "G Neisen Au")
        # Guard: require at least 4 chars so single-letter fragments don't over-match.
        if len(model_compact) >= 4 and len(acc_compact) >= 4:
            if (model_compact == acc_compact
                    or model_compact in acc_compact
                    or acc_compact in model_compact):
                return True

    return False


def answers_match(
    model_answer: Optional[str],
    ground_truth: Optional[str],
    dataset: str,
    sample: Optional[dict] = None,
) -> bool:
    """Robust per-dataset answer comparison.

    Purpose: callers should treat answers as semantically equal when they
    represent the same value, even if the surface strings differ. The
    inline `model_answer == ground_truth` check was treating "6.00" and
    "6" as different on GSM8k, marking correct answers wrong.

    Per-dataset rules:
    - gsm8k: parse both as floats and compare with tolerance. Handles
      "6.00" vs "6", "6.0" vs "6", "06" vs "6", " 6 " vs "6", "1,000" vs
      "1000", "$6" vs "6", "6." vs "6", and small float-precision drift
      from chained calculations.
    - mmlupro / medqa: case-insensitive single-letter compare, after
      stripping whitespace, parentheses, asterisks (markdown bold), and
      trailing periods. Handles "(A)" vs "A", "**A**" vs "A", "a" vs "A".
    - strategyqa: case-insensitive yes/no, with trailing punctuation
      stripped. Handles "yes" vs "Yes", "No." vs "No".
    - triviaqa: delegates to check_triviaqa_correct (alias-aware).
    - any other dataset: stripped string equality.

    Returns False on missing inputs (model_answer is None / "" / ground
    truth is None) — failing closed is consistent with how missing
    extraction is handled upstream.
    """
    if model_answer is None or model_answer == "":
        return False

    if dataset == "triviaqa":
        return check_triviaqa_correct(model_answer, sample) if sample is not None else False

    if ground_truth is None:
        return False

    ma = str(model_answer).strip()
    gt = str(ground_truth).strip()

    if dataset == "gsm8k":
        # Strip currency symbols, comma thousands separators, and whitespace
        # before parsing. extract_model_answer already strips most of these
        # but ground_truth may not — and being defensive on both sides costs
        # nothing.
        def _to_float(s: str):
            cleaned = s.replace(",", "").replace("$", "").strip()
            return float(cleaned)
        try:
            ma_num = _to_float(ma)
            gt_num = _to_float(gt)
        except (ValueError, TypeError):
            # If either side isn't parseable as a number, fall back to
            # whitespace-stripped string equality rather than asserting wrong.
            return ma == gt
        # math.isclose handles "6.00" vs "6" exactly (1e-9 rel tol is plenty
        # for typical GSM8K integer answers) and absorbs tiny float-precision
        # drift like "6.0000000001" that can come out of chained calculations.
        return math.isclose(ma_num, gt_num, rel_tol=1e-9, abs_tol=1e-6)

    if dataset in ("mmlupro", "medqa"):
        # Strip case + common letter wrappers. extract_model_answer already
        # uppercases and unwraps for the primary path, but Priority 2/3
        # fallbacks and ground_truth normalization are less consistent.
        _LETTER_TRIM = "()*. \t\n"
        return ma.upper().strip(_LETTER_TRIM) == gt.upper().strip(_LETTER_TRIM)

    if dataset in ("strategyqa", "legalbench"):
        _YESNO_TRIM = ".!?,;: \t\n"
        return ma.lower().strip(_YESNO_TRIM) == gt.lower().strip(_YESNO_TRIM)

    return ma == gt
