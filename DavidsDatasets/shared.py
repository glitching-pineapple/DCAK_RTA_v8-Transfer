# shared.py - Single source of truth for text constants and extraction helpers
# that were previously copy-pasted across confidence.py / data_utils.py /
# evaluation.py. Duplication here is not cosmetic: a fix landing in one copy
# but not another silently desynchronizes prompt text from extractor behavior.
#
# This module must stay dependency-free within the project (stdlib `re` only)
# so anything can import it without circular-import risk.

import re

# ---------------------------------------------------------------------------
# GPT-OSS "harmony" response format
# ---------------------------------------------------------------------------
# GPT-OSS interleaves an analysis (reasoning) channel and a final-response
# channel without <think> tags. The literal token "assistantfinal" delimits
# the start of the committed final response; everything before the LAST
# occurrence is analysis-channel text and must not reach the answer/confidence
# extractors.
HARMONY_FINAL_DELIM = "assistantfinal"

# Harmony analysis-channel marker: text starting with this is not a real
# answer. No \b — the channel name runs directly into the next word
# ("analysisWe…").
ANALYSIS_MARKER_RE = re.compile(r'^analysis', re.IGNORECASE)

# Qwen3-style reasoning envelope.
THINK_BLOCK_RE = re.compile(r'<think>.*?</think>', re.DOTALL)


def strip_harmony_envelope(response: str) -> str:
    """Return only the post-`assistantfinal` portion if present; else pass through.

    Case-insensitive: handles assistantFinal, AssistantFinal, ASSISTANTFINAL,
    etc. — GPT-OSS occasionally capitalises the token differently across runs.
    """
    if not response:
        return response
    idx = response.lower().rfind(HARMONY_FINAL_DELIM)
    if idx == -1:
        return response
    return response[idx + len(HARMONY_FINAL_DELIM):]


def truncate_to_first_block(response: str) -> str:
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


# ---------------------------------------------------------------------------
# The 10-class verbalized-confidence rubric
# ---------------------------------------------------------------------------
# Used in THREE prompts (first-pass CONF_RUBRIC, Gen 2 self-rating, Gen 3
# blinded critique). Previously inlined in each — the exact hazard this module
# exists to remove: editing one copy would silently break cross-prompt
# comparability. Class N is defined as "(N-1)*10% to N*10% likely correct";
# calibration analysis maps a reported N to the interval midpoint (N-0.5)/10.
RUBRIC_BULLETS = '''- 1 = "Almost no chance" (0-10% likely correct)
- 2 = "Highly unlikely" (10-20% likely correct)
- 3 = "Chances are slight" (20-30% likely correct)
- 4 = "Unlikely" (30-40% likely correct)
- 5 = "Less than even" (40-50% likely correct)
- 6 = "Better than even" (50-60% likely correct)
- 7 = "Likely" (60-70% likely correct)
- 8 = "Very good chance" (70-80% likely correct)
- 9 = "Highly likely" (80-90% likely correct)
- 10 = "Almost certain" (90-100% likely correct)'''

# First-pass rubric block, byte-identical to the original _CONF_RUBRIC in
# confidence.py (leading newline, trailing double newline included).
CONF_RUBRIC = f"""
Then, thoroughly assess your confidence in that answer by evaluating your thinking process so far. Finally, classify your confidence into one of the following classes based on how likely your answer is to be correct, by selecting EXACTLY ONE:

{RUBRIC_BULLETS}

"""
