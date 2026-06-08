# Llama-3.1-8B-Base Pipeline Problems — Targeted Handoff

**Date:** 2026-06-08  
**Model:** Llama-3.1-8B-base (Llama base path only — instruct is unaffected)  
**Status:** Two fixes applied, still failing on re-run of TriviaQA

---

## The Two Broken Columns

### Column A: `single_pass_correct` — always False even when model is right

**Root cause:** `get_gen2_confidence` sends a dense instruction prompt to the base model:
```
[YOUR reasoning chain... Based on YOUR reasoning... Select EXACTLY ONE confidence level: 1='Almost no chance'...]
State if you think your answer is more likely correct than not after "Correct:" (Yes or No).
...
\n\nAssessment:
```
The base model treats "Correct: Yes or No" as a fill-in-the-blank training pattern and outputs "Correct: No" regardless of whether its answer is right. `extract_more_likely_than_not` reads that literal "Correct: No" and returns `False`. This then sets `single_pass_correct = False` for most rows even on correct answers.

**What was tried (evaluation.py lines 195-201):**
```python
single_pass_correct = gen2["gen2_correct"]
# Override with logit comparison for base models
if MODEL_VARIANT == "base":
    single_pass_correct = get_correct_separate_base(
        model, tokenizer, question, model_answer
    )
```
And fallback guard (lines 210-219):
```python
if single_pass_correct is None:
    if MODEL_VARIANT != "base":
        single_pass_correct = extract_more_likely_than_not(response)
    if single_pass_correct is None and MODEL_VARIANT == "base" and model_answer:
        single_pass_correct = get_correct_separate_base(...)
```

**What `get_correct_separate_base` does (confidence.py line 841):**
Single forward pass, no generation. Prompt: `"Q: {question}\nA: {answer}\nQ: Is this answer correct? A:"`. Compares max logit of [" Yes","Yes"," yes","yes"] vs [" No","No"," no","no"]. Returns `True/False`.

**Why the fix might still be failing:**
1. `MODEL_VARIANT` might not equal `"base"` at runtime — check `config.py` and that it's imported correctly inside the function scope (both files use `from config import MODEL_VARIANT` at call time, not module level)
2. The fix code path requires `model_answer` to be non-None. If extraction is failing → `model_answer=None` → the override block never runs → `single_pass_correct` stays `None`
3. `get_correct_separate_base` itself could be unreliable: for short yes/no StrategyQA answers the prompt is clean, but for multi-word TriviaQA answers the logit comparison competes with the continuation signal of a long answer

---

### Column B: `two_pass_confidence` — always None (every single row)

**Root cause:** The compact Q&A two-pass prompt:
```
Q: {question}
A: {answer}
Q: Is this answer correct? Rate confidence 1-10.
A:
```
The base model completes "A:" with a bare digit "10" (mode-default completion in training data), then the `\nQ:` stop_string fires. `extract_verbalized_confidence` requires a "Confidence: N" label — bare digit is never matched → returns `None`.

**Two-pass confidence is therefore always None for Llama base.** This means `verbalized_confidence` always falls back to `single_pass_confidence` (evaluation.py line 252-253):
```python
verbalized_conf = two_pass_results["two_pass_confidence"]  # always None
if verbalized_conf is None:
    verbalized_conf = single_pass_conf  # this is what gets used
```

**Fix 1 (bare-digit extraction) was added then reverted.** It correctly extracted the "10" but that "10" is a meaningless constant — 49/50 StrategyQA rows output "10" regardless of question difficulty. Extracting it replaced meaningful `single_pass_confidence` variation (3,7,8,9,10) with a constant 10. Reverted.

**Current state is the best achievable with this prompt design.** The compact Q&A format doesn't elicit calibrated ratings from Llama base. `verbalized_confidence = single_pass_confidence` is the correct fallback.

---

## What to Check First on a Fresh Re-Run

Run this after generating any CSV — it will tell you immediately where each column is coming from:

```python
import pandas as pd
df = pd.read_csv("your_output.csv")

# Is single_pass_correct still contaminated?
correct_rows = df[df["is_correct"] == True]
print("is_correct=True but single_pass_correct=False (false negatives):")
print(correct_rows[correct_rows["single_pass_correct"] == False][
    ["idx", "question", "model_answer", "ground_truth", "single_pass_correct", "two_pass_critique"]
])

# Is two_pass always None?
print("\ntwo_pass_confidence non-null:", df["two_pass_confidence"].notna().sum(), "of", len(df))

# Does verbalized_confidence match single_pass_confidence (correct fallback)?
print("\nverbalized_conf == single_pass_conf:",
    (df["verbalized_confidence"] == df["single_pass_confidence"]).sum(), "rows")
```

Expected healthy output:
- False-negative count: **0** (or low — some will be model knowledge errors, not contamination)
- `two_pass_confidence` non-null: **0** (always None for Llama base — expected)
- `verbalized_conf == single_pass_conf`: **all rows** (fallback working correctly)

---

## Key Code Locations

| What | File | Lines |
|------|------|-------|
| Fix 2 — override gen2_correct for base | evaluation.py | 195-201 |
| Fix 3 — guard fallback path for base | evaluation.py | 210-219 |
| `get_correct_separate_base` (logit compare) | confidence.py | 841-893 |
| `get_gen2_confidence` (sends OOD prompt to base) | confidence.py | 896-978 |
| Compact Q&A two-pass prompt for Llama base | confidence.py | 1082-1088 |
| `generate_simple_response` (base_suffix path) | model_utils.py | 55-102 |

---

## Hypotheses for Why Fixes Still Fail

**Most likely: `MODEL_VARIANT` is not `"base"` at runtime**  
Both fix guards are `if MODEL_VARIANT == "base"`. If `config.py` has `MODEL_VARIANT = "instruct"` or the import resolves differently at call time, neither fix triggers and the original contamination path runs. Verify:
```python
from config import MODEL_VARIANT, MODEL_FAMILY
print(MODEL_VARIANT, MODEL_FAMILY)  # must print "base" "llama"
```

**Second most likely: `model_answer` is None for more rows than expected**  
If `extract_model_answer` / forced-answer fallback is failing → `model_answer = None` → the `if model_answer:` guard on line 185 skips all of Gen 2 → `single_pass_conf = None`, `single_pass_correct = None`. With both None, the fallback on line 204-219 runs but `if model_answer:` on line 216 is also False → still no logit comparison. Net: `single_pass_correct = None` forever for those rows. Check `answer_extraction_failed` column prevalence.

**Third: `get_correct_separate_base` is unreliable for TriviaQA open-domain answers**  
For long factual answers (e.g., "Montserrat" → correct, "Mozambique" → incorrect), the logit comparison at "Is this answer correct? A:" may be biased toward "No" if the model has uncertain knowledge about the fact. This is a knowledge limitation, not a code bug, but it looks like `single_pass_correct=False` in the CSV and is indistinguishable from contamination.

---

## What Has NOT Been Fixed and Is a Known Limitation

- **Two-pass for Llama base is structurally uninformative.** The compact Q&A format makes the model output "10" as a default. No extraction fix will make this a useful calibration signal. If two-pass confidence matters for Llama base, the prompt needs a completely different design (not a Q&A continuation; possibly a logit comparison similar to `get_correct_separate_base`).
- **`single_pass_confidence` for Llama base comes from the main response or a separate verbalized call** — it varies (3,7,8,9,10) and is currently the best confidence signal available for this model. Protect it.
