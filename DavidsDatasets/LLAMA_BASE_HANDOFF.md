# Llama-3.1-8B-Base Pipeline — Active Bug + Fix History

**Date:** 2026-06-08
**Model:** Llama-3.1-8B-base
**Status:** 5 previous bugs fixed and confirmed working. One new bug identified from second CSV run. Not yet fixed.

---

## Previously Fixed — Confirmed Working in CSV 2

| Fix | What changed | CSV 2 confirmation |
|-----|-------------|-------------------|
| `get_gen2_confidence` base guard | Returns null dict for base models | `more_likely_than_not` no longer always True |
| `get_correct_separate_base` retired | Returns None unconditionally | `single_pass_correct` no longer always True |
| `get_two_pass_confidence` skip guard | Returns skipped dict for Llama base | `two_pass_finish_reason=skipped`, `two_pass_was_truncated=False` on all rows |
| `_detect_truncation` stop-string case | Added `"stop"` finish reason | Main pass rows correctly show `eos` |
| Dead `repetition_penalty` code | Deleted | No behavioral change, code is clean |

Do not re-examine or re-fix any of the above.

---

## Active Bug: `more_likely_than_not` and `single_pass_correct` are null for ~33% of rows

### What the CSV shows

In CSV 2, five rows have blank values for both `more_likely_than_not` and `single_pass_correct`:

| idx | is_correct | was_forced | more_likely_than_not | single_pass_correct |
|-----|-----------|-----------|---------------------|---------------------|
| 12270 | True | True | **null** | **null** |
| 4809 | True | True | **null** | **null** |
| 12242 | True | True | **null** | **null** |
| 12587 | True | True | **null** | **null** |
| 227 | True | True | **null** | **null** |

**Pattern:** every null row is `was_forced=True`. Every forced row is null.

### Root cause

All five rows have `main_pass_finish_reason=eos` — the model completed naturally. But it buried its answer in prose rather than writing `"Answer: X"` on its own line, so the answer extractor failed and a forced pass was needed.

These prose-style responses also never reach the `"Correct: Yes/No"` template line. Their responses end with things like:
- `"So my final answer is 'The iron lady' with a confidence rating of 7/10."`
- `"I am confident that the correct English name is Florence."`
- `"My confidence level is 5."`

No `"Correct:"` line → `extract_more_likely_than_not` returns `None` → `more_likely_than_not=null` and `single_pass_correct=null`.

Before the fixes, `get_gen2_confidence` (with no base guard) filled in a fake `True` for these rows by template-completing `"Assessment:"`. That fake signal is now correctly gone. But nothing replaced it, leaving null.

### Diagnosis — completed 2026-06-08

**Q1: Is there any fallback after `extract_more_likely_than_not` returns `None` for base models?**

No. The final assignment attempt is at `evaluation.py` lines 219–220:

```python
if single_pass_correct is None and MODEL_VARIANT == "base":
    single_pass_correct = extract_more_likely_than_not(response)
```

After this returns `None`, there is no further fallback. `single_pass_correct` stays `None`.

**Q2: Same question for `more_likely_than_not`?**

Same answer. Lines 252–256:

```python
more_likely = two_pass_results["two_pass_correct"]   # None (two-pass skipped for base)
if more_likely is None:
    more_likely = single_pass_correct                 # None for these rows
```

`more_likely_than_not` ends up `None`.

**Q3: Does the `was_forced` path skip `extract_more_likely_than_not`?**

No — this is **structural, not coincidental, but not a routing skip.** The call at line 220 is reached for ALL base model rows regardless of `was_forced`. The correlation is structural because the same template deviation that causes forcing (model didn't write `"Answer: X"`) also causes the response to omit the `"Correct: Yes/No"` footer. The function runs; it finds no parseable signal in the prose-style response and returns `None` legitimately.

**Decision: Option A — keep `None`.**

The code is already implementing Option A correctly. `extract_more_likely_than_not` IS called for forced rows; it returns `None` honestly because those responses don't contain the structured footer. No logic change is needed.

**Side observation (no action):** `get_correct_separate_base` is called twice for every base model row with a `model_answer` — once at line 198 and again at line 213. Both calls return `None` for Llama base, so this is wasteful but not incorrect. Left alone per the "don't touch previously fixed functions" constraint.

**Fix applied:** Added a comment at line 219–220 documenting the known `None` case so future readers don't mistake it for a dead code path or an oversight.

### What NOT to do

Do not re-introduce any logit comparison that outputs Yes/No by asking "is this answer correct?" — that prompt is Yes-biased for Llama base regardless of the answer. This was confirmed as the root cause of the always-True problem in the prior session.

---

## Verification Queries

After any fix, run on the output CSV:

```python
import pandas as pd
df = pd.read_csv("your_output.csv")

# Previously fixed — must still hold
print("two_pass still skipped:", (df["two_pass_finish_reason"] == "skipped").all())
print("two_pass_was_truncated any True:", df["two_pass_was_truncated"].any())

# New bug — check null coverage
print("\nmore_likely_than_not null count:", df["more_likely_than_not"].isna().sum(), "of", len(df))
print("single_pass_correct null count:", df["single_pass_correct"].isna().sum(), "of", len(df))

# Null rows should correlate with was_forced
null_mltn = df["more_likely_than_not"].isna()
print("\nNull more_likely_than_not rows — was_forced values:")
print(df[null_mltn]["was_forced"].value_counts())

# Correct source: rows with "Correct: No" in response should have more_likely_than_not=False
no_in_response = df["full_response"].str.contains(r"Correct:\s*No", regex=True, na=False)
print("\nCorrect:No in response but more_likely_than_not=True (should be 0):")
print((no_in_response & (df["more_likely_than_not"] == True)).sum())

# Fallback still working
print("\nverbalized_conf == single_pass_conf:",
    (df["verbalized_confidence"] == df["single_pass_confidence"]).sum(), "of", len(df))
```

---

## Key Code Locations

| What | File | Where to look |
|------|------|--------------|
| `more_likely_than_not` assignment for base | evaluation.py | Find the base model path after gen2 guard |
| `single_pass_correct` fallback routing | evaluation.py | Find the block added in last session's routing fix |
| `extract_more_likely_than_not` | confidence.py | 508–528 |
| `get_gen2_confidence` base guard | confidence.py | ~896 (fixed — do not change) |
| `get_correct_separate_base` (returns None) | confidence.py | ~850 (fixed — do not change) |
| `get_two_pass_confidence` skip guard | confidence.py | ~993 (fixed — do not change) |

---

## What Is Working and Should Not Be Touched

- `two_pass` correctly skipped for Llama base — `finish_reason=skipped`, `was_truncated=False`
- `extract_more_likely_than_not` correctly reads `"Correct: Yes/No"` from main pass responses
- `verbalized_confidence` correctly falls back to `single_pass_confidence` when `two_pass_confidence=None`
- `single_pass_confidence` numeric 1–10 rating varies meaningfully and is the best confidence signal available
- `answer_extraction_failed` all False — extraction healthy
- Forced answer path working correctly
