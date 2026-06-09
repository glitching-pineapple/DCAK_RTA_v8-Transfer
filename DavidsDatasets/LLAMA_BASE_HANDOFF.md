# Llama-3.1-8B-Base Pipeline — Confirmed Bugs & Fix Targets

**Date:** 2026-06-08
**Model:** Llama-3.1-8B-base (Llama base path only — instruct is unaffected)
**Status:** Five confirmed bugs in confidence.py. None are fixed yet. All causes are known.

---

## Confirmed Bug 1: `more_likely_than_not` always True

**Root cause: `get_gen2_confidence` has no base model guard (confidence.py:896)**

The function docstring says "Gen 2 verbalized confidence for qwen3" but there is no early return or guard for `MODEL_VARIANT == "base"`. It runs unconditionally for Llama base.

What happens: the full instruction prompt is sent with `base_suffix="\n\nAssessment:"`. The last lines of the prompt the base model sees are:

```
Do NOT write any explanation. Your entire visible response must consist of ONLY these two lines:
Confidence: <1-10>
Correct: Yes or No

Assessment:
```

The base model template-completes "Assessment:" by echoing the format example above it. It outputs something like `"Confidence: 10\nCorrect: Yes"` — filling in `Yes` because it is the first option listed in `"Yes or No"`. `extract_more_likely_than_not` finds `"Correct: Yes"` and returns `True` for every row.

**This is not the model's real self-assessment. It is fill-in-the-blank completion.**

**Fix:** Add a base model guard at the top of `get_gen2_confidence`. Return a null result dict immediately when `MODEL_VARIANT == "base"`:
```python
if MODEL_VARIANT == "base":
    return {"gen2_confidence": None, "gen2_correct": None, "gen2_response": ""}
```
`more_likely_than_not` in evaluation.py should then fall through to `extract_more_likely_than_not` on the main pass response, which correctly reads the model's `"Correct: No"` outputs.

---

## Confirmed Bug 2: `single_pass_correct` always True

**Root cause: `get_correct_separate_base` is Yes-biased for all inputs (confidence.py:841)**

The logit comparison prompt is:
```
Q: {question}
A: {answer}
Q: Is this answer correct? A:
```

For Llama 3.1 8B base, at this completion position the model's pretraining distribution is overwhelmingly skewed toward Yes-variant tokens. In pretraining Q&A corpora, `"Q: Is this answer correct? A: Yes"` is far more common than `"A: No"` — people writing Q&A pairs typically confirm, not deny. The model does not evaluate the answer at all; it predicts the statistically expected next token for this prompt structure.

**CSV confirmation:** 15/15 rows return `single_pass_correct=True`, including row 8983 where the model gave the book title ("The Sea-Wolf") instead of the author name when asked "Who wrote The Sea Wolf." The function called the wrong answer correct.

The fix that replaced always-False overcorrected to always-True. The logit comparison prompt design is the problem, not the comparison mechanism itself.

**Fix options (choose one):**
1. Compare logits for the *answer token itself* vs. a plausible alternative, rather than asking "is this correct?" — the model can't reliably introspect on correctness, but it does have calibrated next-token predictions.
2. Use a contrastive format: give two candidate answers (one correct-looking, one wrong) and ask the model to complete which is right — forces a choice rather than a free yes/no.
3. Accept that `single_pass_correct` is not reliably obtainable from Llama base and set it to `None` for this model path.

Option 3 is the safest until a reliable method is found. Document it as a known limitation rather than shipping a broken True-everywhere signal.

---

## Confirmed Bug 3: `_detect_truncation` misclassifies stop-string endings

**Root cause: no third case for stop-string termination (confidence.py:24)**

Current logic: last token == EOS → `"eos"`, else → `"length"`. Stop-string termination is silently collapsed into `"length"`.

When the two_pass stop string `"\nQ:"` fires for Llama base, generation halts before EOS and before hitting `max_new_tokens`. The last generated token is NOT EOS, so `finish_reason="length"` and `was_truncated=True` are set. Then `expect_confidence_markers=True` additionally checks for `"Confidence:"` and `"Correct:"` — absent from `"10\nQ:"` — making `was_truncated=True` doubly confirmed.

**CSV confirmation:** Every row shows `two_pass_finish_reason=length, two_pass_was_truncated=True` even though the `"\nQ:"` stop string was working correctly.

**Fix:** After generation, check if `outputs.sequences` stopped before `max_new_tokens` for a non-EOS reason. HuggingFace `generate` returns `stopping_criteria` information or you can infer it: if `generated_ids.numel() < max_new_tokens` and `last_id != eos_id`, the stop string fired. Add `"stop"` as a third `finish_reason` and only set `was_truncated=True` for genuine length truncation.

---

## Confirmed Bug 4: Dead `repetition_penalty` code in `generate_with_logits`

**Location: confidence.py:155–188**

`_needs_rep_penalty = False` is hardcoded unconditionally, making the entire downstream path dead:

```python
_needs_rep_penalty = False         # always False
if repetition_penalty is None:
    repetition_penalty = 1.2 if _needs_rep_penalty else 1.0  # always 1.0
...
use_penalty = bool(repetition_penalty) and repetition_penalty != 1.0  # always False
...
if use_penalty:                    # never executes
    gen_kwargs["repetition_penalty"] = repetition_penalty
```

`_needs_rep_penalty`, the `repetition_penalty` conditional, and the `if use_penalty` block are all dead code. No runtime effect, but it's misleading noise.

**Fix:** Delete `_needs_rep_penalty`, the `if repetition_penalty is None` block for it, `use_penalty`, and the `if use_penalty` branch. If `repetition_penalty` is passed in as a non-None argument by a caller, it should still be respected — keep only that path.

---

## Confirmed Bug 5: Two-pass for Llama base always returns None — silently

**Location: confidence.py:1082–1147**

The compact Q&A format:
```
Q: {question}
A: {answer}
Q: Is this answer correct? Rate confidence 1-10.
A:
```
produces `"10\nQ:"` — a bare digit then the stop string fires. The code then calls:
- `extract_verbalized_confidence` — requires `"Confidence: N"` or `"Confidence N"`. Never matches `"10"`. Returns `None`.
- `extract_more_likely_than_not` — requires `"Correct: Yes/No"`. Not present. Returns `None`.

Both always return `None`. The entire two_pass branch for Llama base is theater — it runs a forward pass, stops on the stop string, then silently returns `two_pass_confidence=None, two_pass_correct=None` every time.

This is a known structural limitation: the compact Q&A format does not elicit a labeled confidence rating. The "10" the model emits is also a constant (49/50 StrategyQA rows output "10" regardless of difficulty), so extracting it as a number would not be useful either.

**Fix:** Skip two_pass entirely for Llama base. Add a guard in `get_two_pass_confidence` before the prompt is built:
```python
if MODEL_FAMILY == "llama" and MODEL_VARIANT == "base":
    return {
        "two_pass_confidence": None,
        "two_pass_correct": None,
        "two_pass_critique": "",
        "two_pass_finish_reason": "skipped",
        "two_pass_was_truncated": False,
    }
```
This is honest about the absence of signal rather than running code that will always produce nothing.

---

## Fix Order

Fix in this sequence — each unblocks the next:

1. **Bug 1** (`get_gen2_confidence` guard) — fixes `more_likely_than_not`
2. **Bug 2** (`get_correct_separate_base` / `single_pass_correct`) — fixes or retires the correctness signal
3. **Bug 5** (two_pass guard) — simplifies Llama base path, removes misleading None
4. **Bug 3** (`_detect_truncation`) — fixes metadata columns in CSV
5. **Bug 4** (dead code) — cleanup, no behavioral change

---

## Verification Queries After Fixes

Run on the output CSV after any re-run:

```python
import pandas as pd
df = pd.read_csv("your_output.csv")

# Bug 1 fixed: more_likely_than_not should match "Correct: Yes/No" in full_response
# For rows where full_response contains "Correct: No", mltn should be False
no_in_response = df["full_response"].str.contains(r"Correct:\s*No", regex=True, na=False)
print("Correct:No in response but more_likely_than_not=True (should be 0):")
print((no_in_response & (df["more_likely_than_not"] == True)).sum())

# Bug 2 fixed / retired: single_pass_correct should not be all-True
print("\nsingle_pass_correct value counts:")
print(df["single_pass_correct"].value_counts())

# Bug 5 fixed: two_pass should be fully absent for Llama base
print("\ntwo_pass_confidence non-null (should be 0):", df["two_pass_confidence"].notna().sum())
print("two_pass_finish_reason unique:", df["two_pass_finish_reason"].unique())

# Fallback still working
print("\nverbalized_conf == single_pass_conf:",
    (df["verbalized_confidence"] == df["single_pass_confidence"]).sum(), "of", len(df))
```

Expected healthy output after fixes:
- `Correct:No in response but more_likely_than_not=True`: **0**
- `single_pass_correct` value counts: **mixed True/False** (or all None if option 3 chosen)
- `two_pass_confidence` non-null: **0**
- `two_pass_finish_reason` unique: **`["skipped"]`**
- `verbalized_conf == single_pass_conf`: **all rows**

---

## Key Code Locations

| What | File | Lines |
|------|------|-------|
| `get_gen2_confidence` — needs base guard | confidence.py | 896–978 |
| `get_correct_separate_base` — Yes-biased logit compare | confidence.py | 841–893 |
| `_detect_truncation` — stop-string blindness | confidence.py | 24–60 |
| Dead `repetition_penalty` path | confidence.py | 155–188 |
| Two_pass Llama base compact Q&A — needs skip guard | confidence.py | 1076–1108 |
| `extract_more_likely_than_not` — correct logic, wrong source | confidence.py | 508–528 |
| `generate_with_logits` — guards active / clean forward pass | confidence.py | 84–230 |

---

## What Is Working and Should Not Be Touched

- `verbalized_confidence` falls back to `single_pass_confidence` when `two_pass_confidence=None` — correct behavior, leave it
- `answer_extraction_failed` all False — extraction is healthy
- `extract_verbalized_confidence` — correctly reads `single_pass_confidence` from main pass responses
- `no_repeat_ngram_size=3` anti-loop guard — working correctly for base model over-generation
- Stop strings `["\nQuestion:", "\nAnswer the following", "\nSolution:"]` on main pass — working
- Forced answer path (`was_forced=True` rows) — working correctly, extraction is valid
