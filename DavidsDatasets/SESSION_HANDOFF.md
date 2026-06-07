# Session Handoff — DCAK RTA v8 Transfer (DavidsDatasets)
**Last updated:** 2026-05-13  
**Most recent model evaluated:** Gemma 4 31B IT (`gemma4` family, instruct) — see `mmlupro_confidencewithnewSE_Gemma4-31B-instruct.csv`  
**Newest models configured (not yet run):** GPT-OSS-20B (`gptoss` family, `openai/gpt-oss-20b`); Gemma 4 31B base (`gemma4` family, base variant, `google/gemma-4-31b`)  
**Primary dataset:** MMLU-Pro (`mmlu35b.csv`, `21mmlupro_confidencewithnewSE_*.csv`)  
**Working directory:** `/Users/davidzhu/Documents/GitHub/DCAK_RTA_v8-Transfer/DavidsDatasets/`

---

## Project Overview

This project evaluates LLM uncertainty/confidence metrics on benchmark datasets (GSM8K, MMLU-Pro, StrategyQA, MedQA, TriviaQA). It measures:

- **Logit-based confidence**: sequence log-prob sum, min token prob, geometric mean, arithmetic mean
- **Verbalized confidence**: 1–10 scale elicited from the model directly (single-pass and two-pass critique)
- **Answer-token logit entropy (ATE)**: Shannon entropy over answer-letter distribution at the answer token position (MCQ datasets only)
- **Semantic entropy (SE)**: Samples N answers at temperature 0.5, clusters by NLI bidirectional entailment, computes entropy over cluster probabilities

---

## Prior Session Summary (context inherited)

A prior session added:
- Qwen3 model family support (`config.py`, `model_utils.py`)
- ATE implementation (`confidence.py::extract_answer_token_entropy`)
- SE sampling with model-conditional token budgets
- Qwen3 `<think>...</think>` stripping before answer/confidence parsing
- The `use_safetensors=True` fix required for loading Qwen3.6-35B-A3B on H200s

The prior session's token budgets were: `MAX_NEW_TOKENS=4096` and `SE_MAX_NEW_TOKENS=2048` for qwen3. These proved insufficient (see below).

---

## This Session: Problem Diagnosis

### CSV Symptoms Observed in `mmlu35b.csv`

Certain rows (notably indices 10474, 9157, and others) had blank or wrong values:

| Column | Problem |
|--------|---------|
| `verbalized_confidence` | Blank |
| `single_pass_confidence` | Blank |
| `more_likely_than_not` | Blank |
| `single_pass_correct` | Blank |
| `two_pass_critique` | Blank |
| `answer_token_entropy` | `inf` (should be `nan` or a valid float) |
| `top_answer_letter` | Blank |
| `chosen_answer_raw_prob` | Blank |
| `prob_A` through `prob_J` | Blank |
| `sampled_answers` | `[]` |
| `se_extraction_failure_rate` | `1.0` |
| `semantic_entropy` / `predictive_entropy` | `inf` |

### Root Cause: Token Budget Exhaustion

Qwen3 is a **thinking model** — it emits a `<think>...</think>` block before structured output. These blocks consume 1,000–3,000+ tokens regularly. With `MAX_NEW_TOKENS=4096`, the budget was exhausted before the model ever reached `Confidence: N` or `Correct: Yes/No`.

Consequences:
- `verbalized_confidence` / `single_pass_confidence`: extraction returned `None` (nothing generated)
- `answer_token_entropy`: the "Answer: X" token was sometimes never reached → `float('inf')` stored (semantically wrong; now corrected to `float('nan')`)
- `sampled_answers = []` / `se_extraction_failure_rate = 1.0`: SE sampling with `SE_MAX_NEW_TOKENS=2048` also ran out before "Answer: X" was emitted, causing all 5 sample extractions to fail

### Secondary Issue: Inflated Same-Generation Confidence (Design Problem)

Even when verbalized confidence was extracted, the original single-generation approach had a known bias: the model would generate a coherent reasoning chain, immediately rate that chain highly (sunk-cost effect), and produce inflated confidence scores. A two-pass critique existed to mitigate this, but for Qwen3 it was also failing due to token limits.

---

## This Session: Architectural Redesign — Three-Generation Flow (qwen3 only)

The core redesign separates what was one generation into three distinct calls.

### Gen 1 — Reasoning + Answer Only
- **Prompt:** reasoning instructions + `Answer: X` format. **No confidence rubric, no `Confidence:/Correct:` output request.**
- **Token budget:** 8,192 (increased from 4,096)
- **Output stored in:** `full_response`
- `<think>...</think>` block is stripped before parsing

### Gen 2 — Own-Work-Aware Verbalized Confidence
- **Prompt:** "The following is YOUR OWN reasoning chain that YOU previously produced. Based on YOUR reasoning, how confident are you that YOUR answer is correct?"
- Model is explicitly told the work is its own (to activate self-reflection rather than detached evaluation)
- Short generation (max 512 tokens) — cannot be truncated
- Extracts `Confidence: N` and `Correct: Yes/No`
- **Output stored in:** `single_pass_confidence`, `single_pass_correct`
- Reasoning trimmed to 3,000 chars (Gen 2 benefits from more detail since it is own-work-aware)

### Gen 3 — Blinded Two-Pass Critique
- **Prompt:** "You are reviewing a solution submitted by **someone else**..." — model is NOT told this is its own work
- Includes Gen 2's self-reported score as context ("The respondent self-assessed: Confidence X/10, More likely correct: Yes/No") so the reviewer can push back on it
- Model independently critiques and re-rates
- Max 1,024 tokens (increased from 512)
- **Output stored in:** `two_pass_critique`, and `verbalized_confidence` / `more_likely_than_not` (primary)
- Reasoning trimmed to 2,000 chars (blinded reviewer gets a summary; longer bloats the critique prompt)
- Fallback: if Gen 3 fails extraction, falls back to Gen 2 values

### Why This Design Works

| Problem | Solution |
|---------|----------|
| Token limits cut off verbalized confidence | Gen 2 is a short dedicated call — always completes |
| Inflated same-generation confidence | Gen 2 is already a separate call; Gen 3 is blinded to authorship |
| Two-pass critique was also failing | Gen 3 input is compact (2,000-char trim) and well within budget |

### Acknowledged Limitation: Internal State Abstraction

By generating verbalized confidence (Gen 2) as a separate forward pass, the model reads its own reasoning as **text** rather than from the **internal hidden states** that were active during Gen 1. Subtle uncertainty signals embedded in activations are not accessible in Gen 2.

This is an acknowledged methodological limitation. The logit-based metrics (`seq_confidence_mean`, `logit_confidence_geom`, etc.) — computed from Gen 1's actual token probabilities — partially compensate by capturing internal-state signals. The combination of logit metrics (internal) + Gen 2 verbalized confidence (semantic retrospective) + Gen 3 blinded critique gives three complementary signals.

### Other Model Families
The original single-generation flow is **unchanged** for `qwen`, `llama`, `gemma`. The three-generation branch is gated strictly on `MODEL_FAMILY == "qwen3"`.

---

## Temporary Speed Settings (Restore Before Production)

These are intentional short-term settings for fast debugging runs:

```python
# config.py
SE_NUM_SAMPLES = 1          # Restore to 5 for full runs
SKIP_NLI_CLUSTERING = True  # Set False to re-enable DeBERTa NLI clustering
```

With `SKIP_NLI_CLUSTERING = True`, the SE computation block in `evaluate_sample()` is skipped entirely. SE columns will be absent from results. With `SE_NUM_SAMPLES = 1`, semantic entropy would be meaningless even if clustering were enabled.

---

## Complete List of Code Changes This Session

### `config.py`
| Setting | Before | After | Reason |
|---------|--------|-------|--------|
| `MAX_NEW_TOKENS` (qwen3) | 4,096 | 8,192 | Gen 1 reasoning chains can exceed 4,096 tokens |
| `SE_MAX_NEW_TOKENS` (qwen3) | 2,048 | 4,096 | SE samples also hit token limits before "Answer: X" |
| `SE_NUM_SAMPLES` | 5 | 1 | Temporary speed setting |
| `SKIP_NLI_CLUSTERING` | (didn't exist) | `True` | Temporary: bypass slow DeBERTa NLI clustering |

### `confidence.py`

**`create_prompt()` — `include_confidence: bool = True` parameter added**
- When `False` (used for qwen3 Gen 1): all confidence rubric text and `Confidence:/Correct:` output instructions are stripped from the prompt for all 5 datasets
- When `True` (default, all other models): original prompt unchanged

**`get_gen2_confidence()` — new function**
- Implements Gen 2: presents reasoning back to the model as its own work, asks for verbalized confidence + MLN
- Returns `{"gen2_confidence", "gen2_correct", "gen2_response"}`

**`get_two_pass_confidence()` — updated**
- New optional params: `gen2_confidence`, `gen2_correct`
- When provided (qwen3), the critique prompt frames the work as anonymous and includes the self-reported Gen 2 score as context
- `max_new_tokens` increased 512 → 1,024
- Framing changed from "you are reviewing a solution" → "submitted by someone else" to make anonymity explicit

**Refactor/cleanup (simplify pass):**
- `_format_choices(choices)` extracted as module-level helper — replaced 6 inline `"\n".join([f"{chr(65+i)}. {c}" ...])` repetitions
- `_CONF_RUBRIC` moved to module level — was being reconstructed as a local variable inside `create_prompt()` on every call
- Stale `import re as _re` removed from inside `extract_answer_token_entropy()` (module-level `import re` already available)
- Documented intentional trim asymmetry with inline comments: Gen 2 trims at 3,000 chars, Gen 3 at 2,000 chars

### `evaluation.py`

**Flow changes:**
- Imports `SKIP_NLI_CLUSTERING` and `get_gen2_confidence`
- `evaluate_sample()` branches on `MODEL_FAMILY == "qwen3"` for 3-generation flow; all other models use original single-generation flow
- SE computation now gated: `if compute_semantic_entropy and semantic_calculator is not None and not SKIP_NLI_CLUSTERING`
- `float('inf')` → `float('nan')` for all SE metrics when extraction fails (< 2 valid samples) — `inf` was semantically wrong; `nan` correctly signals "not computable"

**Refactor/cleanup (simplify pass):**
- `_QWEN3_THINK_RE = re.compile(r'<think>.*?</think>', re.DOTALL)` added at module level — previously recompiled on every call via `re.sub(...)`, which during SE sampling happened up to 5× per question
- Both `<think>` stripping uses now call `_QWEN3_THINK_RE.sub('', ...)`
- `token_probs: list = []` safety init added before the qwen3/other if/else branch — ensures `compute_confidence_metrics(token_probs)` always has a defined variable regardless of branch

### `model_utils.py`

**`generate_simple_response()` — new shared helper**
- Handles instruct vs. base model chat template formatting + tokenize + generate + decode
- Used by `get_verbalized_confidence_separate()`, `get_gen2_confidence()`, and `get_two_pass_confidence()` — replaces identical 12-line boilerplate that existed independently in all three

---

## Column Mapping Under New Architecture (qwen3)

| CSV Column | Source Under New Architecture |
|------------|-------------------------------|
| `full_response` | Gen 1 output (think block stripped) |
| `single_pass_confidence` | Gen 2 extracted confidence |
| `single_pass_correct` | Gen 2 extracted MLN judgment |
| `two_pass_critique` | Gen 3 raw response text |
| `verbalized_confidence` | Gen 3 extracted confidence (falls back to Gen 2 if None) |
| `more_likely_than_not` | Gen 3 extracted judgment (falls back to Gen 2 if None) |
| `seq_confidence_mean` / `logit_*` | Computed from Gen 1 token probabilities |
| `answer_token_entropy` / `prob_*` | Computed from Gen 1 raw logit scores at answer token |
| SE columns | Skipped while `SKIP_NLI_CLUSTERING=True` |

---

## Known Limitations & Open Items

1. **Internal state abstraction** (acknowledged): Gen 2 verbalized confidence is a retrospective text-based assessment, not drawn from Gen 1's internal activations. Logit metrics partially compensate.

2. **SE settings must be restored before analysis:** `SE_NUM_SAMPLES=1` and `SKIP_NLI_CLUSTERING=True` make SE columns absent or meaningless. Restore both before any semantic entropy analysis.

3. **SE may still fail on very hard questions:** Even at 4,096 tokens for SE sampling, extremely complex Qwen3 think blocks could still exhaust the budget. Monitor `se_extraction_failure_rate` after restoring full SE settings.

4. **3× inference time for qwen3:** Three sequential generations per question. No parallelism is possible here since Gen 2 depends on Gen 1 output and Gen 3 depends on Gen 2.

5. **`answer_token_entropy` still returns `nan` if answer token unreachable:** If even Gen 1 with 8,192 tokens fails to emit "Answer: X", ATE will be `nan`. This should be rare at 8,192 tokens but is worth monitoring.

---

## Files Modified This Session

| File | Changes |
|------|---------|
| `config.py` | Token limits, `SE_NUM_SAMPLES`, `SKIP_NLI_CLUSTERING` |
| `confidence.py` | `create_prompt` include_confidence param, `get_gen2_confidence`, updated `get_two_pass_confidence`, module-level helpers, cleanup |
| `evaluation.py` | qwen3 3-gen branch, SE guard, `nan` fix, compiled regex, `token_probs` init |
| `model_utils.py` | `generate_simple_response` shared helper |

## Files NOT Modified This Session

| File | Status |
|------|--------|
| `semantic_entropy.py` | Unchanged — clustering bypass handled in `evaluation.py` |
| `data_utils.py` | Unchanged |
| `main.py` | Unchanged |
| `save_utils.py` | Unchanged |
| `visualization.py` | Unchanged |

---

## File Map (Full Pipeline)

| File | Purpose |
|------|---------|
| `config.py` | All hyperparameters, model family selection, dataset selection, token budgets |
| `model_utils.py` | Model + tokenizer loading; `generate_simple_response` helper |
| `data_utils.py` | Dataset loading + answer extraction (regex parsers per dataset) |
| `confidence.py` | `generate_with_logits`, ATE, verbalized confidence, Gen 2, Gen 3 critique |
| `evaluation.py` | Per-sample evaluation orchestration; qwen3 vs. other-model branching |
| `semantic_entropy.py` | SE calculator: NLI clustering, sampling, entropy computation |
| `main.py` | Entry point — loops over N_SAMPLES, aggregates results, saves CSV |
| `save_utils.py` | CSV + JSON save; filename: `{DATASET}_confidence_{label}.csv` |
| `visualization.py` | AUROC, calibration curves, plots |

---

# Session 2026-05-10 — Rubric Harmonization, Truncation Handling, Merge Resolution

This session addressed three things: (1) the verbalized-confidence rubric was inconsistent across datasets (MMLU Pro had no rubric at all; StrategyQA/TriviaQA used a different scale than the others), (2) Qwen3 was truncating frequently on hard MMLU Pro math questions and producing garbage answers from `extract_model_answer`'s last-letter fallback, and (3) the working tree had unresolved Git merge conflicts blocking any run.

## 1. Rubric harmonization across all 5 datasets

**Before:** Different datasets showed different rubrics in their first-pass prompts:
- GSM8K, MedQA: numbered `1 = "Almost no chance" (0-10% likely correct)` style (`_CONF_RUBRIC`)
- StrategyQA, TriviaQA: bulleted `- "Almost no chance" (0.0–0.1)` ranges, then asked for `Confidence: <1-10>` output (scale mismatch)
- **MMLU Pro: no rubric** — only said "Rate your confidence as an integer from 1 to 10"

**After:** All 5 datasets share `_CONF_RUBRIC` (a single source of truth in `confidence.py`). Format chosen by the user (Option 3):

```
- 1 = "Almost no chance" (0-10% likely correct)
- 2 = "Highly unlikely" (10-20% likely correct)
…
- 10 = "Almost certain" (90-100% likely correct)
```

Bulleted, leading 1-10 number, percentage descriptors kept. The same bulleted format was also propagated to:
- `get_gen2_confidence` (qwen3 self-rating prompt)
- `get_two_pass_confidence` (blinded critique prompt)

So every confidence elicitation — first-pass, Gen 2, two-pass critique — now uses the same 10 verbal labels with the same percentage ranges.

Each first-pass prompt also now ends with a strict format reminder:
```
The Confidence number MUST match the class you selected — for example, if you select
"Better than even" you MUST write Confidence: 6, not any other number.
```

**Implication:** Any verbalized-confidence column from runs prior to this session reflects a different prompt and is not directly comparable. **All 5 datasets should be re-run** to get clean rubric-aligned scores. Logit metrics on GSM8K/MedQA shift only slightly (cosmetic format change); MMLU Pro/StrategyQA/TriviaQA shift more substantially because their first-pass prompt content actually changed.

## 2. Truncation detection + forced-answer fallback

**Problem:** Qwen3's `<think>` block on hard MMLU Pro math/physics problems regularly exceeds 8,192 tokens. The main pass would hit the token cap mid-reasoning, never emit `Answer: X`, and `extract_model_answer`'s Priority-3 fallback ("last standalone letter A-J in the response") would return whatever letter happened to appear in the chain of thought (e.g., "I" from "First, **I** need to recall..."). Example: `21mmlupro` row idx=11717 had `model_answer="I"` despite no commitment in the response.

**Decision:** Force a final answer rather than dropping the row. Reasoning:
- Calibration analysis needs paired (answer, confidence) on every sample
- The dropout would be **non-random** — only hard math truncates — biasing calibration toward easy samples
- A forced answer is itself a calibration signal: well-calibrated models should give *low* verbalized confidence on forced guesses

**Implementation:**

| Component | Change |
|---|---|
| `confidence.py::generate_with_logits` | Returns 5-tuple now, ending with `meta = {"finish_reason", "was_truncated"}`. Uses the existing `_detect_truncation` helper. |
| `confidence.py::get_forced_answer` | New function. When the main pass truncates, it shows the model the (incomplete) reasoning + question + choices and asks for ONLY the answer in the dataset's expected format. Tight token budgets (8 for letters/Yes-No, 16 for GSM8K numbers, 32 for TriviaQA). Uses `extract_model_answer` to parse. |
| `evaluation.py::evaluate_sample` | When `main_meta["was_truncated"]` is true, calls `get_forced_answer` and overwrites the unreliable Priority-3 fallback. Sets `was_forced=True`. |

**New columns in the result CSV:**
- `main_pass_finish_reason` — `"eos"` or `"length"`
- `main_pass_was_truncated` — bool
- `was_forced` — bool, true when `get_forced_answer` produced the model_answer
- `forced_answer_response` — raw text of the forced-answer call (None if no truncation)

These supplement the pre-existing `two_pass_finish_reason` / `two_pass_was_truncated` columns added by the parallel branch (which tracks truncation on the two-pass critique).

**Filtering convention for analysis:** to compare {forced vs. organic}, partition rows on `was_forced`. To compute "honest" accuracy, you can exclude `was_forced=True` rows; for calibration analysis, keep them in.

## 3. Merge conflict resolution

The working tree had unresolved Git conflict markers in three files (HEAD vs commit `9a721c8`). Conflicts resolved as follows:

**`evaluation.py` (4 conflicts):**
- Variable init: kept BOTH `_empty_two_pass` template (parallel branch) AND `was_forced`/`forced_response` init (this session).
- qwen3 generate: kept `main_meta` naming (more descriptive than `gen_info`) AND the parallel branch's `reasoning_for_critique` extraction. The latter is a real win — pulls `<think>` content out and feeds it to the critic instead of just the answer line. Without this, qwen3 critics confabulate ("the reasoning is sound") on easy questions and abdicate ("no detailed reasoning, confidence 1") on hard ones.
- non-qwen3 generate: kept `main_meta` naming for consistency.
- Result dict: merged both column sets (truncation + forced-answer columns from this session, two-pass-truncation columns from parallel branch).

**`confidence.py` (1 latent bug after merge):**
- The merge had left `generate_with_logits` returning **4 things** while `evaluation.py` was unpacking **5** — would have crashed on every sample at runtime. Fixed by making it return the `meta` dict via `_detect_truncation`.

**`data_utils.py` (1 conflict):**
- Used the parallel branch's for-loop refactor for the medqa branch (cleaner than the if/elif version).
- Side fix: an indentation error from prior corruption was already preventing import.

## 4. New verification script — `verify_rubric.py`

Self-contained, runs without torch/numpy/transformers/datasets installed (stubs them). Checks:

1. All 5 first-pass prompts contain the full 10-class rubric (labels, percentages, `- N = ` indices).
2. All 5 prompts ask for `Confidence: <1-10>` output and include the class-matching reminder.
3. `extract_verbalized_confidence` regex still parses canonical and edge-case formats.
4. `_CONF_RUBRIC`, gen2, and two-pass critique all embed the same 10-class rubric.
5. `generate_with_logits` returns a 5-tuple via `_detect_truncation`.
6. `get_forced_answer` builds correct dataset-specific prompts and `extract_model_answer` recovers the forced answer (mocks `generate_simple_response`).
7. `evaluate_sample` result dict exposes `main_pass_finish_reason`, `main_pass_was_truncated`, `was_forced`, `forced_answer_response`, `two_pass_finish_reason`, `two_pass_was_truncated`.

Run: `python DavidsDatasets/verify_rubric.py` → `ALL CHECKS PASSED`.

## 5. Files modified this session

| File | Changes |
|------|---------|
| `confidence.py` | `_CONF_RUBRIC` rewritten in bulleted Option-3 format; gen2 + two-pass critique rubrics aligned to same format; `generate_with_logits` returns 5-tuple via `_detect_truncation`; new `get_forced_answer` function. |
| `evaluation.py` | Imports `get_forced_answer`; both branches detect truncation and call forced-answer fallback; new columns in result dict; merge conflicts resolved keeping `main_meta` naming + `reasoning_for_critique` extraction + `_empty_two_pass` template. |
| `data_utils.py` | Merge conflict resolved (medqa for-loop refactor); indentation error fixed. |
| `verify_rubric.py` | New file — offline verification suite, no GPU/model needed. |

## 6. What should be re-run

| Dataset | Reason | Re-run? |
|---|---|---|
| **MMLU Pro** | First-pass prompt added rubric (was missing entirely) | **Yes — biggest delta** |
| **StrategyQA, TriviaQA** | First-pass rubric scale changed (0.0–1.0 ranges → 1-10 indices); two-pass rubric reformatted | **Yes** |
| **GSM8K, MedQA** | First-pass rubric format changed (numbered → bulleted); two-pass rubric reformatted; forced-answer + truncation tracking now active | **Yes** |

For Qwen3 specifically, expect a non-trivial number of `was_forced=True` rows on MMLU Pro hard-science questions. Check `main_pass_was_truncated` rate per dataset; if it's >20% on any dataset, consider raising `MAX_NEW_TOKENS` further for that dataset.

## 7. Open housekeeping items (not done this session)

- **No `.gitignore`** in the repo. `__pycache__/` and `.ipynb_checkpoints/` are being tracked, which is why `git status` shows `.pyc` files as modified after every run. Suggested `.gitignore`:
  ```
  __pycache__/
  *.pyc
  .ipynb_checkpoints/
  ```
- The `.ipynb_checkpoints/` folder contains stale snapshots of `confidence.py`, `data_utils.py`, `evaluation.py` with old syntax errors. These are harmless (Jupyter regenerates them) but show up in compile sweeps.

## 8. Known limitations carried forward

All limitations from the prior session still apply (internal-state abstraction in Gen 2, 3× inference cost for qwen3, SE settings need restoration before analysis). New items:

- **Forced-answer prompt design is heuristic.** The forced call shows up to 3,000 chars of truncated reasoning and asks for an answer in 8 tokens. If the truncated reasoning is so incomplete the model has no basis to commit, the forced answer is essentially a guess — but the verbalized confidence elicited *after* the forced answer should reflect that.
- **Mocked vs. real verification.** `verify_rubric.py` confirms code shape and prompt content but does not run the model. Sanity-check `was_forced=True` rows in the first real run output.

---

# Session 2026-05-12 — Add Gemma 4 31B IT (`gemma4` family)

Added a new model family `gemma4` pointing at `google/gemma-4-31b-it`. The model is configured but not yet evaluated. Treated as a reasoning model with `<think>` blocks (like `qwen3`).

## 1. config.py changes

| Setting | Change |
|---|---|
| `MODEL_FAMILY` docstring | Now lists `"qwen", "qwen3", "llama", "gemma", or "gemma4"` |
| `MODEL_NAMES["gemma4"]` | New entry, instruct-only: `"instruct": "google/gemma-4-31b-it"` (mirrors `qwen3` shape — no `base` variant) |
| `_MAX_NEW_TOKENS_BY_FAMILY["gemma4"]` | 8,192 (qwen3-equivalent — thinking-model budget) |
| `_SE_MAX_NEW_TOKENS_BY_FAMILY["gemma4"]` | 4,096 |
| `_TWO_PASS_MAX_NEW_TOKENS_BY_FAMILY["gemma4"]` | 4,096 |
| `TWO_PASS_DISABLE_THINKING` | Now `MODEL_FAMILY in ("qwen3", "gemma4")` (was `== "qwen3"`) |
| `get_model_label` labels dict | Added `"gemma4": "Gemma4-31B"` |

## 2. evaluation.py changes

Two `MODEL_FAMILY == "qwen3"` gates widened to `MODEL_FAMILY in ("qwen3", "gemma4")`:
- Line 86: three-generation flow entry (Gen 1 reasoning + Gen 2 own-work confidence + Gen 3 blinded critique)
- Line 314: `<think>...</think>` stripping on SE-sampled answers before extraction

The qwen3 path was reused wholesale on the assumption that Gemma 4 31B IT also emits `<think>...</think>` tags. **If Gemma 4 IT uses a different reasoning-block convention (or no tag at all), the regex `_QWEN3_THINK_RE` will not match and `reasoning_for_critique` will fall back to the stripped response.** Verify on the first real run by inspecting `full_response` for `<think>` markers.

## 3. model_utils.py — intentionally NOT changed

The user opted to keep gemma4 on the small-model loading path:
- `large_model_families = {"qwen3"}` — gemma4 **not** added → single-GPU loading (no `device_map="auto"`)
- `dtype = torch.bfloat16 if MODEL_FAMILY == "qwen3" else torch.float16` — gemma4 stays at fp16

**Memory implication:** 31B params × 2 bytes (fp16) ≈ 62 GB VRAM just for weights, before KV cache. This will OOM on anything smaller than an H100/H200 or A100-80GB. If loading fails, the first thing to flip is `large_model_families = {"qwen3", "gemma4"}` (multi-GPU shard) and the dtype check to `MODEL_FAMILY in ("qwen3", "gemma4")` (bfloat16 — Gemma family historically required bf16 for numerical stability).

## 4. What was *not* updated and may need attention

- `verify_rubric.py` — does not currently exercise the gemma4 path; if you want gemma4 covered, the family-gate widening in evaluation.py is the only logic change to assert.
- Confidence-rubric prompts (`_CONF_RUBRIC`, gen2, two-pass critique) — unchanged. Gemma 4 inherits the same 10-class bulleted rubric standardized in the 2026-05-10 session.
- Forced-answer fallback (`get_forced_answer`) — runs for gemma4 automatically via `main_meta["was_truncated"]` regardless of family.

## 5. To run gemma4

```python
# config.py
MODEL_FAMILY = "gemma4"
MODEL_VARIANT = "instruct"
```

Then run `main.py` as usual. The output CSV label will be `Gemma4-31B-instruct`. Expect:
- 3× inference cost per question (three-generation flow, same as qwen3)
- High VRAM usage; monitor for OOM
- Non-zero `was_forced=True` rate if `<think>` blocks blow past 8,192 tokens (same failure mode as qwen3 on hard math)

## 6. Files modified this session

| File | Changes |
|------|---------|
| `config.py` | New `gemma4` entry in `MODEL_NAMES`; three token-budget dicts extended; `TWO_PASS_DISABLE_THINKING` widened; label added |
| `evaluation.py` | Two `MODEL_FAMILY == "qwen3"` gates widened to include `"gemma4"` |

---

# Session 2026-05-13 — Add `gptoss`, fix `was_forced` inversion, split `gemma4` instruct vs. base

Three changes this session:

1. Added a new reasoning-model family `gptoss` pointing at `openai/gpt-oss-20b`.
2. Fixed a real bug in `was_forced` semantics — the flag was essentially inverted from intent.
3. Added the `gemma4` base variant (`google/gemma-4-31b`) and introduced `USE_REASONING_FLOW` so gemma4 *instruct* keeps the three-gen reasoning flow while gemma4 *base* falls through to the standard single-pass flow.

## 1. New family: `gptoss` (`openai/gpt-oss-20b`)

Verified `google/gemma-4-31B` (base) and `google/gemma-4-31B-it` exist on HuggingFace via WebFetch before wiring base in (the prior session's handoff noted "no base variant" — that was stale, the base model is published).

Wired `gptoss` like a large reasoning model (parallel to `qwen3`/`gemma4`):

| File | Change |
|---|---|
| `config.py` | `MODEL_NAMES["gptoss"] = {"instruct": "openai/gpt-oss-20b"}`; budgets 8192 / 4096 / 4096 (matches qwen3/gemma4 reasoning budgets); added `"gptoss": "GPT-OSS-20B"` to label dict. |
| `model_utils.py` | `large_model_families = {"qwen3", "gemma4", "gptoss"}` → uses `device_map="auto"`. Added `gptoss` to the `bfloat16` dtype branch (gpt-oss-20b is bfloat16 native). |
| `evaluation.py` | Both reasoning-flow gates now include `gptoss` (later switched to `USE_REASONING_FLOW`, see §3). |

**Caveat re harmony / channel format:** gpt-oss models use OpenAI's harmony format (analysis/commentary/final channels). If the HF chat template doesn't wrap the analysis channel in a `<think>...</think>`-shaped block, the `_QWEN3_THINK_RE` regex won't catch it, and `reasoning_for_critique` will fall back to the stripped response. Worth eyeballing `full_response` on the first run.

## 2. `was_forced` was inverted — root cause + fix

**User-reported symptom:** in `mmlupro_confidencewithnewSE_Gemma4-31B-instruct.csv`, rows whose `full_response` ended with a clean `Answer: X` line were marked `was_forced=True`, and rows where the response was a chaotic loop with no `Answer:` line were marked `was_forced=False`. Essentially inverted from what the column name suggests.

**Root cause:** the trigger for the forced-answer call was `main_meta["was_truncated"]`, which only checks whether the last generated token equals EOS. On Gemma4 with an 8192 budget, most responses don't emit EOS in time — even when they contain a clean `Answer: I` line near the end. So:
- Clean responses → flag fired → forced call ran → forced call usually returned the same letter → `model_answer` got overwritten with the same value → `was_forced=True` (looked wrong to user).
- Chaotic responses → flag fired → forced call ran with its own ~8-token budget → forced call ALSO got truncated mid-thought → `forced_answer=None` → kept Priority-3 garbage → `was_forced` stayed `False` (looked wrong to user).

**Fix** (in both reasoning-flow and standard branches of `evaluation.py::evaluate_sample`):
- Use `extract_model_answer_strict` (already available at [data_utils.py:281](DavidsDatasets/data_utils.py#L281)) on the main response — Priority-1 only, requires an explicit `Answer:` line.
- If strict extraction succeeds → trust it, `was_forced=False`, skip the forced call entirely (saves a generation when the response was already clean).
- If strict extraction fails → set `was_forced=True` *first* (the row is unreliable regardless of what happens next), then attempt the forced call; if forced succeeds use its answer, else fall back to the lax `extract_model_answer` but keep `was_forced=True`.

**New semantics:** `was_forced=True` ⇔ the main response did not produce a clean `Answer:` line. Matches user intuition.

**Verification:** simulated the three example rows from the disputed CSV:

| idx | response shape | old was_forced | new was_forced |
|---|---|---|---|
| 8592 | ends `Answer: I` | True | **False** ✓ |
| 628 | infinite-loop ramble, no Answer line | False | **True** ✓ |
| 9205 | rambles, no Answer line | False | **True** ✓ |

**Existing CSVs:** the `was_forced` column on every CSV written before this session is unreliable — re-run rather than trust it. `forced_answer_response` text remains useful (non-empty means a forced call was attempted).

## 3. `gemma4` base variant + `USE_REASONING_FLOW` flag

The prior session's note "no base variant" was wrong — `google/gemma-4-31b` exists. Added it with a routing decision: gemma4-instruct keeps the three-gen reasoning flow, gemma4-base does NOT (no chat template, no `<think>` scaffolding — pushing base through the reasoning flow would produce garbage).

`config.py` changes:
- `MODEL_NAMES["gemma4"]` gains `"base": "google/gemma-4-31b"`.
- New derived flag:
  ```python
  USE_REASONING_FLOW = (
      MODEL_FAMILY in ("qwen3", "gptoss")
      or (MODEL_FAMILY == "gemma4" and MODEL_VARIANT == "instruct")
  )
  ```
- `TWO_PASS_DISABLE_THINKING = USE_REASONING_FLOW` (was a family-set check).
- Budget override for `gemma4` base: when `not USE_REASONING_FLOW`, drops `MAX_NEW_TOKENS` 8192→1024, `SE_MAX_NEW_TOKENS` 4096→256, `TWO_PASS_MAX_NEW_TOKENS` 4096→1024. Base completion doesn't emit `<think>` blocks, so the reasoning-model budgets just waste compute.

`evaluation.py` changes:
- Imports `USE_REASONING_FLOW`.
- Both `MODEL_FAMILY in ("qwen3", "gemma4", "gptoss"):` branch gates replaced with `if USE_REASONING_FLOW:`.

`model_utils.py` — **unchanged for gemma4 base**: stays in `large_model_families` (both gemma4 variants are 31B and need `device_map="auto"`). Dtype handling for gemma4 unchanged (float16, same as prior session — note: Gemma family historically prefers bfloat16, flip to `MODEL_FAMILY in ("qwen3", "gptoss", "gemma4")` if you see numerical-stability issues).

Routing summary now:

| Family | Variant | Reasoning flow? | Notes |
|---|---|---|---|
| qwen3 | instruct | Yes | unchanged |
| gptoss | instruct | Yes | new this session |
| gemma4 | instruct | Yes | unchanged from 2026-05-12 |
| gemma4 | base | **No** | new this session — standard single-pass flow |
| qwen, llama, gemma | either | No | unchanged |

## 4. Stale handoff note corrected

The 2026-05-12 session said `model_utils.py` was intentionally NOT changed for gemma4. By the time this session started, the working tree had already added `gemma4` to `large_model_families` and the bfloat16 dtype branch — that pending diff was either reverted or never committed. After this session, the final state is:

```python
large_model_families = {"qwen3", "gemma4", "gptoss"}
dtype = torch.bfloat16 if MODEL_FAMILY in ("qwen3", "gptoss") else torch.float16
```

i.e. gemma4 (both variants) uses auto device_map but float16 dtype. If gemma4 OOMs or shows NaN losses, the dtype line is the lever — extend to `("qwen3", "gptoss", "gemma4")`.

## 5. Files modified this session

| File | Changes |
|------|---------|
| `config.py` | `gptoss` entry in MODEL_NAMES + budgets + label; `gemma4` base entry in MODEL_NAMES; new `USE_REASONING_FLOW` derived flag; variant-aware budget override for gemma4 base; `TWO_PASS_DISABLE_THINKING` keyed to the flag. |
| `model_utils.py` | `gptoss` added to `large_model_families` and to the bfloat16 dtype branch. |
| `evaluation.py` | Both reasoning-flow gates switched from `MODEL_FAMILY in (...)` to `USE_REASONING_FLOW`; both branches now gate the forced-answer call on `extract_model_answer_strict` failure (not `main_meta["was_truncated"]`); `was_forced=True` now means "main response had no clean Answer line", and the forced call is skipped entirely when strict extraction succeeds. |

## 6. Files NOT modified this session

| File | Status |
|------|--------|
| `data_utils.py` | Unchanged — pre-existing `extract_model_answer_strict` is the helper the fix relies on. |
| `confidence.py` | Unchanged — `get_forced_answer` interface unchanged. |
| `verify_rubric.py` | Unchanged — its `check_evaluate_sample_columns` assertion (both branches call `get_forced_answer` at least once each) still holds. |
| `save_utils.py`, `main.py`, `semantic_entropy.py`, `visualization.py` | Unchanged. |

## 7. Open items / things to watch on next run

- **Re-run any dataset where `was_forced` matters for analysis.** Old CSVs have the inverted flag. Logit/SE metrics, model_answer values, and is_correct are unaffected by this fix; only `was_forced` (and now skipping unneeded forced calls) changes.
- **Inspect `full_response` from a `gptoss` run** for whether the harmony analysis channel comes through as `<think>...</think>` or some other delimiter. If different, `_QWEN3_THINK_RE` will need extending (or generalize to a per-family regex).
- **Gemma4 base correctness gate is now `extract_model_answer_strict`.** Base models rarely write `Answer: X` unprompted, so expect a high `was_forced` rate on the first gemma4-base run. That's not a bug — it's the flag doing its new job.
- **Memory: gptoss-20B at bfloat16 + 256k-ish context.** 20B × 2 bytes ≈ 40 GB weights before KV cache. Single H100/H200 should be fine; multi-shard kicks in via `device_map="auto"` if needed.

---

# Session 2026-06-02 — Four compounding extraction & prompt bugs surfaced from Gemma4-base GSM8K and GPT-OSS CSVs

This session was triggered by the user sharing three CSVs and asking why specific rows had wrong or missing `model_answer` values: `95seed100gsm8k_confidencewithnewSE_Gemma4-31B-base.csv`, `50seed20legalbench_confidence_GPT-OSS-20B-instruct.csv`, and `40seed99triviaqa_confidence_GPT-OSS-20B-instruct.csv`. Investigation surfaced four distinct bugs that had compounded across prior sessions. They are independent — each one masked or was masked by the others — which is why the chain of effects had been hard to see from any single CSV.

## 1. Bug 1 — `extract_model_answer_strict` matched the FIRST `Answer:` line instead of the last

`extract_model_answer_strict` in `data_utils.py` used `re.search` with a case-insensitive `[Aa]nswer\s*:` pattern. `re.search` returns the first match. Gemma4 base routinely writes mid-reasoning phrases like "Now let me assess my confidence in this answer:" followed by a bullet list. The case-insensitive `[Aa]nswer` matched `answer:` inside that phrase, the capture grabbed the next non-empty line (e.g., `- I carefully calculated... reach the $90 threshold.`), the number regex pulled `$90` out, and the extractor returned `"90"`. The actual `Answer: 12` line at the end of the response was ignored.

Concretely visible in the Gemma4-base GSM8K CSV at row 12 (Carlos lemon tree) where `model_answer=90` instead of `12`. Same root cause hit rows 1173, 1247, 101, 993, 671, and 863 — where strict returned None, the forced-answer fallback ran on a base model, and the lax extractor's Priority-3 fallback grabbed digits from template placeholders.

## 2. Bug 2 — Lax extractor's Priority-3 fallback grabbed digits from template placeholders

`extract_model_answer` had a Priority-3 fallback that returned the last digit-only token anywhere in the response. When the forced-answer call ran on a base model and the base model echoed back template fragments like `Confidence: <0-10>\nCorrect: <Yes/No`, Priority-3 grabbed `"10"` from `<0-10>` and stored that as `model_answer`. The row then had `model_answer="10"`, `is_correct=False`, `answer_extraction_failed=False` — looks like the model answered wrong, but the model never actually committed to anything. Row 863 (Gretchen coins, ground_truth=110) is the canonical example.

## 3. Bug 3 — Prompt template put `Solution:` *after* `Answer:` in the format example

`create_prompt` in `confidence.py` built every dataset's prompt ending with both an `Answer: <X>` format example AND a trailing `Solution:\nLet me think through this step by step.` line. The trailing line was originally intended as a continuation primer for *base* models (which don't follow instructions and need to be primed mid-sentence to keep generating). But the same string was being sent to instruct/reasoning models inside the chat template, where it appeared *after* a phrase like "you MUST end with EXACTLY this format: Answer: X". The model received two contradictory end-of-response instructions and tried to reconcile them.

For Qwen3 and Gemma4-instruct, the resulting confusion happened inside `<think>...</think>` and got stripped before extraction — mostly looked like wasted tokens. For GPT-OSS-20B, which uses OpenAI's harmony channel format (analysis + final, no `<think>` tags), the confusion poured into the analysis channel directly. Row 261 of the LegalBench CSV (and many others) shows the model writing the same paragraph dozens of times trying to figure out whether to put `Answer:` first or `Solution:` first. Token budgets get eaten, the model frequently never reaches a clean commit, `main_pass_finish_reason="length"` rows are common, and the `seq_confidence_mean` distribution gets dragged to extreme negatives (–400 to –1400 for GPT-OSS vs. –30 to –90 for Gemma4 base on the same dataset).

## 4. Bug 4 — GPT-OSS harmony envelope wasn't stripped before extraction

GPT-OSS-20B outputs its response in OpenAI's harmony channel format: `analysis<reasoning>assistantfinal<final answer>`. The literal token `assistantfinal` delimits the channels, and crucially there is no newline between `assistantfinal` and the start of the final answer. So a typical GPT-OSS response ends with `assistantfinalAnswer: money`.

The extractors had no concept of this delimiter. Two failure modes:
- **Strict (anchored regex, after the Bug 1 fix)**: `(?m)^[^a-zA-Z\n]*[Aa]nswer:` would not match `assistantfinalAnswer:` because `assistantfinal` contains letters before `Answer`. Strict returned None.
- **Lax (unanchored regex, pre-Bug 1 fix)**: matched the first `answer:` in the response, which was often inside mid-CoT phrases like "We need to answer:" in the analysis channel. Non-greedy capture sometimes ran to end-of-response, returning the entire blob as `model_answer`.

For TriviaQA specifically, `check_triviaqa_correct` does substring matching against ground-truth aliases. Returning the entire reasoning blob accidentally produced `is_correct=True` whenever the right answer phrase happened to appear anywhere in the looping text — so this bug was largely invisible from accuracy numbers alone. Row 5588 (Thursday Next) showed `is_correct=True` because `model_answer` was thousands of characters containing "Jasper Fforde" somewhere; same masking pattern for rows 9254 (caballo→horse), 12734 (Pennsylvania→Harrisburg), etc.

## 5. Fixes applied

### `data_utils.py`

- `extract_model_answer_strict` and `extract_model_answer` Priority-1 regex rewritten to use `re.findall` with start-of-line anchor `(?m)^[^a-zA-Z\n]*[Aa]nswer[^a-zA-Z\n]*:` and take the LAST match. The `[^a-zA-Z\n]*` prefix accepts markdown decorations (`**`, `###`, `-`, numbered lists, indentation) but rejects letters, so phrases like `in this answer:`, `my answer:`, `the answer:` no longer match.
- Dropped the lax Priority-3 last-number-in-response fallback for GSM8K. Rows where the model never produced a clean Answer line now correctly receive `model_answer=None` and `answer_extraction_failed=True` instead of a confident-looking fabricated number.
- New `_strip_harmony_envelope(response)` helper called at the top of both extractors. If `assistantfinal` is in the response, returns everything after the last occurrence; otherwise pass-through. No-op for any model that doesn't use harmony format.

### `confidence.py`

- `create_prompt` rewritten to build `instruction_body` + `base_primer` separately. Instruct models receive only `instruction_body` (clean prompt ending with "End your response with a single line: Answer: X" or the three-line confidence variant). Base models receive `instruction_body + base_primer`, where `base_primer` is `\n\nSolution:\nLet me {primer_verb} this step by step.\n\n`. Per-dataset `primer_verb` ("work through", "think through", "analyze each option") preserved from the old prompts.
- `get_forced_answer` body no longer contains the template line `Answer: <number>`. Instead it passes `base_suffix="\n\nAnswer: "` so base models complete the answer slot via next-token continuation rather than echoing prompt fragments. For instruct models the chat template ignores `base_suffix`, so no change there. Includes a retry that prepends `Answer: ` if direct extraction fails on a bare completion.

### `evaluation.py`

- Module-level constant `_HARMONY_FINAL_DELIM = "assistantfinal"` added next to the existing `_QWEN3_THINK_RE`.
- Reasoning-flow path now strips the harmony envelope from `response` (everything before the last `assistantfinal`) and pulls the *pre*-`assistantfinal` content as `reasoning_for_critique` (with the leading `analysis` channel marker trimmed). Falls back to `<think>...</think>` extraction for Qwen3, and to the stripped response if neither envelope is present.
- SE sampling path applies the same harmony strip to each of the 5 sampled answers before extraction.

### `verify_rubric.py`

- Forced-answer assertion text updated to match the new prompt body (`Output only` instead of `Output ONLY`) and to assert that the call passes `base_suffix == "\n\nAnswer: "`. Otherwise unchanged. Still passes end-to-end with `ALL CHECKS PASSED`.

## 6. Implications for existing CSVs by model

| Model | Re-run? | Why |
|---|---|---|
| **Gemma4 base** | Optional, but worth it on GSM8K | Bug 1 affected ~6 visible GSM8K rows that wrote "in this answer:" mid-reasoning. Full_response and logit metrics are unaffected. Can re-parse the saved `full_response` offline rather than re-run inference for an exact diff. |
| **Qwen3 instruct** | Probably no | Qwen3's confusion (Bug 3) happened inside `<think>...</think>` and got stripped before extraction. Final `Answer:` line was usually clean. The only loss is some token-budget waste on hard rows that may have caused unnecessary truncations. |
| **Gemma4 instruct** | Check first | Depends on whether the model actually emits `<think>` blocks (prior handoff assumed yes, never verified). Run a one-liner against your CSV: if `<think>` appears in 80%+ of `full_response` rows, treat like Qwen3. If `assistantfinal` appears instead or neither marker appears, treat like GPT-OSS. |
| **GPT-OSS instruct** | Yes, fully | All four bugs hit GPT-OSS together. EVERY column in the CSV is meaningfully affected — see §7. |
| **Older base/instruct (Gemma2, Llama, Qwen2.5)** | No | Different output format (`The answer is: X` rather than `Answer:` lines), routes through Priority-2 patterns which I didn't touch. |

## 7. What changes downstream when GPT-OSS is re-run

Beyond `model_answer` itself, re-running GPT-OSS will substantially shift these other columns:

- **Logit confidence** (`seq_confidence_mean`, `logit_confidence_min/geom/mean_prob`): `seq_confidence_mean` was at −400 to −1400 because the model was generating thousands of low-probability repetition tokens trying to resolve the prompt contradiction. After the fix, GPT-OSS finishes in normal token counts and these metrics move into a range comparable to Gemma4 (−30 to −90). The "GPT-OSS has terrible logit calibration" pattern in the old data is an artifact.
- **Verbalized confidence + two-pass critique** (`verbalized_confidence`, `single_pass_confidence`, `more_likely_than_not`, `two_pass_critique`): the critic was being fed the looping confusion as the "reasoning." It either confabulated a score or judged the confusion harshly. With harmony stripping, the critic now reads the actual analysis channel.
- **Truncation / forcing** (`main_pass_finish_reason`, `main_pass_was_truncated`, `was_forced`): many rows that were `length` (truncated) and `was_forced=True` will flip to `eos` and `was_forced=False` after the fix because the loop isn't eating the budget.
- **MCQ answer-token entropy** (`answer_token_entropy`, `answer_letter_probs`, `chosen_answer_raw_prob` for MMLU-Pro / MedQA): the answer token in a clean post-`assistantfinal` context has much sharper letter probability concentration than one buried in confusion. Distributions will look qualitatively different.
- **Semantic entropy** (if `SKIP_NLI_CLUSTERING` is re-enabled): SE was inflated because each of the 5 sampled answers was a different chunk of confused harmony output that the strict extractor couldn't normalize.
- **Calibration / AUROC analysis**: relative comparisons drawn from this data ("Gemma4 vs GPT-OSS" on calibration) shouldn't be trusted on the GPT-OSS side.

## 8. Re-extraction vs. re-inference

Bugs 1, 2, and 4 are *extraction-only* bugs. The full_response column is unchanged; only the parsing of it changes. For any existing CSV that you want to update without re-running inference, the saved `full_response` column can be re-parsed with the new extractors and the result diffed against the stored `model_answer`. This works for Gemma4 base and any other run that wasn't affected by Bug 3.

Bug 3 is a *generation-level* bug. The model's actual output is different under the new prompt. Re-extraction can't recover the unconfused response — only re-running inference can. This is why GPT-OSS requires a full re-run.

## 9. Files modified this session

| File | Changes |
|------|---------|
| `data_utils.py` | Anchored Priority-1 regex (start-of-line, non-letter prefix) + `findall[-1]` in both extractors; dropped GSM8K Priority-3 last-number fallback; new `_strip_harmony_envelope` helper called at top of both extractors. |
| `confidence.py` | `create_prompt` restructured to separate `instruction_body` from `base_primer`; `get_forced_answer` prompts no longer contain template `Answer:` line, pass `base_suffix="\n\nAnswer: "` instead. |
| `evaluation.py` | `_HARMONY_FINAL_DELIM` constant added; reasoning-flow path strips harmony from `response` and uses pre-`assistantfinal` content for `reasoning_for_critique`; SE sampling path strips harmony per sample. |
| `verify_rubric.py` | Updated forced-answer assertions to match new prompt body and `base_suffix` value. |

## 10. Open items / things to watch on next run

- **Verify the Gemma4-instruct envelope assumption.** Run the diagnostic (`<think>` vs `assistantfinal` vs neither in `full_response`) against an existing Gemma4-instruct CSV. The prior handoff assumed `<think>` but never confirmed. If the assumption is wrong, those CSVs need re-running too.
- **Inspect GPT-OSS `full_response` after re-run** to confirm `assistantfinal` is appearing in the expected place and the stripping is doing what it should. The harmony format may have edge cases (multiple `assistantfinal` tokens, missing delimiter on truncation, etc.) — `_strip_harmony_envelope` uses `rsplit(..., 1)` so it takes the LAST occurrence, but if `assistantfinal` doesn't appear at all in a truncated response, the strip is a no-op and extraction falls through to the anchored regex.
- **Re-extraction script for Gemma4 base CSVs**: if you want exact deltas without re-running inference, a small offline script can re-parse `full_response` with the new extractors and diff against the stored `model_answer`. Ask and I'll write it.
- **Forced-answer prompt design is heuristic.** Bug 2 fix removes the worst Priority-3 victims (`<0-10>` → `"10"`), but base models still can produce garbage in the forced-answer slot when their reasoning was too incomplete to support a commit. The `was_forced=True` flag remains the correct way to identify those rows.

## 11. Smoke tests performed this session

- 17-case extractor smoke test covering: row 12 (Carlos lemon tree), row 1173 (dishwasher), plain `Answer:`, markdown bold `**Answer:**`, header `### Answer:`, bullet `- Answer:`, numbered `1. Answer:`, multi-`Answer:` (last wins), Gemma2-style `The answer is: 47.25`, MMLU-Pro plain + bold, `in this answer:` negative, `my answer:` negative, forced-call garbage (`<0-10>`), strict-fallback numeric-only line, triviaqa with rubric, strategyqa plain. All pass.
- 8-case GPT-OSS harmony smoke test covering rows 5539, 5588, 9254, 12734 from the TriviaQA CSV plus harmony-format LegalBench and GSM8K cases plus two non-harmony sanity checks. All pass — harmony extraction now returns the actual answer instead of the whole reasoning blob.
- 6-dataset × 2-variant × 2-confidence `create_prompt` smoke test confirming instruct models get clean prompts (no `Solution:` primer) and base models get the body plus the trailing continuation primer. All pass.

## 12. Follow-up patch — TriviaQA loader hid real errors behind a broken fallback

On the lambda H100 box, `DATASET = "triviaqa"` started failing with:

```
huggingface_hub.errors.HfUriError: Invalid HF URI
'hf://datasets/trivia_qa@<sha>/.huggingface.yaml'.
Repository id must be 'namespace/name', got 'trivia_qa'.
```

### What was actually happening

`load_triviaqa()` in `data_utils.py` had two attempts:

1. Primary: `load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split="validation")`
2. Fallback (bare name): `load_dataset("trivia_qa", "rc.nocontext", split="validation")`

The primary failed silently (caught by a bare `except Exception:`), and execution fell through to the fallback. Newer `huggingface_hub` versions reject the bare repo id `trivia_qa` outright at URI-parse time, producing the `HfUriError` above. So every TriviaQA load failure on a current HF stack surfaced as the same misleading URI error regardless of what the primary loader actually complained about.

### Fix

- Removed the bare-name fallback entirely — it cannot succeed on current `huggingface_hub` and only served to mask the real error.
- Added `trust_remote_code=True` to the primary call (some `datasets` builds need it for `mandarjoshi/trivia_qa`'s loader script).
- Wrapped that with a `TypeError` guard so older `datasets` versions that don't accept the kwarg still work.

The function now lets any genuine loader error from `mandarjoshi/trivia_qa` propagate cleanly so it can be diagnosed.

### Implications for existing TriviaQA CSVs

**None — same dataset, same rows.** The primary call (`mandarjoshi/trivia_qa`, config `rc.nocontext`, split `validation`) is byte-identical to what previous runs used. The only changes are:

- `trust_remote_code=True` is a permission flag for HF to execute the dataset's loader script; it doesn't change which dataset, which split, or which rows are returned.
- Removing the dead fallback path has no effect on any run that actually produced data, since that path was never reached on a successful load (and now crashes on any current HF stack rather than silently corrupting state).

Row order, indices used by seeds, ground-truth aliases, and `check_triviaqa_correct` behavior are unchanged. Previous TriviaQA CSVs remain directly comparable to anything run after this patch.

### Files modified

| File | Change |
| --- | --- |
| `data_utils.py` (`load_triviaqa`) | Dropped bare-name fallback; added `trust_remote_code=True` to primary call with `TypeError` guard for older `datasets` versions. |

### If TriviaQA still fails after this patch

The error you see now is the real one. Most likely fixes, in order:

1. `pip install -U datasets huggingface_hub` on the affected machine (lambda stack is older than what `mandarjoshi/trivia_qa` currently expects).
2. `huggingface-cli login` if the error mentions auth (rare for this public dataset, but possible behind certain proxies).
3. Network reachability to `huggingface.co` — check from the lambda box directly.
- `verify_rubric.py` end-to-end: `ALL CHECKS PASSED`.

---

# Session 2026-06-07 — GPT-OSS repetition-loop diagnosis + decoding-guard fix

## 0. Current status (TL;DR)

GPT-OSS-20B-instruct runs were collapsing into **verbatim repetition loops** —
the same failure class the §18 (confidence_telemetry_handoff.md) base-model work
fixed for Llama base, but on an *instruct* model. The §18.3 repetition guards
were gated on `MODEL_VARIANT == "base"`, so GPT-OSS (config'd `instruct`) got no
guard. **Fixed** by extending the guard predicate in `confidence.py`, scoped so
GPT-OSS gets the inert `no_repeat_ngram_size=3` only (no distribution-warping
`repetition_penalty`), base stays exactly as §18.3 had it, and Qwen/Gemma
instruct stay byte-for-byte reproducible. Code is in the working tree, **not
committed**. No re-run done yet (needs GPU).

## 1. The problem (diagnosed from the GPT-OSS CSVs)

GPT-OSS spins inside its Harmony **`analysis`** (reasoning) channel — a single
line repeated dozens-to-~1000× ("Ok, I'm going insane" ×1072; "She is banned
from the United Kingdom" ×981) — exhausts `max_new_tokens` (8192,
`finish_reason="length"`), and never reaches the `final` channel. So
`model_answer` leaks the truncated analysis text (rows literally start with
`"analysis…"`) and the row is marked wrong.

Loop census across the four consolidated 150-item GPT-OSS files:

| Benchmark | n | acc | loop rows (`finish_reason=="length"`) | % |
|---|---|---|---|---|
| GSM8K | 150 | 94.0% | **0** | 0% |
| LegalBench | 150 | 81.3% | 2 | 1.3% |
| StrategyQA | 150 | 78.0% | 5 | 3.3% |
| TriviaQA | 149 | 68.5% | **24** | 16.1% |

GSM8K is clean (deterministic numeric target → no recall flailing); TriviaQA is
the hotspot. **Not parser-recoverable**: ground truth appears anywhere in the
looped text in only 3/24 TriviaQA rows, and never on an `Answer:` line →
consistent with handoff §18.7 (needs a decoding fix + re-run, not a re-parse).

## 2. Root cause

`confidence.py::generate_with_logits` resolved guards from `MODEL_VARIANT`
only: `repetition_penalty = 1.2 if MODEL_VARIANT == "base" else 1.0`, same for
`no_repeat_ngram_size`. GPT-OSS has **no `base` variant** (config.py:46-47,
`gptoss` is instruct-only) and `MODEL_VARIANT = "instruct"`, so every guard was
off and the loops ran free to the 8192-token budget.

## 3. The fix (DONE — `confidence.py`, working tree, not committed)

Split the single base-only predicate into two, by guard mechanism:

```python
# Anti-loop ngram ban: base (any family) + GPT-OSS. Inert on clean rows.
_needs_ngram_guard = (MODEL_VARIANT == "base") or (MODEL_FAMILY == "gptoss")
# repetition_penalty warps every step → base only, where it's load-bearing.
_needs_rep_penalty = (MODEL_VARIANT == "base")
```

Resolved guard matrix (verified):

| Config | `no_repeat_ngram_size` | `repetition_penalty` | `stop_strings` |
|---|---|---|---|
| Qwen/Gemma **instruct** | 0 | 1.0 | — |
| **GPT-OSS** instruct | **3** | **1.0** | — |
| **base** (Llama/Gemma) | 3 | 1.2 | ✓ |

Also added `MODEL_FAMILY` to the `from config import (...)` block in
`confidence.py`. Docstring rewritten to state the guiding principle.

**The guard policy / rationale (the reasoning behind the numbers):**
- *Use the minimal constraint that prevents non-termination; prefer constraints
  that are INERT on well-behaved generations over `repetition_penalty`, which
  reshapes the distribution at every step and so perturbs the very token
  probabilities the confidence study measures.*
- `no_repeat_ngram_size=3` only fires when a 3-gram would repeat → on a
  non-looping greedy row the output and its scores are **bit-identical** to
  unconstrained decoding. GPT-OSS's loops *are* repeated 3-grams, so this alone
  kills them → GPT-OSS can skip `repetition_penalty` entirely. **3** = tightest,
  already the §18.3 base value (shared mechanism, simple methods description);
  larger values are weaker.
- `repetition_penalty=1.2` is base-only because base degeneration is mixed
  (over-generation + looser rambles that dodge a 3-gram ban), where the penalty
  is load-bearing; `=1.0` everywhere else means "no penalty / don't touch the
  distribution."
- `stop_strings` stays base-only (targets base over-generation: restating
  "Question:"/"Solution:" blocks); they don't fit GPT-OSS's reasoning loops.

## 4. Key decisions (and why)

- **Why NOT apply the change to base too** (user asked): base data is still
  pre-fix/broken (Gemma2-base TriviaQA 15/40 empty answers; Llama-base GSM8K
  44/150 `length`-truncated), so it's getting a full re-run regardless — but its
  degeneration is *texturally different* (over-generation + loose rambles, **0**
  verbatim >50× loops detected vs. GPT-OSS's tight ones), so `repetition_penalty`
  is plausibly load-bearing for base. Keep base on the heavier §18.3 recipe;
  **validate empirically** (re-run ~10 base rows with/without the penalty) before
  the full base re-run.
- **Why NOT apply to all instruct models**: Qwen/Gemma instruct don't loop
  (their HTML pages exist, built from clean runs) and must stay byte-for-byte
  reproducible per handoff §18.5. Add a future model to the predicate only if it
  demonstrably loops.
- **Guard strength for GPT-OSS**: chose ngram-only (not 1.2+ngram3) specifically
  so clean rows are untouched → enables a **selective** re-run (only loop rows),
  which is better for the calibration paper: dropping loop rows would bias
  calibration optimistically (loop rows are the hardest items), and re-running
  recovers a real (answer, confidence, correctness) triple on the high-
  uncertainty tail.

## 5. Verification performed

- `python3 -c "import ast; ast.parse(open('confidence.py').read())"` → parses OK.
- `python3 verify_rubric.py` → **ALL CHECKS PASSED** (5-tuple +
  `_detect_truncation` contract from §18 intact).
- Guard matrix resolved programmatically across 5 (family, variant) combos —
  matches the table in §3.
- Loop census + recoverability computed from the consolidated GPT-OSS CSVs
  (pandas; installed locally this session via `pip3 install pandas`).

## 6. Files modified this session

| File | Change |
|---|---|
| `confidence.py` | Added `MODEL_FAMILY` import; split guard predicate into `_needs_ngram_guard` (base ∪ gptoss) and `_needs_rep_penalty` (base only); rewrote `generate_with_logits` docstring to state the guard policy. |

Nothing committed. (`__pycache__/*.pyc` will also show as modified per the
standing repo quirk — do **not** stage them; see confidence_telemetry_handoff.md
§19.1–19.2.)

## 7. Important file paths

```
Pipeline (source of truth, version-controlled):
  ~/Documents/GitHub/DCAK_RTA_v8-Transfer/DavidsDatasets/confidence.py   ← edited
  ~/Documents/GitHub/DCAK_RTA_v8-Transfer/DavidsDatasets/config.py        (gptoss = instruct-only, lines 46-47)

GPT-OSS data (consolidated 150-item files):
  ~/Desktop/AI Research/GPTOSS/TriviaQA_OSS/150TriviaQAGPT-OSS-20B-instruct.csv
  ~/Desktop/AI Research/GPTOSS/StrategyQA_OSS/150StrategyQAGPT-OSS-20B-instruct.csv
  ~/Desktop/AI Research/GPTOSS/Gsm8k_OSS/150_GSM8K_GPT-OSS-20B-instruct.csv
  ~/Desktop/AI Research/GPTOSS/LegalBenchOSS/150LegalBenchGPTOSS-20B-Instruct.csv
  ~/Desktop/AI Research/GPTOSS/<bench>_OSS/Seperated*/...   (per-seed shards)

Base data (PRE-FIX / broken — full re-run pending):
  ~/Desktop/AI Research/GEMMA/Gemma2Base/...
  ~/Desktop/150GSM8KLlama3.1-8B-base - 1.csv
  ~/Desktop/150strategyqaLlama3.1-8B-base - 15seed1strategyqa_confidence_Llama3.1-8B-base.csv

Handoff docs:
  DavidsDatasets/confidence_telemetry_handoff.md   (standalone reference; §18-19 = pipeline fixes)
  DavidsDatasets/SESSION_HANDOFF.md                 (this chronological log)
```

## 8. Commands run this session

```bash
# locate + read the newest handoff (confidence_telemetry_handoff.md, Jun 6)
find /Users/davidzhu -iname "*handoff*" -type f

# loop census / recoverability over GPT-OSS CSVs
pip3 install pandas        # was missing in the local python3
python3  # ad-hoc analysis: finish_reason counts, repeated-line detection,
         # ground-truth-in-text recoverability, base fix-state + loop severity

# fix verification
python3 -c "import ast; ast.parse(open('confidence.py').read())"
python3 verify_rubric.py   # ALL CHECKS PASSED
```

## 9. Pending tasks / next steps (ready to execute)

1. **Build the scan/merge helper** (not yet written). Three-step workflow:
   - *scan mode* (local, no GPU): read each GPT-OSS CSV, emit the `idx` list of
     loop rows (`main_pass_finish_reason == "length"`) per benchmark.
   - *(you, on GPU box)*: run `main.py` on **only those idxs** with the new
     config (GPT-OSS → ngram-only is now automatic) → small CSV of fresh rows.
   - *merge mode* (local): splice regenerated rows over the old ones by `idx`,
     stamp `regenerated_after_loop = True` (and `False` on untouched rows).
   - **Open check before building**: does `main.py` support running a specific
     idx list? If not, add a small idx-filter hook.
2. **Selective re-run** of GPT-OSS TriviaQA (24), StrategyQA (5), LegalBench (2).
   GSM8K (0 loops) needs no re-run. Because ngram=3 is inert on clean rows, the
   untouched rows stay valid → no full re-run needed.
3. **Validate the base recipe empirically** before the (separately-pending) full
   base re-run: re-run ~10 base loop/empty rows with vs. without
   `repetition_penalty` to confirm ngram-only really is insufficient for base. If
   base loops turn out to also be tight verbatim repeats, base could drop the
   penalty too.
4. **Commit** (per confidence_telemetry_handoff.md §19.1 — only when ready):
   `git add DavidsDatasets/confidence.py` then commit; do NOT stage `*.pyc`.
5. **Secondary symptom (lower priority)**: when truncation still occurs, the
   forcing fallback can leak `analysis…` text into `model_answer` (17 such rows
   pre-fix). The guard removes most truncations; harden the extractor separately
   only if leaks persist after re-run.

---

# Session 2026-06-07 (part 2) — Gemma2-instruct "Confidence: N" mis-parse + refusal detection + re-extraction

## 0. Current status (TL;DR)

Gemma2-9B-instruct TriviaQA rows where the model abstained (empty `Answer:`
line) had `model_answer` literally set to `"Confidence: 3"` and were silently
scored as wrong. Diagnosed as a STALE-DATA artifact (an older extractor without
the current per-branch rubric guards), NOT a live code bug — so the fix is
**re-extraction (CPU), not re-inference (GPU)**. Added a conservative refusal
detector (handoff §19.3), wired it into the live pipeline, built a targeted
re-extraction script, and **applied it to all 19 Gemma2-9B-instruct CSVs**
(`.bak` backups written). 13 rows fixed. Code in working tree, **not committed**.

## 1. The problem (from a user-supplied Gemma2-9B-instruct TriviaQA CSV)

Gemma sometimes abstains — "I can't determine … without", "I'd need to consult
a reliable source" — and emits an empty `Answer:` line followed by
`Confidence: N`. Symptom: `model_answer == "Confidence: 3"`,
`answer_extraction_failed=False`, `was_forced=False` → a garbage answer scored
as a wrong answer (not flagged, not excluded).

Scope (scan for `model_answer` matching `^\*{0,2}\s*(Confidence|Correct)\b`
across ALL csvs under `~/Desktop/AI Research`):
- **Gemma2-9B-instruct: 13 rows, all TriviaQA** (GSM8K/SQA/LegalBench = 0).
- **Llama3.1-8B-*base*: 1 row** (idx 11562, `was_forced=True`, `finish=length`).
- **GPT-OSS / Gemma4-instruct / Qwen: 0** — confirms this is distinct from the
  GPT-OSS loop issue (part 1).

## 2. Root cause + the key correction

The extractor regex `^...Answer...:\s*(.+?)\s*$` uses `\s*` after the colon, and
`\s` matches newlines — so on an empty `Answer:` line it crosses the line break
and `(.+?)` captures the NEXT line (`Confidence: 3`).

**BUT the current repo already neutralizes this** via per-branch post-processing:
the triviaqa/gsm8k branches `re.split` off `Confidence`/`Correct` (data_utils.py
~line 302/331), and the letter/Yes-No branches reject non-answer text. So the
current extractor returns `None` for these rows, not `"Confidence: 3"`. The
stored garbage came from an OLDER extractor. → The data is stale; re-extraction
with the current code fixes it.

**Important reversal:** I first "fixed the root cause" by changing `:\s*` →
`:[^\S\n]*` (no newline crossing) in all 12 branch regexes. A blanket dry-run
showed this BROKE legitimate next-line answers (idx 2349:
`Answer:\nIgnatius J. Donnelly` → captured correctly by the old newline-cross,
lost by the fix). **So the regex change was reverted.** Lesson: the
newline-crossing is load-bearing for "answer on the line after Answer:"; the
per-branch guards already handle the abuse case.

## 3. What changed (code — working tree, NOT committed)

- **`data_utils.py`** — added `is_refusal_response(response, extracted_answer)`
  + `_REFUSAL_PATTERNS`/`_REFUSAL_RE` (conservative; handoff §19.3). Returns True
  only when the answer is empty AND the text matches an abstention pattern.
  Regex extractors are UNCHANGED (reverted).
- **`evaluation.py`** — imports `is_refusal_response`; computes `is_refusal`
  right after `answer_extraction_failed` (it's a subset of it — confidence
  fields are already NaN-ed there); adds `"is_refusal"` to the result dict.
  `save_utils.py` needs no change (`df = pd.DataFrame(results)` in main.py picks
  up the new key automatically).
- **`reextract.py`** (NEW) — re-parses `full_response` in existing CSVs without
  re-running the model. See §5 for the design + the two traps it avoids.

## 4. What changed (data — APPLIED, with .bak backups)

Ran `reextract.py --write` over all 19 `Gemma2-9B-instruct` CSVs
(`~/Desktop/AI Research/GEMMA/Gemma2Instruct/*/*.csv`). Each original saved to
`<name>.csv.bak`.
- **13 bug rows fixed**: `model_answer` → empty, `answer_extraction_failed=True`,
  `is_correct=False`, `verbalized_confidence`/`single_pass_confidence` → NaN,
  `more_likely_than_not`/`single_pass_correct` → blank, `is_refusal` set.
- `is_refusal` column ADDED to all 19 files (also flagged genuine non-bug
  refusals, e.g. 2 in LegalBench).
- Verified on `40seed99triviaqa…`: idx 15103/643/8936 now NaN answer +
  refusal=True + excluded; normal row idx 5588 (`Jasper Fforde`) untouched.

**Net for the paper:** those 13 TriviaQA items move from "wrong answers" to
"abstentions excluded from accuracy/calibration" → Gemma2-instruct TriviaQA
accuracy and calibration both shift, correctly.

## 5. reextract.py — design + the traps a naive version hits

Usage: `python3 reextract.py "<glob>" [--write] [--dataset X]`. Dry-run by
default; `--write` renames original → `.bak` then writes the corrected CSV.
Dataset inferred from filename.

It is TARGETED and conservative — a blanket "re-parse every row" version is
WRONG, as a full-dataset dry-run proved:
- **GSM8K trap**: pandas reads the numeric `model_answer` column as float, so
  `"15"` reads back as `15.0` → every row looks "changed" (`15.0`→`15`), and one
  row genuinely mis-extracted (`'2.0'→'00'`, flipping correct→wrong).
- **LegalBench/forced-answer trap**: `was_forced=True` rows hold their answer in
  `forced_answer_response`, NOT `full_response`. Re-parsing `full_response`
  destroyed correct forced answers (`'No'`→none, correct→wrong).

So reextract.py ONLY rewrites rows whose stored `model_answer` matches the bug
signature `^\*{0,2}\s*(Confidence|Correct)\b` (the unambiguous leaked-rubric
case). Every other row is left exactly as generated; it only gains the
`is_refusal` column (computed from the stored answer). Result: exactly the 13
intended rows change, zero collateral.

## 6. Key decisions

- **No re-run for this fix.** Re-extraction (CPU) is correct and sufficient;
  re-inference would change nothing (regex reverted) and break reproducibility.
- **Refusal handling = flag + exclude** (not coin-flip forcing), per §19.3.
  `is_refusal` is a subset of `answer_extraction_failed`, so exclusion is
  automatic; the flag enables a separate refusal bucket if wanted.
- **Llama base row left alone** — it's in a base file slated for a FULL re-run
  (§18.6), so an in-place re-extraction is moot.
- **Other instruct models (Gemma4-instruct, Qwen, GPT-OSS) NOT re-extracted** —
  they have 0 bug rows; re-extraction would only add the `is_refusal` column.
  Optional, for schema consistency only.

## 7. Verification

- `data_utils.py` / `evaluation.py` / `reextract.py` parse OK.
- `verify_rubric.py` → **ALL CHECKS PASSED** (is_refusal wiring didn't break the
  5-tuple / forced-answer / column contracts).
- Targeted dry-run across all 19 files = exactly 13 changed rows; post-write
  spot check confirmed the fix + an untouched normal row.

## 8. Files modified / added this session (part 2)

| File | Change |
|---|---|
| `data_utils.py` | + `is_refusal_response` + refusal patterns. Regex extractors unchanged (a root-cause edit was made then reverted). |
| `evaluation.py` | import + compute `is_refusal`; add to result dict. |
| `reextract.py` | NEW — targeted, forced-row-safe re-extraction utility. |
| 19× `Gemma2-9B-instruct` CSVs | DATA: 13 rows fixed + `is_refusal` column; each has a `.csv.bak`. |

Not committed. Combined with part 1, the uncommitted source set is:
`confidence.py`, `data_utils.py`, `evaluation.py`, `reextract.py`.

## 9. Pending / next steps

1. **Commit** (when ready): `confidence.py` (GPT-OSS guard, part 1) +
   `data_utils.py`, `evaluation.py`, `reextract.py` (part 2). Do NOT stage
   `*.pyc` (confidence_telemetry_handoff.md §19.1).
2. Optional: `reextract.py --write` on Gemma4-instruct / Qwen / GPT-OSS CSVs to
   standardize the `is_refusal` column (no bug rows there — column-only).
3. The actual GPU re-runs remain separate and pending: GPT-OSS loop rows
   (part 1, selective) and Llama/Gemma base (full, §18.6).
4. The `.bak` files can be deleted once the re-extracted CSVs are confirmed good.

---

# Session 2026-06-07 (part 3) — GPT-OSS two-pass + forced-answer harmony bugs

This session was triggered by the user supplying
`triviaqa_confidencewithnewSE_GPT-OSS-20B-instruct.csv` and asking why specific
rows had empty confidence fields or a bogus `model_answer`.

Three distinct bugs were diagnosed and fixed in `confidence.py`. None are
committed yet (they are in the working tree alongside the earlier part-1 and
part-2 changes).

## 1. Bug A — two-pass critique extractors received harmony analysis text (idx 5)

**Symptom:** `more_likely_than_not` was empty for idx 5, even though the raw
critique response contained `"Correct: Yes"`. `single_pass_confidence` was also
empty.

**Root cause:** `get_two_pass_confidence` passed the raw `critique_response`
(which contains both GPT-OSS harmony channels — `analysis<…>assistantfinal<…>`)
directly to `extract_verbalized_confidence` and `extract_more_likely_than_not`.
The helper `_truncate_to_first_block` cut the response at the first
`Correct: Yes` occurrence, which lived in the analysis channel mid-sentence.
`extract_more_likely_than_not` requires the line-start anchor `^`, so a
mid-sentence `Correct:` never matched — returning `None`.

**Fix:** Strip the harmony envelope before passing to extractors. If
`_HARMONY_FINAL_DELIM` (`"assistantfinal"`) appears in `critique_response`, use
`rsplit(_HARMONY_FINAL_DELIM, 1)[-1].strip()` to get only the committed final
section; otherwise pass the response unchanged. Extractors then see only the
clean final channel, where `Correct:` appears at line start.

## 2. Bug B — two-pass `model.generate()` missing ngram guard for GPT-OSS (idx 8959)

**Symptom:** All confidence fields (`verbalized_confidence`, `more_likely_than_not`,
`single_pass_confidence`, `single_pass_correct`) were empty for idx 8959.

**Root cause:** `get_two_pass_confidence` called `model.generate()` without
`no_repeat_ngram_size`. The main forward pass in `generate_with_logits` already
had the guard for GPT-OSS (from session part 1), but the two-pass critique's
separate `model.generate()` call did not. GPT-OSS's critique generation looped
to the full `max_new_tokens` budget and never produced `Confidence:` or
`Correct:` output.

**Fix:** Added `no_repeat_ngram_size=3` to the two-pass `model.generate()` call
when `MODEL_VARIANT == "base"` or `MODEL_FAMILY == "gptoss"`, mirroring the
same predicate used in `generate_with_logits`.

## 3. Bug C — `get_forced_answer` leaked harmony analysis text into `model_answer` (idx 2322)

**Symptom:** `model_answer` for idx 2322 was a long analysis-channel reasoning
blob (started with `"analysis…"`). `single_pass_confidence` and
`single_pass_correct` were empty.

**Root cause:** When the main pass truncated before reaching `assistantfinal`,
the forced-answer call was triggered with a 32-token budget. GPT-OSS again hit
`max_new_tokens` before `assistantfinal`. `get_forced_answer` had no harmony
stripping — the lax fallback `extract_model_answer(f"Answer: {blob}")` accepted
the analysis text as the answer.

**Fix:** Two guards added to `get_forced_answer`:

1. **Harmony stripping:** If `_HARMONY_FINAL_DELIM` in `forced_response_clean`,
   take the post-delimiter slice before extraction.
2. **Analysis-channel rejection:** If the extracted answer matches
   `_ANALYSIS_MARKER_RE` (`^analysis`, case-insensitive, no `\b` — the channel
   name runs directly into the next word without a separator), set it to `None`.

**Module-level constants added** (after the import block, lines 16–21):
```python
_HARMONY_FINAL_DELIM = "assistantfinal"
_ANALYSIS_MARKER_RE = re.compile(r'^analysis', re.IGNORECASE)
```

## 4. Pending — idx 4809 substring-match investigation (NOT YET FIXED)

**Symptom:** idx 4809 `model_answer="Firenze"`, `ground_truth="Florence"`, but
`is_correct=True`.

**Diagnosis:** `check_triviaqa_correct()` uses `model_lower == acc` against
`normalized_aliases`. TriviaQA's official alias list for "Florence" includes
`"firenze"` (the Italian name) → exact match → correctly marked correct by the
dataset's own standard. This specific case is semantically correct. The broader
substring predicates `model_lower in acc` and `acc in model_lower` can cause
false positives in edge cases, but no fix was implemented this session — the
alias-match path is the TriviaQA official standard and requires further analysis
to confirm false-positive rate before restricting.

## 5. Files modified this session

| File | Changes |
|---|---|
| `confidence.py` | Module-level `_HARMONY_FINAL_DELIM` + `_ANALYSIS_MARKER_RE` constants; `get_two_pass_confidence` gains ngram guard and harmony stripping before extraction; `get_forced_answer` gains harmony stripping and analysis-marker rejection. |

**Not committed.** Combined with part 1 and part 2, the full uncommitted source
set is: `confidence.py`, `data_utils.py`, `evaluation.py`, `reextract.py`.

## 6. Open items / next steps

1. **Commit** all accumulated changes (confidence.py + data_utils.py +
   evaluation.py + reextract.py). Do NOT stage `*.pyc`.
2. **GPU re-run** GPT-OSS TriviaQA loop rows (identified in part 1) — the two-pass
   harmony fix (Bug A) and ngram guard (Bug B) will now produce non-empty confidence
   fields for those rows.
3. **Validate idx 4809 and similar**: confirm the `acc in model_lower` substring
   predicate doesn't create measurable false positives across the full TriviaQA
   alias set before deciding whether to restrict to exact-match only.
4. The `.bak` files from part 2 can be deleted once the re-extracted CSVs are
   confirmed good.

---

# Session 2026-06-07 (part 4) — Llama-3.1-8B-base TriviaQA second run: Priority-2 regression + fix

## 0. Context

User supplied `triviaqa_confidencewithnewSE_Llama3.1-8B-base (2).csv` (40 rows, same
model, newer run that already includes the §21 / `cb0ada1` fixes) and asked whether
the accumulated fixes would address the errors visible in it.

## 1. Analysis: which aef=True rows would the fixes address

| idx | Topic | Verdict |
|-----|-------|---------|
| 5588 | Thursday Next / Jasper Fforde | **Partially fixed** — "My answer is Jasper Ffford." hits new Priority-2 pattern, but captured verbosely without the sentence-boundary split (see §2) |
| 16373 | Achille Lauro | Not fixed — narrative clause, no structured format |
| 6536 | To Kill a Mockingbird | Not fixed — "going to go with X as my answer" unmatched |
| 3101 | Erasmus | Not fixed — response truncated mid-sentence |
| 8936 | Bolton Wanderers | Not fixed for aef — answer in narrative; BUT a **new regression found here** (see §2) |

## 2. Two bugs found and fixed in `data_utils.py` Priority-2

### Bug 1 — sentence-boundary split absent from Priority-2

The fix in commit `cb0ada1` added a sentence-boundary split `\.\s+(?:I[\s\']|My\s|...)` to
Priority-1 to prevent verbose model self-commentary from bloating model_answer. The same
split was missing from Priority-2. For idx 5588, the "My answer is:" pattern captured:
`"Jasper Ffford. I am 70 percent confident in this answer"` — still verbose.

Fix: same split applied to `ans` in the Priority-2 loop immediately after quote-stripping.

### Bug 2 — Priority-2 "My answer is correct" false positive

For idx 8936, the model wrote `"I am 100% confident that my answer is correct."` The
Priority-2 pattern `[Mm]y (?:final )?[Aa]nswer is:?\s*(.+?)(?:\n|$)` matched and extracted
`"correct"` as the trivia answer. `"correct"` is not a bare number so the existing guard
didn't catch it. Result: `model_answer="correct"`, `aef=False`, `is_correct=False` — the
extractor fabricated a wrong answer when the model had no answer at all.

Fix: added a meta-commentary blocklist after the sentence-boundary split:
```python
if re.match(r'^(?:correct|incorrect|right|wrong|true|false|unknown|unsure)$', ans, re.I):
    continue
```
`continue` tries the next Priority-2 pattern rather than returning garbage. Falls through to
`None` → `aef=True`, which is correct.

## 3. Verification

Ran all cases (idx 5588, 8936, 16373, 6536, 3101 + all §21 regressions: strong winds, Gregory
Peck, J.K. Rowling) against the updated code. All pass — idx 5588 → `"Jasper Ffford"` (not
verbose); idx 8936 → `None` (not `"correct"`); all regression cases unchanged.

## 4. Files modified this session

| File | Change |
|---|---|
| `data_utils.py` | Priority-2 triviaqa loop: sentence-boundary split + meta-commentary filter. Committed `056e4ce`. |

## 5. Other CSV (2) observations (no code changes)

- **idx 12604** (Art Tatum two-pass loop): two-pass critique column shows the full prompt
  template looping. Already fixed by `gptossfix` commit's ngram guard in
  `get_two_pass_confidence`. Not visible in this run because that run predated the fix.
- **Verbose model_answers** (idx 2373 River Taff, idx 15595 Fisher King, idx 12604 Art Tatum):
  the sentence-boundary split doesn't apply to these because the verbose clause doesn't follow
  the `. I/My/So/This/It/In ` trigger pattern. All have `is_correct=True` via substring
  matching — ugly but functionally correct. No code change made.
- **No countdown loops** visible in this run. The `_truncate_countdown_loop` fix from the
  Llama-3.1-8B-base first-run session is irrelevant for this particular data snapshot.

## 6. Updated handoff reference

See `confidence_telemetry_handoff.md §24` for the canonical write-up of these fixes.
