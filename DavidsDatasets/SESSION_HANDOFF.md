# Session Handoff — DCAK RTA v8 Transfer (DavidsDatasets)
**Last updated:** 2026-04-24  
**Model evaluated this session:** Qwen3.6-35B-A3B (`qwen3` family)  
**Primary dataset:** MMLU-Pro (`mmlu35b.csv`)  
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
