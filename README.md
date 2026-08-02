# DCAK RTA — LLM Confidence Telemetry Study

**Purpose of this document:** a single, self-contained context file to hand to an LLM (or a new
collaborator) at the start of a session so it doesn't need to re-derive the project from scratch.
It summarizes the research goal, the pipeline architecture, the current active scope, the
non-obvious bugs/decisions baked into the code, and where to look for more detail.

Repo: `github.com/glitching-pineapple/DCAK_RTA_v8-Transfer`, branch `main`.
All real code and docs live under **`DavidsDatasets/`** (the repo root has no other package).

---

## 1. What this project is

A research pipeline that evaluates how well large language models can express their own
**uncertainty/confidence**, across multiple model families and benchmark datasets, using several
independent confidence signals collected side-by-side per sample:

- **Logit-based confidence** — statistics over the actual per-token generation probabilities
  (sequence log-prob, geometric/arithmetic mean, min prob), plus a newer **top-20 logit entropy**
  signal (Shannon entropy over the softmax of the top-20 candidate logits at each output step).
- **Verbalized confidence** — the model self-rates 1–10 on an explicit probability-range rubric,
  collected via a **three-generation elicitation architecture** (see §4) that separates reasoning,
  own-work self-rating, and a blinded critique into independent forward passes, to reduce
  sunk-cost inflation in self-assessment.
- **More-likely-than-not** — a binary Yes/No judgment ("is your answer more likely correct than
  not"), collected the same way.
- **Semantic entropy (SE)** — Kuhn et al. (2023): sample N answers at temperature 0.5, cluster by
  DeBERTa-MNLI bidirectional entailment, compute entropy over cluster probability mass.
- (Historical/deprecated) **Answer-token entropy (ATE)** — Shannon entropy over the A–J logit
  distribution at the answer-letter position, only meaningful for MCQ datasets, which are no
  longer active (see §2).

The end goal is a paper studying LLM calibration across signal types, model families, and task
types — see `DavidsDatasets/Results_Draft.md` for the full related-works framing and
`DavidsDatasets/methods_draft.md` for the methods section draft.

---

## 2. Current active scope (read this before touching anything — a lot has been deprecated)

As of the **2026-06-18 deprecation notice** (recorded in `SESSION_HANDOFF.md` and
`confidence_telemetry_handoff.md`):

| | Active | Deprecated / removed from study |
|---|---|---|
| **Datasets** | GSM8K, StrategyQA, TriviaQA, LegalBench | ~~MMLU-Pro~~, ~~MedQA~~ |
| **Models** | Qwen2.5-7B, Qwen3-30B-A3B, Llama-3.1-8B, Gemma-2-9B, Gemma-4-31B, GPT-OSS-20B (instruct + base variants where they exist) | ~~Llama-4-Scout-17B-16E-Instruct~~ (code for it still exists in `config.py`/`model_utils.py` but should not be used) |
| **Signals** | logit-based, verbalized, more-likely-than-not, semantic entropy, top-20 logit entropy | ~~answer_token_entropy~~, ~~chosen_answer_raw_prob~~, ~~top_answer_letter~~, ~~answer_letter_probs~~, ~~prob_A–prob_J~~ (MCQ-only, no MCQ dataset remains active) |

The code paths for deprecated items still exist in the pipeline for backward compatibility /
potential future reuse — they're just not exercised or reported on in current analysis.

---

## 3. Repository layout

```
DavidsDatasets/
  config.py              All hyperparameters: model family/variant, dataset, token budgets,
                          SE settings. Single source of truth — read this first for "what
                          config produced this CSV".
  main.py                 Entry point. Loads dataset + model, runs a smoke test on one sample,
                          then loops over N_SAMPLES (or SPECIFIC_INDICES), aggregates, saves.
  model_utils.py           Model/tokenizer loading (device_map, dtype by family),
                          `generate_simple_response` shared helper (chat-template-aware,
                          has a `loop_guard` toggle).
  data_utils.py            Dataset loaders (`load_gsm8k`, `load_mmlupro`, `load_strategyqa`,
                          `load_medqa`, `load_triviaqa`, `load_legalbench`), ground-truth
                          extraction, and the answer-extraction regex cascades
                          (`extract_model_answer`, `extract_model_answer_strict`),
                          correctness checkers, refusal detection.
  confidence.py             The core of the pipeline (~1500 lines): prompt construction
                          (`create_prompt`), `generate_with_logits` (the main generation +
                          logit-metric call, with decoding guards), verbalized-confidence
                          extraction/elicitation, the Gen-2/Gen-3 own-work + blinded-critique
                          functions, forced-answer fallback, top-20 logit entropy.
  evaluation.py            Per-sample orchestration (`evaluate_sample`): branches between the
                          three-generation reasoning flow and the standard single-pass flow,
                          wires truncation detection, forced answers, refusal flags.
  semantic_entropy.py       `SemanticEntropyCalculator` — NLI-based clustering + SE computation
                          (Kuhn et al. 2023 implementation).
  save_utils.py             Saves CSV + detailed JSON + a `.pt` file of top-20 logit tensors
                          per run.
  visualization.py          AUROC, calibration tables, plots, semantic-entropy summaries
                          printed/plotted at the end of a `main.py` run.
  verify_rubric.py          Offline (no GPU/torch-model-required — stubs torch/transformers)
                          verification suite for prompt/extractor/contract invariants. Run
                          this after any change to confidence.py/data_utils.py/evaluation.py.
  reextract.py             CPU-only, targeted re-parsing tool for existing CSVs — re-derives
                          derived columns (model_answer, is_correct, is_refusal) from the
                          already-saved full_response text without re-running the model.
                          Deliberately conservative (see §7).
  prm_scoring.py, aditya_prm.ipynb
                          Auxiliary/exploratory: process-reward-model (PRM) step-scoring
                          utilities, separate from the main confidence-telemetry pipeline.

  # Handoff / narrative documents (chronological engineering logs — read for deep dives):
  SESSION_HANDOFF.md              Chronological session-by-session bug/fix log for the
                                 generation pipeline. Long (1600+ lines); most recent entries
                                 are at the bottom of each dated section.
  confidence_telemetry_handoff.md Originally written for the HTML visualization series (see
                                 §8), but its later sections (§18 onward) are the canonical,
                                 more organized write-up of the generation-pipeline fixes —
                                 prefer this over SESSION_HANDOFF.md for the "why" behind a
                                 given guard/fix. Also long (2000+ lines).
  LLAMA_BASE_HANDOFF.md            Focused handoff for Llama-3.1-8B-base bug history.
  methods_draft.md                 Paper methods section draft — the cleanest single summary
                                 of the pipeline's design intent (read this if you only read
                                 one doc besides this README).
  Results_Draft.md                 Related-works / paper-framing document — citations,
                                 novelty claims, narrative frames per benchmark.

  # Data (large, not fully enumerated here):
  GSM8k/, MMLUPro/, StrategyQa/, 150 rubric/, 150 VerbConf/, Trash/
                                 Result CSVs/JSONs from past runs, organized by
                                 dataset/model. `Trash/` and `.ipynb_checkpoints/` are stale
                                 junk. Note: no `.gitignore` exists yet in this repo —
                                 `__pycache__/*.pyc` files are tracked (a known, still-pending
                                 cleanup item).
```

---

## 4. Pipeline architecture

### 4.1 Config-driven single run

Every run is driven entirely by `config.py`: set `MODEL_FAMILY`, `MODEL_VARIANT`, `DATASET`
(+ `LEGALBENCH_TASK` if applicable), `N_SAMPLES`/`RANDOM_SEED` or `SPECIFIC_INDICES`, then run
`main.py`. Output filenames encode dataset + model label:
`{DATASET}_confidence_{label}.csv`, `_detailed_{label}.json`, `_top20_logits_{label}.pt`.

### 4.2 `USE_REASONING_FLOW` — the central branch

Models that emit a reasoning/thinking block before committing to an answer (Qwen3, GPT-OSS,
Gemma-4-instruct) go through a **three-generation pipeline**; everything else (Qwen2.5, Llama-3.1,
Gemma-2, Gemma-4-**base**) goes through the original **single-pass flow**. The flag is computed in
`config.py`:

```python
USE_REASONING_FLOW = (
    MODEL_FAMILY in ("qwen3", "gptoss")
    or (MODEL_FAMILY == "gemma4" and MODEL_VARIANT == "instruct")
    or (MODEL_FAMILY == "llama4scout" and MODEL_VARIANT == "instruct")  # deprecated model
)
```

**Why:** thinking blocks (`<think>...</think>` for Qwen3/Gemma4, or OpenAI's harmony
`analysis...assistantfinal...` envelope for GPT-OSS) can consume 1,000–3,000+ tokens before any
structured output, exhausting token budgets if confidence elicitation is attempted in the same
pass. Separately, a single-pass design lets the model rate the reasoning it *just wrote*
immediately — a "sunk cost" effect that inflates verbalized confidence.

**Gen 1 (reasoning + answer only):** no confidence rubric in the prompt at all
(`include_confidence=False`). Produces `full_response`, and all logit-based metrics
(`seq_confidence_mean`, `logit_confidence_*`, `top20_entropy_*`) are computed from this pass's
actual token probabilities.

**Gen 2 (own-work-aware confidence):** a short, separate forward pass. The model is told
explicitly "the following is YOUR OWN reasoning chain... how confident are you that YOUR answer is
correct?" Produces `single_pass_confidence` / `single_pass_correct`. (This own-work framing was
later — 2026-06-08 — extended to **all** models, not just reasoning-flow ones, via
`get_gen2_confidence`, so the "single_pass" columns are consistently sourced this way across the
whole model lineup.)

**Gen 3 (blinded critique):** the model is told it's reviewing **someone else's** solution (not
told it's its own), given Gen 2's self-reported score as context to push back against. Produces
`two_pass_critique`, and the primary `verbalized_confidence` / `more_likely_than_not` columns
(falls back to Gen 2's values if Gen 3 extraction fails).

Acknowledged limitation: Gen 2/3 read the model's own reasoning as **text** in a fresh forward
pass, not from Gen 1's internal activations — logit metrics partially compensate for this.

### 4.3 Confidence rubric

A single 10-class bulleted rubric (`_CONF_RUBRIC` in `confidence.py`) is shared across every
elicitation context (Gen 1 prompt, Gen 2, Gen 3) and all 4 active datasets, harmonized on
2026-05-10 — data from before that date uses a different rubric per dataset and should not be
mixed into cross-dataset calibration analysis.

### 4.4 Base vs. instruct handling

Base models get fundamentally different prompting than instruct models throughout the pipeline,
because they have no instruction-following training and treat any instruction-style prompt as raw
continuation (producing EOS or garbage):

- Main prompt: instruct gets the full rubric via chat template; base gets the same content plus a
  trailing `Solution:\nLet me {verb} this step by step.` continuation primer (`create_prompt`
  builds `instruction_body` and `base_primer` separately).
- Forced-answer fallback: instruct gets an instruction-style "commit to your best guess" prompt;
  base gets a minimal `Q: {question}\nA:` matching its pretraining distribution.
- Two-pass critique: instruct uses the chat template; Llama-base specifically uses a minimal
  native Q&A format (`Q:...\nA:...\nQ: Is this answer correct?...\nA:`) because it uniquely can't
  even pattern-complete the denser Qwen/Gemma-base-style critique prompt.
- `single_pass_correct` for base models: NOT elicited generatively (found to be unreliable — the
  model's next token after `A:` isn't consistently Yes/No). Uses **direct logit comparison**
  (`get_correct_separate_base`) between the ` Yes` / ` No` token logits at the first output
  position instead of generation.

### 4.5 Decoding guards (anti-loop)

Greedy decoding on base models (and GPT-OSS, which is instruct but prone to Harmony-channel
repetition loops) can degenerate into verbatim repetition. The resolved guard policy, applied
per `model.generate()` call site independently (a lesson learned after guards were added to the
main pass but missed on the two-pass/forced-answer calls, silently leaving those un-guarded):

| Config | `no_repeat_ngram_size` | `repetition_penalty` | `stop_strings` |
|---|---|---|---|
| Qwen/Gemma/Llama **instruct** | 0 | 1.0 | — |
| **GPT-OSS** instruct | 3 | 1.0 | — |
| **base** models | 3 | 1.0 (disabled — see below) | ✓ (`\nQuestion:`, `\nSolution:`, etc.) |

Guiding principle: *use the minimal constraint that prevents non-termination; prefer constraints
that are inert on well-behaved generations over `repetition_penalty`*, because
`repetition_penalty` warps every token's distribution (including the very output-format tokens —
`Answer:`, `Confidence:` — the eval prompt asks for), which was empirically found to make base
models abandon structured output entirely in favor of unparseable free prose.
`generate_simple_response` also has a `loop_guard` toggle so it can be turned **off** for the
forced-answer call specifically — that call's context is often dense with the question's own
`Answer:`/`Question:` tokens, and the n-gram guard was found to over-ban legitimate completions.

### 4.6 Answer extraction, truncation, forced-answer recovery

- `generate_with_logits` returns a 5-tuple `(text, token_probs, tokens, raw_scores, meta)`, where
  `meta` (from `_detect_truncation`) carries `finish_reason` (`eos`/`stop`/`length`) and
  `was_truncated`.
- `extract_model_answer_strict` (Priority-1 only — anchored `Answer:` line) is the correctness
  gate: if it fails, `was_forced=True` and `get_forced_answer` is called to force a commitment
  (rather than dropping the row, which would bias calibration toward easy items). Forced rows are
  **kept** in the dataset; partition on `was_forced` for "honest vs. forced" analysis.
- `extract_model_answer` (lax, Priority-1 → 2 [→ 1.5 for TriviaQA]) is used on forced-answer
  output and as a general fallback; has dataset-specific guards (bare-number rejection for
  TriviaQA, meta-commentary blocklists, sentence-boundary truncation to avoid capturing the
  model's self-commentary after its actual answer).
- GPT-OSS's Harmony envelope (`analysis...assistantfinal...`) and Qwen3/Gemma4's `<think>...
  </think>` block are both stripped before any extraction attempt (`_strip_harmony_envelope`,
  `_QWEN3_THINK_RE`).
- `is_refusal_response` flags genuine abstentions (conservative — 10-pattern regex scanned only
  against the **tail** of long responses, so mid-reasoning "I don't recall exactly..." phrases in
  an opening sentence don't false-positive). Refusals are a subset of
  `answer_extraction_failed`; excluded from accuracy/calibration rather than force-guessed.

### 4.7 Re-run vs. re-extract — the maintenance decision rule

When a CSV looks wrong, the fix is **not always** a GPU re-run:

| | Re-inference (`main.py`) | Re-extraction (`reextract.py`) |
|---|---|---|
| Changes | The generated tokens themselves | Only derived fields (`model_answer`, `is_correct`, `is_refusal`) parsed from already-saved `full_response` |
| Use when | The answer isn't recoverable from stored text (loop/truncation), or a **decoding setting** changed (alters every row's tokens) | The raw record is fine and only the **parsing** of it was buggy |
| Cost | GPU hours; breaks byte-for-byte reproducibility of the run | CPU, seconds; `.bak`-backed, reversible |

`reextract.py` is deliberately narrow/targeted (only rewrites rows matching a specific known
bug signature) — a blanket re-parse-everything version was tried and found to corrupt otherwise-
good rows (e.g. pandas reads numeric GSM8K answers back as floats; forced-answer rows store their
real answer in `forced_answer_response`, not `full_response`).

---

## 5. Result CSV schema (per-sample columns)

Common to (almost) every run:

```
idx, question, ground_truth, model_answer, is_correct
verbalized_confidence, single_pass_confidence, single_pass_correct, more_likely_than_not
logit_confidence_geom, logit_confidence_mean_prob, logit_confidence_min, seq_confidence_mean
top20_entropy_mean, top20_entropy_last_token        (newest signal, June 2026)
two_pass_critique, two_pass_finish_reason, two_pass_was_truncated
main_pass_finish_reason, main_pass_was_truncated, was_forced, forced_answer_response
answer_extraction_failed, is_refusal
full_response
```

MCQ-only columns (`answer_token_entropy`, `chosen_answer_raw_prob`, `answer_letter_probs`,
`prob_A`…`prob_J`, `top_answer_letter`) appear in older MMLU-Pro/MedQA files only — deprecated,
see §2. A separate `.pt` file per run stores the raw top-20 per-token logit tensors that back the
`top20_entropy_*` columns (too large to embed in the CSV).

---

## 6. HTML visualization series (separate sub-project)

There is also a series of self-contained, single-file HTML pages — one per (model × benchmark)
cell — that embed a cleaned CSV as inline JSON and render KPIs, a hoverable confidence histogram,
a calibration table (click-to-filter into the item browser), a confusion matrix or signal-
comparison panel, and a searchable item browser with full prompt/response/critique text. Each page
has its own hand-picked visual identity (e.g. LegalBench = judicial/parchment, GSM8K-Gemma =
blueprint-engineer) but shares a common structural rhythm and CSS class vocabulary. There's also a
cross-cluster Pearson/Spearman/Kendall correlation-matrix page (currently Qwen-only).

Full build process, per-cluster aesthetics, common pitfalls (e.g. `</script>` escaping, boolean/
NaN cleanup, static count strings needing manual updates on refresh) are documented in
`confidence_telemetry_handoff.md` §1–17. This is a distinct workflow from the Python generation
pipeline (§1–4 above) — the HTML pages consume already-generated CSVs, they don't run models.

---

## 7. Known open items (as of the most recent commits)

- No `.gitignore` — `__pycache__/*.pyc` is tracked and shows as modified after every import.
  Fix documented in `confidence_telemetry_handoff.md` §19.2 but not yet applied.
- Base-model (Llama, Gemma) full re-runs across benchmarks are pending — their pre-fix CSVs have
  unrecoverable loop rows and were generated before the decoding-guard fixes landed.
- Selective GPU re-run pending for GPT-OSS loop rows on TriviaQA/StrategyQA/LegalBench (GSM8K had
  zero loops, needs nothing).
- `check_triviaqa_correct`'s substring-match tiers (`model_lower in alias`, `alias in model_lower`)
  have at least one known false-positive edge case under investigation (short answers matching
  inside long aliases) — see `confidence_telemetry_handoff.md` §28.1.
- SE settings (`SE_NUM_SAMPLES=1`, `SKIP_NLI_CLUSTERING=True` in `config.py`) are temporary speed
  settings for active development — must be restored (`SE_NUM_SAMPLES=5`,
  `SKIP_NLI_CLUSTERING=False`) before any semantic-entropy-dependent analysis.
- Several commits with non-descriptive messages (`plswork`, `fixxx`, `HOPEFULFIXFORLlamaTriviaqa`)
  reflect an iterative, CSV-driven debugging style — check `git log -- <file>` and the handoff
  docs together when trying to understand why a given line exists.

---

## 8. Where to look for more

| Question | Look here |
|---|---|
| "What does the pipeline do, end to end, methodologically?" | `DavidsDatasets/methods_draft.md` |
| "Why does this specific guard/regex/branch exist?" | `DavidsDatasets/confidence_telemetry_handoff.md` §18+ (organized) or `SESSION_HANDOFF.md` (chronological, more granular) |
| "What's the paper's related-work / novelty framing?" | `DavidsDatasets/Results_Draft.md` |
| "Llama-3.1-8B-base specific bug history" | `DavidsDatasets/LLAMA_BASE_HANDOFF.md` |
| "How do I add a new HTML visualization page?" | `DavidsDatasets/confidence_telemetry_handoff.md` §4–13 |
| "What's the current model/dataset/config for a given CSV?" | The CSV filename encodes `{N}seed{S}{dataset}_confidence{withnewSE?}_{ModelLabel}.csv`; cross-reference `config.py::get_model_label()` |
