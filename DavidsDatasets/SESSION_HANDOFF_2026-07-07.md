# Session Handoff — Code Review Fixes (2026-07-06 → 07)

**Project:** DCAK RTA v8 · Confidence-Estimation Pipeline (`DavidsDatasets/`)

> **Context:** A full review of the pipeline found 20 issues; all were fixed, tested, and left
> **uncommitted** for review. Nothing changes prompts or decoding behavior — all 54 prompt
> strings are byte-pinned and proven identical to the pre-refactor code (git HEAD).
>
> A PDF version of this handoff exists at `SESSION_HANDOFF_2026-07-07.pdf`.

---

## Result-corrupting bugs (P0 — old CSVs affected)

| # | Problem | Fix |
|---|---------|-----|
| 1 | `seq_confidence_mean` was actually the log-prob **sum** → AUROC partly measured answer length, not confidence | Now a true per-token mean; sum kept as new `seq_log_prob_sum` column; both shown in the AUROC table. ⚠️ **In old CSVs the column still holds the sum** |
| 2 | Calibration/ECE binned 1–10 values on a 0–1 grid → every row became NaN, **ECE was never computed** | Rubric-aware binning (class N → midpoint (N−0.5)/10); ECE vs empirical bin confidence. Verified on the real MedQA CSV |
| 3 | `verbalized_confidence` silently mixed two-pass and single-pass scores on difficulty-correlated rows | New `two_pass_confidence` column + `verbalized_conf_source` tracking; per-method AUROC & calibration reported separately |
| 4 | Failed samples were silently dropped → accuracy biased toward easy items | Failures logged with traceback, loud bias warning with failure rate, saved to `*_errors_*.json` sidecar |
| 5 | Config claimed SE enabled while debug flags silently disabled it (NLI model still loaded onto GPU) | Single `SEMANTIC_ENTROPY_ACTIVE` flag consumed everywhere; loud import-time warning when vetoed |
| 6 | Only numpy was seeded → sampled generation non-reproducible despite the advertised seed | `set_seed()` seeds python / numpy / torch CPU+CUDA at both entry points |

## Correctness & robustness (P1)

| # | Problem | Fix |
|---|---------|-----|
| 7 | Orphaned SE helper crashed with `TypeError` if ever called | Removed (zero callers confirmed, incl. notebooks) |
| 8 | Base/GPT-OSS rows decoded under different anti-loop guard constraints — an invisible cross-family confound | New `decoding_guards_active` column per row so analyses can condition on decoding policy |
| 9 | TriviaQA substring match: "no" ⊂ "North Carolina", "art" ⊂ "Descartes" scored as correct | Word-boundary containment + length floor + stopword block; 16-case regression suite passes |
| 10 | Answer-letter entropy discarded the pre-renormalization probability mass (its reliability signal) | Persisted as `answer_letter_mass` column — low mass ⇒ that row's entropy is untrustworthy |

## Performance & infrastructure (P1)

| # | Problem | Fix |
|---|---------|-----|
| 11 | Serial batch-size-1 generation (up to 5 sequential `generate` calls per sample) | Opt-in `DCAK_GEN1_BATCH_SIZE` batched Gen-1 (guarded families auto-fallback to serial); validate per model with `smoke_test_batched.py` before enabling |
| 12 | Guarded clean re-scoring pinned multi-GB logit tensors on GPU per sample | Only needed positions materialized (`num_logits_to_keep` + version fallbacks), slice cloned, offloaded to CPU |
| 13 | Crash at row N lost every completed result; CSV round-trip mangled Unicode (the "â¯" artifacts) | `IncrementalJSONLWriter`: per-row flushed `*_rows_*.jsonl`, exact Unicode round-trip — the new source of truth |
| 14 | Smoke-test row evaluated twice; detected device never passed to the loader (CPU-only crashed) | Smoke result cached & reused; device passed through with CPU fallback |
| 15 | `trust_remote_code=True` everywhere with no revision pinning (supply-chain + reproducibility hole) | All 12 models + 6 datasets + NLI + PRM pinned to HF commit SHAs; flag dropped for natively-supported families |
| 16 | Any config change required editing source (blocked sweeps) | `DCAK_*` env overrides for every knob, e.g. `DCAK_DATASET=gsm8k python3 main.py`; derived values track automatically |

## Maintainability & hygiene (P2)

| # | Problem | Fix |
|---|---------|-----|
| 17 | Rubric ×3, harmony delimiter ×3, extraction helpers ×2 copy-pasted across modules | Single-sourced in new `shared.py`; Gen-2/3 prompts extracted as pure builders — **proven byte-identical to git HEAD** |
| 18 | Regex extractors untestable — every model added fragile special cases | `check_extraction_golden.py` pins behavior over **1,410 real captured responses**; `check_prompt_golden.py` pins all 54 prompts |
| 19 | `prm_scoring.py` loaded a 7B model on import, hardcoded paths, no main guard | Proper CLI with `main()` guard, safe token-id lookup, added the AUROC/std the TODO asked for |
| 20 | 116 junk files tracked (checkpoint copies, `.pyc`, `.DS_Store`, 0-byte `=*`); blanket warning filter muted the pipeline's own diagnostics | `.gitignore` added, junk staged for deletion, warning filter narrowed, `plt.close()` added, `requirements.txt` created |

---

## Verification — all green, no GPU needed

| Check | Result |
|-------|--------|
| `verify_rubric.py` | ALL CHECKS PASSED |
| `check_prompt_golden.py` | 54/54 prompts byte-identical |
| `check_extraction_golden.py` | 1,410/1,410 real responses, no drift |
| `tests/test_batched_mock.py` | batched ≡ serial decoding (incl. pad==eos, truncation) |
| `tests/test_clean_logits.py` | guarded-path re-scoring, 3 transformers-version branches |
| `tests/test_evaluate_sample_e2e.py` | full result-dict assembly, merge + NaN policy (20/20) |

See `tests/README.md` for the full pre-commit routine and `--update` re-pinning workflow.

## New files this session

| File | Purpose |
|------|---------|
| `shared.py` | Single source for rubric, harmony delimiter, extraction helpers |
| `check_prompt_golden.py` + `prompt_golden.json` | Byte-pins all 54 prompts |
| `check_extraction_golden.py` + `extraction_golden.json` | Pins extractor behavior on 1,410 real responses |
| `smoke_test_batched.py` | GPU-box validation of batched decoding, per model |
| `tests/` (3 suites + README) | Mock-model unit tests, runnable anywhere |
| `requirements.txt` | Dependency spec (pin exact versions via `pip freeze` on the GPU box) |
| `.gitignore` (repo root) | Blocks checkpoints, `__pycache__`, `.DS_Store`, `*.bak`, `=*` |

## Your next steps

1. **Review & commit:** `git status` shows 114 staged deletions (junk) + 11 modified + ~10 new files.
2. **On the GPU box:** run `python3 smoke_test_batched.py` once per model family before enabling
   `DCAK_GEN1_BATCH_SIZE>1`; run `pip freeze` to pin exact versions into `requirements.txt`.
3. **Remember:** old CSVs' `seq_confidence_mean` holds the *sum*; `Trash/` was left untouched for you to delete.
4. **For real SE runs:** set `SKIP_NLI_CLUSTERING = False` and `SE_NUM_SAMPLES = 5` (or via
   `DCAK_SKIP_NLI_CLUSTERING=false DCAK_SE_NUM_SAMPLES=5`).
