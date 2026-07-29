# Code Review Changelog — Complete Record of Changes, Reasons, and Design Decisions

**Project:** DCAK RTA v8 · Confidence-Estimation Pipeline (`DavidsDatasets/`)
**Session dates:** 2026-07-06 → 2026-07-08
**Status:** All changes left **uncommitted** for review. Nothing in this session altered prompt text or decoding behavior — all 54 prompt strings are byte-pinned and proven identical to the pre-refactor code (git HEAD).

---

## Table of Contents

1. [How to read this document](#how-to-read-this-document)
2. [Impact on results — what changes and what doesn't](#impact-on-results)
3. [P0 — Result-corrupting bugs (fixes #1–#6)](#p0--result-corrupting-bugs)
4. [P1 — Correctness & robustness (fixes #7–#10)](#p1--correctness--robustness)
5. [P1 — Performance & infrastructure (fixes #11–#16)](#p1--performance--infrastructure)
6. [P2 — Maintainability & hygiene (fixes #17–#20)](#p2--maintainability--hygiene)
7. [Final audit pass — residual issues found & fixed](#final-audit-pass)
8. [New files created](#new-files-created)
9. [Verification matrix](#verification-matrix)
10. [Next steps](#next-steps)

---

## How to read this document

Each fix records four things:

- **The issue** — what was wrong and the mechanism by which it caused harm
- **Why it mattered** — the concrete effect on results, performance, or maintainability
- **The fix** — what was changed, in which files
- **Why this way** — the design alternatives considered and why this one was chosen

---

## Impact on results

**Guaranteed unchanged:** every prompt is byte-identical (54 golden snapshots verified against git HEAD), decoding settings are unchanged (greedy, same anti-loop guards, same token budgets), and the extractors produce identical outputs on all 1,410 pinned real responses. Re-running the same model on the same rows produces the same generations.

**Numbers that WILL legitimately change on re-analysis** (the old numbers were wrong):

| Metric | Direction of change |
|---|---|
| `seq_confidence_mean` AUROC | Was partly measuring **answer length**; corrected value can move either way |
| ECE / calibration tables | Were **never computed** (empty output); now produced for the first time |
| TriviaQA accuracy | Can only go **down or stay flat** (substring false positives removed) |
| Verbalized-confidence AUROC | Merged number replaced by per-method numbers, which may differ substantially |
| Accuracy on runs that had crashes | May drop slightly — failed (hard) rows are now counted in the denominator |

Any paper table built on `seq_confidence_mean` AUROC, ECE, or TriviaQA accuracy should be recomputed before being trusted.

---

## P0 — Result-corrupting bugs

### Fix #1 — `seq_confidence_mean` was the log-prob SUM, not a mean (length confound)

**The issue.** `evaluation.py` assigned `confidence_metrics["log_prob_sum"]` to the column named `seq_confidence_mean`. `compute_confidence_metrics` never computed a mean log-prob at all. Commit `ec7b005` ("Fix seq_confidence_sum to seq_confidence_mean") had renamed the **column** but not the **value**. The column fed AUROC directly with `higher_is_better=True`.

**Why it mattered.** `Σ log p` decreases monotonically with sequence length: a verbose-but-correct CoT scores below a terse-but-wrong one purely because it has more tokens. Correct and wrong answers systematically differ in CoT length, so the AUROC for this metric partly measured **answer length, not confidence** — a classic confound in the uncertainty-estimation literature that can fabricate or hide signal in a headline result.

**The fix.**
- `confidence.py` → `compute_confidence_metrics` now returns `mean_log_prob` (length-normalized) alongside `log_prob_sum`, with comments stating which is safe for cross-sample comparison.
- `evaluation.py` → `seq_confidence_mean` now holds the true mean; the sum is preserved as a **new** column `seq_log_prob_sum`.
- `visualization.py` → the AUROC table shows **both** ("Mean Log-Prob (per-token)" and "Log-Prob Sum (length-confounded)") so the next run directly quantifies how much of the old AUROC was the length artifact.
- Last-token variants point at `mean_log_prob` — value-identical (sum ≡ mean for one token), preserving comparability with old CSVs.

**Why this way.** Alternatives were (a) rename the column to `seq_log_prob_sum` and drop the mean, or (b) silently swap the value. (a) breaks every downstream consumer of the column name; (b) hides the correction. Keeping the honest name pointing at the honest quantity, plus preserving the sum under a new explicit name, keeps old files interpretable (documented caveat: **in old CSVs, `seq_confidence_mean` still holds the sum**). Note `logit_confidence_geom = exp(mean_log_prob)` was always length-normalized, so it is the trustworthy anchor for old results.

---

### Fix #2 — Calibration/ECE silently never computed (1–10 data binned on a 0–1 grid)

**The issue.** `extract_verbalized_confidence` returns integers 1–10, but `calibration_analysis` binned with `bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0]` and computed ECE against fixed midpoints `[0.1 … 0.9]`. Every 1–10 value fell outside the bins → `pd.cut` produced all-NaN → the groupby table was empty and the ECE loop iterated over nothing. Verified on the real MedQA CSV: **0 of 10 rows survived binning**. Print labels claimed "(0-1 scale)" in three places.

**Why it mattered.** ECE/calibration is a core deliverable of a calibration study, and it was never produced for any run. Empty output is easy to misread as "well-calibrated" or "nothing to report".

**The fix.** `visualization.py` → `calibration_analysis` rewritten:
- Maps the 1–10 rubric to probabilities via the **interval midpoint** `(N − 0.5) / 10` — the prompt defines class N as "(N−1)·10% to N·10% likely correct", so class 7 = 0.65, not 0.70. A `rubric_scale=False` flag handles columns already on [0, 1].
- ECE computed against each bin's **empirical mean stated confidence** (the standard definition), not fixed midpoints — fixed midpoints misstate ECE whenever the within-bin distribution is non-uniform, which is always true for a 10-class rubric.
- Out-of-range guard prints a warning and excludes rather than silently mis-binning.
- The three mislabeled "(0-1 scale)" strings corrected to "(1-10 rubric)".

**Why this way.** Dividing by 10 at the extraction source was the other option, but that changes stored column values and breaks comparability with every existing CSV. Converting inside the analysis keeps the stored data stable and makes the probability mapping (midpoint semantics) explicit and documented where it is used.

---

### Fix #3 — `verbalized_confidence` silently mixed two elicitation methods on difficulty-correlated rows

**The issue.** The primary column was built as `two_pass_confidence`, silently falling back to `single_pass_conf` when the two-pass value was `None`. The fallback fires exactly when the Gen-3 critique failed to parse — truncated `<think>` blocks, loops, messy long reasoning — i.e., disproportionately on **hard questions**. The raw two-pass score was not stored anywhere else; it only survived inside the merge.

**Why it mattered.** AUROC/ECE over the merged column measured an uncontrolled blend of two methods (blinded critique vs. self-rating), with the method-to-row assignment itself carrying difficulty information. Neither method was evaluated on a representative sample, and the merged number belongs to neither — the session's synthetic demo showed a merged AUROC of 0.73 while the components were 0.69 and 0.35. The core comparative claim ("is the blinded critique better calibrated than self-rating?") was unanswerable from the data.

**The fix.**
- `evaluation.py` → `two_pass_confidence` / `two_pass_correct` stored as their own columns; `verbalized_conf_source` / `more_likely_source` record which method produced each merged value; sources cleared in the hard-failure NaN block.
- `visualization.py` → AUROC table adds per-method rows; `compute_auroc` casts `astype(float)` (object-dtype columns holding `None`s).
- `main.py` → prints the source mix every run; calibration runs for all three columns.

**Why this way.** The clean-room alternative — force a single fixed method — throws away two-pass signal on rows where it worked and breaks continuity with existing CSVs that reference `verbalized_confidence`. Keeping the merge but making the mixture **visible and filterable** (source column) plus reporting **attributable per-method numbers** preserves both coverage and interpretability.

---

### Fix #4 — Main loop silently swallowed exceptions (selection-biased accuracy)

**The issue.** `main.py` caught every exception per sample, printed one line, and `continue`d. Failed rows vanished from `results` with no record.

**Why it mattered.** Failures are not random — OOM and truncation edge cases concentrate on long/hard inputs. Dropping them silently means accuracy/AUROC were computed over an **easier-than-random subsample**, with no visible denominator and no reproducibility (same seed, different effective sample across environments).

**The fix.**
- `main.py` → `_record_error` captures `{idx, error, traceback}`; after the loop, a loud warning prints the failure count/rate with an explicit "treat accuracy as computed over a non-random (easier) subsample" message and the failed indices; all-failed runs exit gracefully.
- `save_utils.py` → `save_results` writes failures to a `{dataset}_errors_{model}.json` sidecar.

**Why this way.** The alternative — inserting failure rows into the analysis DataFrame with `is_correct=None` — poisons dtypes (`object` columns break `.mean()`/groupby) and every downstream aggregate. Keeping errors **out of the DataFrame but loudly reported and persisted** keeps analysis type-safe while making the denominator auditable.

---

### Fix #5 — SE config lied about what runs (debug flags silently vetoed it)

**The issue.** `COMPUTE_SEMANTIC_ENTROPY = True` while `SKIP_NLI_CLUSTERING = True` and `SE_NUM_SAMPLES = 1` (debug leftovers). The SE computation was gated off in `evaluation.py`, yet `main.py` still loaded DeBERTa onto the GPU, printed "SE enabled", and called the SE analysis. Even if enabled, SE over 1 sample is mathematically undefined (needs ≥ 2).

**Why it mattered.** VRAM wasted on a model that never runs; result files silently missing the SE columns the config claims; anyone re-running "the medqa experiment" from this config believes SE was computed. Debug flags that invert the meaning of a results file are a reproducibility hazard.

**The fix.** `config.py` → derived `SEMANTIC_ENTROPY_ACTIVE = COMPUTE_SEMANTIC_ENTROPY and not SKIP_NLI_CLUSTERING and SE_NUM_SAMPLES >= 2`, with a loud import-time notice listing the veto reasons. All consumers (`main.py` NLI load, both `evaluate_sample` calls, prints, analysis; `evaluation.py` gate) use this single flag. `print_config` reports Requested / Skip-NLI / **ACTIVE THIS RUN** separately.

**Why this way.** A hard `assert` would block the debugging workflow the flags exist for. The user's debug values were deliberately **not** changed — the fix makes the config *honest*, not different. The notice uses `print` rather than `warnings.warn` because `main.py`'s warning filter (see fix #20) would have silenced it.

---

### Fix #6 — `RANDOM_SEED` did not make runs reproducible

**The issue.** The seed was applied only to `np.random` for index selection. SE's sampled generation (`do_sample=True`) drew from torch's unseeded global RNG.

**Why it mattered.** Any sampled generation differed run-to-run while the config banner advertised a seed — combined with #4's variable sample drop, "reproducible" runs weren't.

**The fix.** `main.py` → `set_seed()` seeds python `random`, numpy, torch CPU and all CUDA devices; called at the top of `main()` and `run_quick_test()`. numpy is **re-seeded immediately before `np.random.choice`** so the selected indices remain identical to historical runs regardless of intervening draws.

**Why this way.** `torch.use_deterministic_algorithms` was considered and rejected — it guarantees bitwise identity across kernel choices at a real speed cost; per-process reproducibility on a fixed stack is the right trade for this study. The re-seed-before-choice detail preserves backward compatibility of *which rows* a given seed selects.

---

## P1 — Correctness & robustness

### Fix #7 — Orphaned SE helper crashed with `TypeError` if ever called

**The issue.** `semantic_entropy.py::compute_semantic_entropy_for_sample` passed `reasonings=` to `compute_semantic_entropy`, whose signature has no such parameter — guaranteed `TypeError` on first call. Zero callers existed anywhere (verified incl. notebooks).

**The fix.** Removed the broken function (tombstone comment points to git history). `cluster_by_reasoning` — correct but unused — was kept with an explicit "currently unused" note and a warning about the kwarg trap.

**Why this way.** Dead code that *cannot work* has negative value (it's a trap for the next person). Keeping the small, correct `cluster_by_reasoning` preserves a documented alternative clustering mode at negligible cost.

---

### Fix #8 — Cross-family decoding-policy confound was invisible in the data

**The issue.** Anti-loop guards (`no_repeat_ngram_size=3`, `stop_strings`) apply only to base/GPT-OSS generations. The scores are correctly re-derived clean, but the **decoded text itself** was produced under different constraints per family — so base-vs-instruct confidence comparisons compare model×decoding-policy, not model. This was documented only in a docstring.

**The fix.** `generate_with_logits` stamps `meta["decoding_guards_active"]`; `evaluation.py` surfaces it as a per-row column with a comment telling analysts to condition cross-family comparisons on it.

**Why this way.** The guards are *necessary* (they stop non-termination), so the confound can't be "fixed" — the correct move is making it a first-class, filterable variable in the data instead of tribal knowledge. An on-GPU ablation (guards on/off over non-looping rows) remains recommended and is now easy to run via `SPECIFIC_INDICES`.

---

### Fix #9 — TriviaQA substring matching produced false positives

**The issue.** Tiers 1/2/3a of `check_triviaqa_correct` accepted raw bidirectional containment with **no length floor and no word boundaries**: `"no" ⊂ "north carolina"`, `"art" ⊂ "descartes"`, `"ring" ⊂ "bringiton"` (tier 3b compact) all scored as correct.

**Why it mattered.** Silent false positives inflate TriviaQA accuracy and contaminate every metric conditioned on `is_correct`.

**The fix.** `data_utils.py` → new `_contains_match` comparator used by tiers 1/2/3a:
- exact equality always matches;
- containment requires the shorter string be ≥ 3 chars, **not a stopword** (blocklist: the/and/no/in/of/…), and appear **word-boundary anchored** in the longer string;
- tier 3b (compact — no word boundaries can survive compaction) keeps equality at ≥ 4 chars but raises the **containment** floor to ≥ 6 chars, since the artifact cases it exists for (mangled multi-word names like `gneisenau`) produce long compacts.

Verified with a 16-case regression suite: all documented true positives still match (`"The USS Missouri"`, double-encoded `"JÃºpiter"`, split `"Gâ¯Neisenâ¯Au"`, short-but-legit `"Rio"`), all false positives now rejected.

**Why this way.** The alternative — the standard SQuAD/TriviaQA normalized-exact-match scorer — would be cleaner but changes scoring semantics wholesale and re-labels old runs for reasons unrelated to the bug. Tightening the existing containment design preserves its intentional flexibility (answer-inside-longer-phrase) while closing the degenerate cases. `min_len=3` (not 4) keeps genuinely short answers ("Rio", "Mao") matching; the stopword list covers the 2–3-char hazards.

---

### Fix #10 — Answer-token entropy discarded its own reliability signal

**The issue.** `extract_answer_token_entropy` renormalizes over the answer-letter simplex and computes entropy there — by construction this **understates** uncertainty whenever the model put real probability on non-letter continuations. The pre-renormalization letter mass (`total`) — exactly the quantity that says how trustworthy the renormalized entropy is — was computed and thrown away.

**The fix.** Persisted as `answer_letter_mass` in the return dict and as a result column, with a comment: low mass (e.g. < 0.5) ⇒ that row's entropy is unreliable; filter or down-weight on it.

**Why this way.** Changing the entropy definition itself (full-vocab entropy) would alter an established metric mid-study. Keeping the metric and shipping its reliability weight lets analysis decide, row by row.

---

## P1 — Performance & infrastructure

### Fix #11 — No batching: serial batch-size-1 generation throughout

**The issue.** Every sample ran up to 5 sequential `model.generate` calls at batch size 1. GPU decoding at batch 1 is memory-bandwidth-bound; utilization was a fraction of capacity, full test sets (MedQA 1,273 / GSM8K 1,319 rows) were 10–50× slower than necessary — and this pressure is visibly what forced `SE_NUM_SAMPLES=1`.

**The fix (opt-in, additive — serial path untouched and remains default):**
- `confidence.py` → `generate_with_logits_batched(prompts, batch_size)`: left-padded batched greedy decode returning a list of the exact same 5-tuples as the serial function. Handles the two subtle correctness traps: (a) left padding aligns all samples' generated tokens to a shared `input_length`, making `outputs.scores[t][i]` indexing correct; (b) right-side pad trimming retains **exactly one** trailing EOS when `pad_token_id == eos_token_id`, so `finish_reason` classification stays correct.
- **Guarded families (base variants, GPT-OSS) automatically fall back to the serial path** — their clean teacher-forced re-scoring under left padding would need `position_ids` bookkeeping where a silent mistake corrupts the exact probabilities the study measures.
- `evaluation.py` → `evaluate_sample(..., gen1_precomputed=...)` accepts a precomputed Gen-1 result; `get_question_and_choices()` factored out so `main.py` builds byte-identical prompts.
- `config.py` → `GEN1_BATCH_SIZE = 1` default (`DCAK_GEN1_BATCH_SIZE` env override); `main.py` runs chunked batches when > 1, freeing each chunk's `raw_scores` before the next (bounded memory).
- `smoke_test_batched.py` (new, repo) → per-model GPU validation that batched ≡ serial (text identity, prob diffs, finish reasons) before trusting large runs.
- `tests/test_batched_mock.py` → mock-model equivalence proof for the bookkeeping: 4 suites incl. pad==eos and per-row truncation in mixed batches.

**Why this way (and not vLLM).** vLLM is the best generic throughput answer but wrong for this codebase *first*: (1) the study needs **raw pre-warp logits** (`output_scores` + teacher-forced re-scoring); vLLM's logprobs pass through its processor stack and are top-k truncated, breaking `extract_answer_token_entropy`; (2) the carefully derived per-family guard policy has no direct vLLM equivalent; (3) switching engines mid-study makes old-vs-new comparisons engine-confounded — the same class of problem as fix #8. Staged HF batching gets ~5–10× with zero semantic drift; vLLM remains the right *second* step for full-test-set sweeps, done as a deliberate migration with a calibration check.

---

### Fix #12 — Guarded-path re-scoring pinned multi-GB logits on GPU

**The issue.** The clean forward pass ran `model(outputs.sequences).logits[0]`, materializing logits for **every** position (prompt included) over a ~150k vocab. Worse, the per-position rows appended to `raw_scores` were **views**, pinning the entire `(seq_len × vocab)` storage alive for the row's lifetime; everything stayed on GPU. For GPT-OSS at 8k budgets: multi-GB per sample.

**The fix.** `confidence.py` → `_clean_generated_logits()`:
1. requests only the last `num_gen + 1` positions via `num_logits_to_keep` (falls back to `logits_to_keep`, then full forward, and **shape-checks** the result because some models silently ignore unknown kwargs);
2. `.clone()`s the generated-region slice before the full tensor goes out of scope (breaking the view→storage pin);
3. moves the result to **CPU** — the only downstream consumer (answer-token entropy) is device-agnostic single-row reads.

**Why this way.** The version-fallback chain keeps the code working across the transformers versions this project straddles; the shape check turns "kwarg silently ignored" from a silent wrong-slice bug into a handled case. Verified by `tests/test_clean_logits.py`: all three branches produce identical values, and the guarded end-to-end path is **bit-identical** to unguarded on a loop-free generation — proving in code the "guards are inert on clean rows" invariant the docstrings claimed.

---

### Fix #13 — All-or-nothing saves + CSV Unicode mangling

**The issue.** `save_results` ran only after the whole loop — a crash at row 149/150 lost everything. Separately, round-tripping raw LLM text through CSV produced the mojibake (`â¯`, double-encoded UTF-8) that an entire repair subsystem (`_trivia_norm_*`, `reextract.py`) exists to undo.

**The fix.** `save_utils.py` → `IncrementalJSONLWriter`: writes each result as one JSON line, flushed immediately, `ensure_ascii=False`, to `{dataset}_confidence_rows_{model}.jsonl`. `main.py` writes every row in both serial and batched paths. Final JSON/CSV artifacts unchanged.

**Why this way.** Verified by crash simulation (rows readable before `close()`) and exact Unicode round-trip of the documented artifact strings. JSONL over Parquet: append-safe (Parquet isn't), human-inspectable, zero new dependencies. The CSV stays as the analysis-friendly export; the JSONL is the source of truth for re-analysis.

---

### Fix #14 — Duplicate smoke-test compute + device never passed

**The issue.** `main.py` fully evaluated `test_idx` as a smoke test, then evaluated it **again** in the loop whenever it was sampled — guaranteed double compute when `SPECIFIC_INDICES` is set (the smoke test uses its first element). Separately, `get_device()`'s result was never passed to `load_model_and_tokenizer`, which hardcoded `"cuda:0"` — crashing CPU-only debug sessions.

**The fix.** Smoke result cached in `smoke_cache` and reused in both loop paths. `load_model_and_tokenizer(model_device=None)` auto-resolves (cuda:0 → cpu fallback), normalizes `"cuda"` → `"cuda:0"`, and `main()` passes the detected device.

---

### Fix #15 — `trust_remote_code=True` everywhere, unpinned

**The issue.** Every `from_pretrained`/`load_dataset` executed with `trust_remote_code=True` and no `revision=` — arbitrary code from whatever the repo's latest commit happened to be, plus a reproducibility hole (the "same" model can change under you mid-study).

**The fix.**
- `config.py` → `MODEL_REVISIONS` (12 models), `DATASET_REVISIONS` (6 datasets), `NLI_MODEL_REVISION`, `PRM_MODEL_REVISION` — **real commit SHAs fetched live from the HF Hub on 2026-07-06** (gemma-4 entries are `None`: gated, pin once you have access).
- `model_utils.py` → both loads pass `revision=`; `trust_remote_code` restricted to the exotic families (`qwen3`, `gemma4`, `gptoss`, `llama4scout`) — Qwen2.5 / Llama-3.1 / Gemma-2 are natively supported and get `False`.
- `data_utils.py` → every dataset loader passes its pinned revision (TriviaQA keeps `trust_remote_code` for old `datasets` versions that run the loading script — the pin means that script can't change).
- `semantic_entropy.py` / `prm_scoring.py` / `confidence.py` `__main__` block → pinned; the PRM keeps `trust_remote_code` (genuinely custom head) behind its pin.

**Why this way.** Pin + minimal-trust is strictly safer than either extreme: dropping the flag everywhere breaks models that need it on older transformers; keeping it everywhere unpinned is open supply-chain exposure. With a pinned SHA, even a later upstream compromise cannot execute new code.

---

### Fix #16 — Config editable only by editing source (blocked sweeps)

**The issue.** Every knob was a module-level constant; running a different dataset/model/seed meant editing `config.py`. Derived values (`USE_REASONING_FLOW`, budgets) freeze at import; `verify_rubric.py` had to monkeypatch config globals just to test.

**The fix.** `config.py` → every experiment knob reads a `DCAK_*` environment variable with the in-file value as default: `DCAK_MODEL_FAMILY`, `DCAK_MODEL_VARIANT`, `DCAK_DATASET`, `DCAK_LEGALBENCH_TASK`, `DCAK_N_SAMPLES`, `DCAK_SEED`, `DCAK_SPECIFIC_INDICES` (comma list), `DCAK_SE_NUM_SAMPLES`, `DCAK_SKIP_NLI_CLUSTERING`, `DCAK_COMPUTE_SEMANTIC_ENTROPY`, `DCAK_GEN1_BATCH_SIZE`. Overrides apply **before** the derived values compute, so everything stays consistent — verified: `DCAK_MODEL_FAMILY=qwen3` correctly flips `USE_REASONING_FLOW=True` and the 8192 budgets.

```bash
DCAK_DATASET=gsm8k DCAK_MODEL_FAMILY=llama python3 main.py
DCAK_N_SAMPLES=150 DCAK_SEED=7 python3 main.py
DCAK_SPECIFIC_INDICES=258,301 python3 main.py
```

**Why this way (dataclass refactor deliberately deferred).** The full frozen-dataclass refactor's headline benefit — multiple configs in one process — has no use case here: every config change swaps a multi-GB model, so runs are one-config-per-process anyway. What sweeps actually need is *launching* with different configs without source edits, which env overrides deliver at ~5% of the refactor's risk. The invasive refactor remains a documented future option.

---

## P2 — Maintainability & hygiene

### Fix #17 — Dangerous duplication of prompts, constants, and helpers

**The issue.** The 10-class confidence rubric was inlined **three times** (first-pass `_CONF_RUBRIC`, Gen-2 prompt, Gen-3 critique); `_truncate_to_first_block` was copy-pasted verbatim in `confidence.py` and `data_utils.py`; the harmony delimiter `"assistantfinal"` was defined in **three files**; the `<think>` regex in two. A fix landing in one copy but not another silently desynchronizes prompt text from extractor behavior — `verify_rubric.py` even had a test asserting the rubric appeared "≥ 3 times in source", a test that only existed because of the duplication.

**The fix — with a hard byte-stability constraint.** Prompts are part of the experimental setup: any wording/whitespace drift changes model behavior and breaks comparability with committed results. So the refactor was executed golden-first:
1. **`check_prompt_golden.py` written and 36 goldens captured BEFORE touching anything.**
2. New `shared.py` single-sources: `HARMONY_FINAL_DELIM`, `ANALYSIS_MARKER_RE`, `THINK_BLOCK_RE`, `strip_harmony_envelope`, `truncate_to_first_block`, `RUBRIC_BULLETS`, `CONF_RUBRIC`. `confidence.py`, `data_utils.py`, `evaluation.py` import (with their old local alias names to minimize churn).
3. Gen-2 and Gen-3 prompts extracted as **pure builder functions** (`build_gen2_prompt`, `build_two_pass_prompt`) composed from `RUBRIC_BULLETS`.
4. **Proof:** the old `confidence.py` was extracted from git HEAD and executed with capture stubs; the new builders' output is **byte-identical** to the old inline f-strings across datasets, with/without Gen-2 scores, and with >3000-char reasoning exercising the trim paths (12/12 identical). All 36 pre-existing goldens passed unchanged; the 18 new builder prompts were then pinned (54 total).
5. `verify_rubric.py`'s source-grep test replaced with rendering the actual builders — a strictly stronger test.

**Why this way — and what was deliberately NOT done.** The six per-dataset `create_prompt` branches were *not* collapsed into a template table: their strings differ in small deliberate ways, and templating risks byte drift for a purely aesthetic win. The golden file now pins them, making a future table refactor safe to attempt; the duplication that was actually *dangerous* (rubric, delimiter, shared helpers) is gone.

---

### Fix #18 — Extraction stack had no regression protection

**The issue.** The regex extractors (Priorities 1/1.5/2/3, prose-opener filters, length caps) grew a special case per model; each patch was locally justified but the aggregate was unfalsifiable — nobody could say what a change did to other models' outputs.

**The fix.** `check_extraction_golden.py` (new) → samples real `full_response` texts from every result CSV in the repo (root, per-dataset dirs, `Trash/` — old files exercise old models' failure modes, which is exactly the value) and pins the outputs of all five extractors per response. **1,410 cases pinned**, keyed by file+row+content-hash so shifted files can't cause false alarms. `--update` re-pins after intentional changes.

**Why this way (and not constrained decoding).** Constrained/guided decoding (Outlines, vLLM guided JSON) would make answers parseable by construction, but it's a new dependency that alters generation semantics — the same class of measurement risk as the vLLM migration (fix #11). Golden regression protection makes the existing extractors safely evolvable now; constrained decoding remains the right long-term direction, adopted deliberately.

---

### Fix #19 — `prm_scoring.py` was an un-runnable-twice script

**The issue.** Loaded a 7B model **at import time**, hardcoded an input CSV not present in the repo, no `main()` guard, no CLI, and ended with a literal TODO ("add an AUROC and Standard Deviation"). Also `tokenizer.encode("<extra_0>")[0]` silently breaks if the tokenizer prepends BOS or splits the marker.

**The fix.** Rewritten as a proper CLI (`python3 prm_scoring.py input.csv -o out.csv`): `main()` guard, argparse, `convert_tokens_to_ids` for the separator, per-row failure counting, PRM revision pinned (fix #15), and a summary with mean/std and a dependency-free rank-based AUROC — fulfilling the TODO. Scoring logic itself unchanged.

---

### Fix #20 — Repo hygiene and self-defeating global settings

**The issues and fixes:**

| Issue | Fix |
|---|---|
| `.ipynb_checkpoints/` committed everywhere — including a 525-line stale `confidence-checkpoint.py` diverged from the live 1,449-line file, and compiled `.pyc` checkpoints | All removed from index and tree (114+ files staged as deletions) |
| Tracked `__pycache__/*.pyc` (were being *modified* by test runs) | Untracked and removed |
| 0-byte `=3.1.0` / `=4.40.0` (shell redirect from unquoted `pip install pkg>=…`), `.DS_Store` | Removed |
| No `.gitignore` | Added at repo root: checkpoints, `__pycache__`, `.DS_Store`, `*.bak`, `=*` |
| `warnings.filterwarnings('ignore')` as line 1 of `main.py` muted **the pipeline's own diagnostic** — the `RuntimeWarning` raised when the emitted answer letter disagrees with the letter-probability argmax (a tokenizer-mapping bug signal) | Narrowed to `FutureWarning` / `DeprecationWarning` / `UserWarning` only |
| `plt.show()` in a batch script, figures never closed (blocks headless, leaks memory across runs) | `show()` wrapped in try/except, `plt.close(fig)` always |
| No dependency spec | `requirements.txt` with bounded specs + instruction to pin exact versions via `pip freeze` on the GPU box (generate() behavior varies across transformers versions) |
| `get_correct_separate_base` documented dead code | Kept per its documented rationale (returns None by design for Llama-base) |
| `Trash/` (~6 MB superseded CSVs) | **Deliberately left untouched** — user data; flagged for the owner to delete |

---

## Final audit pass

A full second review of every changed file found and fixed four residual issues:

1. **Tracked `__pycache__/*.pyc`** — removed from the index (fix #20 addendum).
2. **`nan/10` printout** — `main.py`'s smoke-test print showed "nan/10" for extraction-failed rows because NaN passes `is not None`; changed to `pd.notna()`.
3. **Last unpinned load** — `confidence.py`'s `__main__` smoke block still used `trust_remote_code=True` unpinned; now pinned via `MODEL_REVISIONS`, flag dropped.
4. **Dead imports & stale comment** — removed `extract_reasoning` / `check_triviaqa_correct` / `MODEL_FAMILY` / `Optional` (evaluation), `MODEL_VARIANT` (visualization), `defaultdict` / `warnings` / `Optional` (semantic_entropy), `# Update import` (save_utils).

Plus one new test covering the only untested surface: `tests/test_evaluate_sample_e2e.py` — full `evaluate_sample` assembly with mocked generation (merge logic, source tracking, NaN policy, `gen1_precomputed`, mean-vs-sum columns): 20/20 assertions.

---

## New files created

| File | Purpose |
|------|---------|
| `shared.py` | Single source for rubric, harmony delimiter, `<think>` regex, extraction helpers |
| `check_prompt_golden.py` + `prompt_golden.json` | Byte-pins all 54 prompts; `--update` to re-pin after intentional changes |
| `check_extraction_golden.py` + `extraction_golden.json` | Pins 5 extractors over 1,410 real captured responses |
| `smoke_test_batched.py` | Per-model GPU validation: batched ≡ serial before enabling `GEN1_BATCH_SIZE > 1` |
| `tests/test_batched_mock.py` | Mock-model proof of batching bookkeeping (padding, pad==eos, truncation) |
| `tests/test_clean_logits.py` | Guarded-path clean re-scoring: 3 version branches + guarded ≡ unguarded e2e |
| `tests/test_evaluate_sample_e2e.py` | Full result-dict assembly: merge, sources, NaN policy (20 assertions) |
| `tests/README.md` | Pre-commit test routine + golden re-pinning workflow |
| `requirements.txt` | Dependency spec (pin exact versions on the GPU box) |
| `.gitignore` (repo root) | Blocks checkpoints, `__pycache__`, `.DS_Store`, `*.bak`, `=*` |
| `SESSION_HANDOFF_2026-07-07.md` / `.pdf` | Short-form handoff of this session |

---

## Verification matrix

All runnable without a GPU (heavy deps stubbed; torch needed only for the tensor-logic tests):

| Check | Scope | Result |
|-------|-------|--------|
| `python3 -m py_compile *.py` | all 16 modules | OK |
| `verify_rubric.py` | prompts, rubric, extractors, forced-answer paths, result columns | ALL CHECKS PASSED |
| `check_prompt_golden.py` | 54 prompts | byte-identical |
| `check_extraction_golden.py` | 1,410 real responses × 5 extractors | no drift |
| `tests/test_batched_mock.py` | batched ≡ serial (pad!=eos, pad==eos, truncated, mixed) | ALL PASS |
| `tests/test_clean_logits.py` | 3 fallback branches; guarded ≡ unguarded scores | ALL PASS |
| `tests/test_evaluate_sample_e2e.py` | merge/sources/NaN/metrics assembly | 20/20 |
| Builder byte-equivalence vs git HEAD | 12 prompt comparisons | ALL IDENTICAL |
| Env-override propagation | `DCAK_MODEL_FAMILY=qwen3` → derived flags | OK |
| `reextract.py` import compatibility | against changed `data_utils` | OK |

---

## Next steps

1. **Review & commit** — `git status`: ~114 staged junk deletions, 11 modified modules, ~11 new files. Nothing has been committed.
2. **On the GPU box** — `python3 smoke_test_batched.py` once per model family before setting `DCAK_GEN1_BATCH_SIZE > 1`; `pip freeze` the exact versions into `requirements.txt`.
3. **Re-analysis** — recompute any table built on `seq_confidence_mean` AUROC, ECE, or TriviaQA accuracy; remember old CSVs' `seq_confidence_mean` holds the **sum**.
4. **For real SE runs** — `DCAK_SKIP_NLI_CLUSTERING=false DCAK_SE_NUM_SAMPLES=5` (or edit config); the import-time notice confirms `ACTIVE THIS RUN: True`.
5. **`Trash/`** — delete when ready; it was intentionally left for you.
6. **Future options documented but deferred** — vLLM migration (with calibration check), constrained decoding for extraction, full config dataclass, prompt-template table (now safe to attempt under the golden guard).
