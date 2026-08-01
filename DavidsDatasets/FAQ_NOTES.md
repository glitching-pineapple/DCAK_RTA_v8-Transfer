# FAQ / Notes — Edge Cases, Nuances, and Apparent Inconsistencies

Reference for anyone analyzing the results files or reading the code. Each entry answers a
question that will *look* like a bug or inconsistency at first glance but is expected behavior.
Nothing here requires code changes — it's the "why is it like that" record.

Last updated: 2026-07-31. Active benchmarks: **gsm8k, triviaqa, strategyqa, legalbench**.
Semantic entropy: **off**. Results are final; no reruns planned.

---

## A. Columns & schemas

### Q: Why does `seq_confidence_mean` disagree between old and new results files?
In **old CSVs** (everything compiled before 2026-07), this column holds the **SUM** of per-token
log-probabilities despite its name. In files produced by the current code it holds the true
**mean** (and the sum lives in a new column `seq_log_prob_sum`). Do not compare this column
across old and new files directly.

- Length-normalized value for ANY file, old or new: `np.log(df["logit_confidence_geom"])` —
  the geometric-mean column always was the normalized metric.
- Token count for old files: `seq_confidence_mean / np.log(logit_confidence_geom)` (sum ÷ mean).
- Caution: the sum is length-confounded. Measured on our GSM8K runs, wrong answers are longer
  (Llama-3.1: r = −0.44 between length and correctness), so sum-based AUROC partly measures
  verbosity. Use the geometric mean for confidence claims.

### Q: Why do the Llama runs have fewer columns than the Qwen GSM8K run?
Different pipeline generations, not data loss. The four Llama-3.1 runs (gsm8k/medqa/strategyqa/
triviaqa, 150 rows each) predate the two-pass critique system — they have no
`single_pass_confidence`, `two_pass_critique`, etc. Their `verbalized_confidence` is a single
elicitation method throughout. The Qwen GSM8K run uses the newer schema with the extra columns.

### Q: In two-pass-schema runs, what exactly is in `verbalized_confidence`?
It's a merge: the two-pass (blind-critique) score when that parsed, else the single-pass
(self-rating) score. In the Qwen GSM8K run this fallback fired on only **3/150 rows (2%)**.
To un-mix without rerunning: re-parse the stored critique text —
`extract_verbalized_confidence(row.two_pass_critique, "gsm8k")` — rows where that returns a
value are pure two-pass. Files written by current code store both methods separately plus a
`verbalized_conf_source` column.

### Q: Why are `answer_token_entropy`, `answer_letter_probs`, `chosen_letter`, `prob_A…` empty/missing?
They exist **only for multiple-choice datasets** (medqa, mmlupro), which we dropped. For
gsm8k/triviaqa/strategyqa/legalbench these are `None`/absent on every row by design.
(If MCQ ever returns: `answer_letter_mass` is the fraction of probability actually on letter
tokens at the answer position — low mass means that row's entropy is an unreliable reading.)

### Q: The `question` column is cut off mid-sentence with "..." — is data missing?
Intentional: questions are truncated to 200 characters in the results for file-size sanity.
The full question is recoverable from the dataset via the `idx` column.

### Q: What is `idx` exactly?
The row number in the HuggingFace dataset split (test split; TriviaQA uses validation). The 150
evaluated rows were drawn randomly with seed 42 — same seed, same 150 rows, every run.

---

## B. Correctness grading (why a row is marked right/wrong)

### Q: TriviaQA — `is_correct=True` but `model_answer` doesn't equal `ground_truth`?
Expected. TriviaQA grading is **alias-aware**: the dataset ships many acceptable forms
("USS Missouri", "U.S.S. Missouri", …) but the CSV stores only the primary value in
`ground_truth`. A row can match an alias you can't see in the CSV. Also, matching survives
formatting noise: containment ("The USS Missouri" matches "USS Missouri"), accents/Unicode
mangling ("JÃºpiter" matches "Jupiter"), and split letters ("G Neisen Au" matches "Gneisenau").
Guards prevent junk matches (word boundaries, minimum lengths, stopword blocklist) —
verified against our stored answers: zero stored labels would change under the current matcher.

### Q: GSM8K — model answered "72.00" but ground truth is "72"; why is it correct?
Numeric comparison, not string comparison: "6.00" ≡ "6", "1,000" ≡ "1000", "$65,960" ≡ "65960",
with a tiny float tolerance for chained-arithmetic drift. Ground truth is parsed from the
dataset's `#### N` line with commas stripped.

### Q: StrategyQA / LegalBench — "no." vs "No"?
Case-insensitive with trailing punctuation stripped: "yes", "Yes", "YES.", "yes!" all equal.

### Q: `model_answer` says B, but I can't find "Answer: B" anywhere in `full_response`?
Check `was_forced`. When the main response never committed to a clean `Answer:` line (usually the
chain-of-thought hit the token budget), a short follow-up "forced answer" call asked the model to
commit; that committed answer lives in `forced_answer_response`, not `full_response`.

### Q: The response contains a later/different answer than the one extracted — why?
Two deliberate extraction rules:
1. **First-block only:** base models often answer correctly, then keep generating brand-new
   hallucinated Q&A pairs. Everything after the first completed Answer/Confidence/Correct block
   is ignored, so a "different answer" later in the text is the model answering a question nobody
   asked.
2. **Within the block, LAST match wins:** if the model self-corrects mid-solution ("Answer: 5 …
   wait … Answer: 7"), the final commitment (7) is taken.
Also, `answer:` appearing mid-sentence in reasoning ("in this answer: …") is intentionally NOT
matched — only line-anchored `Answer:` counts (plus a guarded mid-line form for TriviaQA
commit phrases like "So overall, Answer: Henry II").

### Q: A row has `is_correct=False` and `verbalized_confidence=NaN` — corrupted?
No — deliberate policy. `answer_extraction_failed=True` means no parseable answer existed (main
AND forced pass both failed). Such rows are auto-marked incorrect and ALL confidence fields are
NaN'd so a half-parsed signal can't leak into calibration. Exclude these rows from calibration
analysis; `is_refusal=True` further marks the subset that read as genuine abstentions
("I don't have access to…") rather than truncations.

---

## C. Confidence numbers

### Q: What scale is `verbalized_confidence` on, and what does a "7" mean?
Integer 1–10 from the model writing "Confidence: 7". The prompt rubric defines class N as
"(N−1)·10% to N·10% likely correct" — so 7 means "60–70%", and the faithful probability for
calibration is the interval midpoint **(N − 0.5)/10 = 0.65**, not 0.70.

### Q: The model wrote "Confidence: 85%" or "0.85" — what got stored?
Auto-normalized to the 1–10 scale: values > 10 are divided by 10 (85% → 8.5 → rounds to 9? no —
85/10 = 8.5 → round → 8~9 per Python banker's rounding; stored after `round()` and clamped to
[1, 10]). Decimals ≤ 1.0 are multiplied by 10 (0.85 → 8.5 → round). Rare legacy formats only;
the prompt asks for a bare integer.

### Q: Is `more_likely_than_not` the same thing as calibration?
No. `more_likely_than_not` is a separate binary self-judgment ("Correct: Yes/No" ≈ "am I >50%
likely right?"). Calibration/ECE is computed from `verbalized_confidence` (the 1–10 rating).
Note: **no run before 2026-07 ever produced a calibration table or ECE** — the old analysis code
binned 1–10 values into 0–1 buckets, discarding every row. ECE for compiled results is being
added in post-processing from the stored `verbalized_confidence` column (owner task).

### Q: `single_pass_correct` is None on many base-model rows?
By design. The logit-comparison approach for base models (`get_correct_separate_base`) returns
None unconditionally — Llama-3.1-base's pretraining prior answers "Yes" regardless of answer
quality, so the reading is meaningless. The pipeline falls through to reading the model's own
"Correct: Yes/No" line instead; rows where that line never appeared stay None honestly.

### Q: `two_pass_finish_reason == "skipped"`?
Llama-base only: its compact critique format reliably produced unextractable output, so the
two-pass call is skipped instead of burning compute on a dead end.

### Q: `finish_reason` values — what's the difference?
`eos` = model finished naturally · `stop` = a stop-string fired (base-model over-generation
guard) · `length` = ran out of token budget (`was_truncated=True`). An EMPTY response with
`eos` is a base model bailing instantly on an out-of-distribution prompt — counted as
non-truncated on purpose (it didn't run out of room; it chose to stop).

---

## D. Reproducibility & generation

### Q: Are runs reproducible?
Greedy decoding (all our runs) is deterministic: same model + same prompt + same hardware/stack
→ same output, seed-irrelevant. The seed fixes WHICH 150 questions are drawn (numpy, seed 42 —
re-seeded immediately before selection so historical index sets are preserved). Sampling paths
(only used by semantic entropy, which is OFF) draw from torch's generator — seeded in current
code, unseeded historically; irrelevant while SE stays off.

### Q: Do the anti-repetition guards change our numbers?
Not for our compiled results: guards (`no_repeat_ngram_size=3`, stop strings) apply only to
**base**-variant models and GPT-OSS. All our compiled runs are instruct Qwen/Llama — guards off,
generations untouched. (A per-row guard flag existed briefly and was removed from outputs by
owner decision 2026-07-31; the internal mechanism remains for base/GPT-OSS runs.)

### Q: Is batching used? Does it change outputs?
Default is OFF (`GEN1_BATCH_SIZE=1`) — behavior identical to all historical runs. If enabled
(env var `DCAK_GEN1_BATCH_SIZE`), batched greedy decoding is designed to be output-identical and
must be validated once per model with `smoke_test_batched.py` first.

### Q: The config file says X but the run log shows Y?
Every knob can be overridden per-run by environment variables (`DCAK_DATASET`, `DCAK_SEED`,
`DCAK_N_SAMPLES`, …) without editing the file. **Trust the printed CONFIGURATION banner in the
run log**, not the file, when reconstructing what a run did.

### Q: Are model downloads stable?
Yes — every model/dataset load is pinned to an exact HuggingFace commit SHA
(`config.MODEL_REVISIONS` / `DATASET_REVISIONS`, captured 2026-07-06). "Latest" upstream changes
cannot silently alter our models. (gemma-4 entries are unpinned — gated repo; pin if ever used.)

---

## E. Files & formats

### Q: Which file is the source of truth for a run?
Current-code runs write three: `*_confidence_rows_*.jsonl` (per-row, crash-safe, exact Unicode —
**preferred for any text-field analysis**), `*_confidence_detailed_*.json` (full record),
`*_confidence_*.csv` (analysis convenience). Old runs have JSON+CSV only.

### Q: Old CSVs contain garbage characters like `â¯` or `JÃºpiter`?
CSV round-trip mojibake from the era before the JSONL log (narrow-no-break-space and
double-encoded UTF-8). The TriviaQA matcher deliberately normalizes through this, so grading is
unaffected; just don't eyeball-compare those strings. New runs' JSONL is clean.

### Q: What is `*_errors_*.json`?
Written only if samples crash during a run (index, error, traceback). Audit 2026-07-31: **all
compiled runs are complete at 150/150 rows — no run ever crashed**, so no such files exist for
them.

### Q: What are `prompt_golden.json` / `extraction_golden.json` / `tests/`?
Tripwires, not data. They freeze (a) all prompt strings byte-for-byte and (b) extractor behavior
on 1,410 real stored responses. If ANY future edit changes either, `check_prompt_golden.py` /
`check_extraction_golden.py` fail and show exactly what moved. After an *intentional* change,
re-pin with `--update`. Run both (plus `verify_rubric.py` and `tests/`) before committing
pipeline changes — no GPU needed.

### Q: `reextract.py`, `.bak` files?
A one-time repair tool from a historical extractor bug (an empty `Answer:` line could capture the
next line's "Confidence: 3" as the answer). It rewrites only rows matching that exact signature
and leaves a `.bak` of the original. Not part of normal operation.

---

## F. Known-and-accepted items (deliberately not "fixed")

| Item | Status |
|---|---|
| Old CSVs' `seq_confidence_mean` = sum | Kept as-is (owner decision); use `log(logit_confidence_geom)` for normalized analysis |
| ECE for compiled results | To be added in post-processing from stored `verbalized_confidence` (owner task) |
| `compute_semantic_entropy_for_sample` in `semantic_entropy.py` | Unused; kept per owner decision; would need a small wiring fix (`reasonings=` param) before first use |
| Two-pass fallback rows in Qwen GSM8K | 3/150 rows; identifiable by unparseable `two_pass_critique`; exclude or footnote |
| `Trash/`, legacy result folders | Historical data, intentionally untouched |
| Guard-flag column in outputs | Removed by owner decision; revisit only if base-vs-instruct comparisons become a headline claim |
