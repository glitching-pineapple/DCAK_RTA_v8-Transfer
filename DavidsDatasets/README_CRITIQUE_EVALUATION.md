# Evaluation of the Code-Review Edits Against Owner Critiques

**Date:** 2026-07-31
**Context:** The owner reviewed the 20 fixes from the 2026-07 code-review session point by point.
This document records, for each critique: the answer, the evidence (measured from the **existing
result CSVs — no model was rerun for any of this**), the action taken, and — where relevant —
copy-paste recipes for post-hoc analysis and suggested limitations wording.

**Ground rules confirmed by the owner:** results are final, nothing will be rerun; active
benchmarks are **gsm8k, triviaqa, strategyqa, legalbench** (medqa and mmlupro dropped);
semantic entropy is off.

---

## 1. The sum-vs-mean confidence metric

**Owner position:** we know `seq_confidence_mean` is a sum; leave it. Can it be normalized
post-hoc without rerunning? Token counts are ~180–210 for GSM8K so it shouldn't matter much.

**Answer: yes — the normalized version already exists in every CSV you have.**
No rerun, no new computation needed:

- `logit_confidence_geom` **is** the length-normalized metric (it equals
  `exp(mean log-prob per token)`). Use it directly, or take `ln()` of it:

  ```python
  df["mean_log_prob"] = np.log(df["logit_confidence_geom"])          # length-normalized
  df["n_tokens"]      = df["seq_confidence_mean"] / df["mean_log_prob"]  # bonus: recovers token count
  ```
  (In old CSVs `seq_confidence_mean` holds the sum, so sum ÷ mean = token count.)

**Evidence — the length assumption doesn't hold, and the confound is real (measured 2026-07-31):**

| Run (150 rows each) | Implied token count (min / median / max) | Mean length, correct rows | Mean length, wrong rows | corr(length, correct) |
|---|---|---|---|---|
| Qwen2.5-7B GSM8K | 108 / 223 / 571 | 228.9 | 261.1 | **−0.101** |
| Llama3.1-8B GSM8K | 200 / 360 / 512 | 352.9 | 444.5 | **−0.436** |

Wrong answers are systematically **longer** (Llama: ~90 tokens longer on average). So the sum
metric partially re-encodes "long answer ⇒ probably wrong" — real signal about correctness, but
not *confidence* signal. For Llama especially, any AUROC computed from the sum is substantially a
length effect.

**Recommendation:** keep `seq_confidence_mean` untouched in your files (as you prefer), but use
`logit_confidence_geom` (or its log) for any confidence-vs-correctness claim. If you report the
sum anywhere, the honest framing for limitations: *"sequence log-probability conflates per-token
confidence with response length (r = −0.44 between length and correctness for Llama-3.1 on GSM8K);
we therefore rely on the length-normalized geometric-mean token probability."*

**Action taken:** none further needed — the code fix (new runs write both columns) stays; old
files need only the one-liner above at analysis time.

---

## 2. The calibration / ECE fix — which metric was actually affected

**Owner position:** "I think what you're seeing as the 0–1 metric is `more_likely_than_not`.
I don't understand what the calibration score is or which column it referss to."

**Clarification — it is NOT `more_likely_than_not`.**

- `more_likely_than_not` is the Yes/No column ("Correct: Yes"). It was never binned by anything
  and was **never affected** by this bug.
- The affected code is the function `calibration_analysis()` in `visualization.py`, which operates
  on the **`verbalized_confidence`** column — the 1–10 number extracted from the model writing
  "Confidence: 7".

**What calibration/ECE is, plainly:** take all rows where the model said "7" (which your rubric
defines as 60–70% likely correct) and check what fraction were actually correct. Do that for every
rating level. A well-calibrated model saying "7" is right ~65% of the time. **ECE (Expected
Calibration Error)** is the single-number summary: the average gap between stated confidence and
actual accuracy, weighted by how many rows sit at each level. ECE 0 = perfectly calibrated;
ECE 0.25 = on average the model's stated confidence is 25 points off from reality.

**The bug:** the function sorted the 1–10 ratings into buckets labeled 0–1 (0.2, 0.4, …). A "7"
fits in none of them, so every row was discarded and the output table was empty — every run,
silently. Your runs never produced a calibration table or an ECE.

**Post-hoc:** fully computable from your existing CSVs — `verbalized_confidence` is stored.
Run the fixed function on any results file:

```python
import pandas as pd
from visualization import calibration_analysis
df = pd.read_csv("gsm8k_confidence_Qwen2.5-7B-instruct.csv")
calibration_analysis(df, "verbalized_confidence")   # now prints buckets + ECE
```

(medqa being dropped changes nothing here — the fix is dataset-agnostic.)

---

## 3. The two-pass / single-pass mixture — separable WITHOUT rerunning?

**Owner position:** understand the difficulty confound; results are final; is there a post-hoc
separation? Otherwise it goes in limitations.

**Answer: yes — and the problem is much smaller than feared. Measured today:**

1. **Your four Llama runs (gsm8k, medqa, strategyqa, triviaqa) use the old schema** — no
   `two_pass_critique` or `single_pass_confidence` columns exist. Those runs predate the two-pass
   system entirely, so their `verbalized_confidence` is a single method throughout.
   **No mixing exists in them. No limitation needed.**
2. **The Qwen GSM8K run uses the two-pass schema**, and the raw critique text is stored per row.
   Re-parsing `two_pass_critique` with the same extractor recovers the true two-pass score:
   **147/150 rows parse, and all 147 match the merged column exactly** — i.e. those rows ARE
   two-pass scores. Only **3/150 rows (2%)** fell back to single-pass.

**Recipe (no model, seconds to run):**

```python
import pandas as pd
from confidence import extract_verbalized_confidence
df = pd.read_csv("gsm8k_confidence_Qwen2.5-7B-instruct.csv")
df["two_pass_recovered"] = df["two_pass_critique"].apply(
    lambda t: extract_verbalized_confidence(t, "gsm8k") if isinstance(t, str) else None)
df["conf_source"] = df["two_pass_recovered"].notna().map({True: "two_pass", False: "single_pass"})
# analyze df["two_pass_recovered"] and df["single_pass_confidence"] separately;
# optionally drop the 3 fallback rows (2%) for a pure two-pass analysis.
```

**How the code fix handles it going forward:** new runs store `two_pass_confidence` and
`single_pass_confidence` as separate columns plus a `verbalized_conf_source` column, and the
analysis reports each method's AUROC/calibration separately — the merge still exists for
continuity but is labeled as a merge.

**Limitations wording if you want it (only for two-pass-schema runs):** *"For runs using the
two-stage critique, the primary verbalized-confidence column substitutes the self-rated score on
rows where the critique output was unparseable (2% of rows, GSM8K/Qwen); per-method re-analysis
excluding these rows was materially unchanged."* — you can state that last clause after running
the recipe.

---

## 4. Which model–benchmark pairs produced crashed/missing rows?

**Owner position:** we compiled everything and see no missing rows — maybe crashes never happen.

**Answer: you are right — verified. None of your runs crashed.** Row-count audit of every main
results file (a crashed sample leaves a *missing* row, not a blank one):

| File | Rows |
|---|---|
| Llama3.1-8B gsm8k / medqa / strategyqa / triviaqa | 150 / 150 / 150 / 150 |
| Qwen2.5-7B gsm8k | 150 |

All complete. The silent-exception bug was a **latent** risk (it would have hidden crashes *if*
they occurred), not a realized one — your compiled results are unaffected. The fix (record +
warn + errors sidecar) stays as pure insurance for future runs. **No action, no limitation.**

---

## 5. Semantic entropy

**Owner position:** not using it — too much compute; it can stay off.

**Agreed; nothing to do.** The fix only makes the config *honest* about SE being off: the run now
prints "ACTIVE THIS RUN: False" instead of claiming it's enabled, and — a small win — it no longer
loads the DeBERTa NLI model onto the GPU for a computation that never runs (frees ~1.6 GB VRAM
and a few seconds of startup).

---

## 6. "How is random sampling different from the seed?"

The seed is not specifically a key for picking questions — it is the starting value for a
**random-number generator**, and a program can contain *several independent generators*. Each one
needs its own seed. Your pipeline has two:

1. **numpy's generator** — used once, to pick which 150 questions to evaluate. This one *was*
   seeded. Same seed → same questions. ✓
2. **PyTorch's generator** — used only when the model generates text *with randomness turned on*
   (temperature sampling: instead of always taking its top-probability next word, the model rolls
   weighted dice each word). This one was *never* seeded, so those dice differed every run.

**Why this doesn't affect your results at all:** your generation is **greedy** — the model always
takes its single most-likely next word, no dice involved. Greedy output is identical every run
regardless of any seed. The only place the pipeline rolls dice is semantic-entropy sampling,
which (per your point 5) you don't use. So fix #6 changes nothing about your compiled numbers;
it only matters if sampling is ever switched on.

---

## 7. The unused `compute_semantic_entropy_for_sample` function

**Owner position:** never called, never broke anything — leave it in; don't fix what isn't broken.

**Action taken: restored** (2026-07-31), verbatim from git history, with one added docstring note
recording the owner decision and the known issue (it passes a `reasonings=` argument its callee
doesn't accept — wire that through before first use). All tests still pass.

---

## 8. The `decoding_guards_active` column

**Owner position:** superficial; doesn't change technical workings; remove it and note it for a
later cleanup pass.

**Action taken: removed from results output** (2026-07-31) — `evaluate_sample` no longer writes
the column, so the results schema matches your compiled files. The flag still exists *internally*
(in generation metadata, invisible in outputs) because two unit tests use it to verify the
anti-loop guards don't alter clean generations.

**Deferred-cleanup note (as requested):** if base-model or GPT-OSS comparisons ever become a
headline claim, reintroduce per-row guard visibility (or run the guards-on/off ablation) so
cross-family confidence comparisons can be conditioned on decoding policy. Until then it stays
out of the data.

---

## 9. TriviaQA matcher — can we verify impact without rerunning?

**Owner position:** is there a way to check whether the loose matching actually affected results?

**Answer: yes — checked today on your stored answers (no model needed).** Old matcher vs new
matcher over all 150 rows of the Llama TriviaQA run (ground-truth-only comparison):

- **3 rows differ — and all 3 are cases the OLD matcher would have missed** (quote characters:
  `'CRIMSON TIDE'`; accents: `John le Carré` vs `John Le Carre`) that the new matcher correctly
  matches. Your stored labels were already correct for these (the original run matched via the
  alias list).
- **Zero rows where a stored "correct" would flip to "wrong."** The degenerate patterns the
  tightening guards against ("no" ⊂ "North Carolina") did not occur in this run's actual answers.

**Conclusion: no evidence the matcher change would alter any of your compiled TriviaQA numbers.**
The caveat: this check compared against ground truth only, not the full alias lists (aliases
aren't stored in the CSV). For a 100%-complete audit, run this once on a machine with the
dataset cached — still no model, no GPU:

```python
import pandas as pd
from datasets import load_dataset
from data_utils import check_triviaqa_correct
ds = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split="validation")
df = pd.read_csv("triviaqa_confidence_Llama3.1-8B-instruct.csv")
flips = [(r.idx, r.model_answer, r.is_correct,
          check_triviaqa_correct(r.model_answer, ds[int(r.idx)]))
         for r in df.itertuples() if isinstance(r.model_answer, str)]
print([f for f in flips if bool(f[2]) != f[3]])   # [] expected
```

---

## 10. What `answer_letter_mass` is — and why it's moot for you

**Owner position:** don't understand — which uncertainty metric, and what's the "can it be
trusted" thing?

**The metric:** `answer_token_entropy`. It exists **only for multiple-choice datasets** (medqa,
mmlupro). At the exact moment the model prints its answer letter, the code reads the probability
the model assigned to each letter (A: 62%, B: 20%, C: 8%…), rescales them to sum to 100%, and
computes the entropy (spread) of that distribution — peaked = confident, flat = uncertain.

**The trust issue:** the rescaling step. Suppose at that moment the model put only 10% of its
total probability on letters and 90% on other words ("Let", "The", "I"…). The entropy then gets
computed from that 10% sliver, and a sliver can look artificially peaked — the metric reports
"confident" when the model mostly wasn't choosing a letter at all. `answer_letter_mass` is simply
that percentage (0.10 in the example), saved per row, so low-mass rows can be discarded as
unreliable readings of an otherwise fine instrument.

**Why it's moot:** you dropped medqa and mmlupro. For gsm8k / triviaqa / strategyqa / legalbench
this entire metric is `None` on every row — the column and its trust score never activate.
No action needed; the fix simply sits dormant unless MCQ datasets return.

---

## Summary of dispositions

| # | Critique verdict | Disposition |
|---|---|---|
| 1 | Keep the sum; normalize post-hoc | **No rerun needed** — `logit_confidence_geom` is the normalized metric; confound confirmed real (Llama r = −0.44), use geom for claims |
| 2 | Confused with `more_likely_than_not` | Clarified — affected metric is ECE on `verbalized_confidence`; computable post-hoc on existing CSVs |
| 3 | Separate post-hoc if possible | **Done** — Llama runs never mixed (old schema); Qwen GSM8K 98% pure two-pass, recoverable by re-parsing stored critiques |
| 4 | Crashes may never have fired | **Confirmed** — all files 150/150; fix is insurance only |
| 5 | SE stays off | Agreed; fix just makes config honest + frees VRAM |
| 6 | Seed vs sampling unclear | Explained — greedy runs are dice-free; fix has zero effect on your results |
| 7 | Leave the unused function | **Restored** per owner decision, with docstring note |
| 8 | Remove superficial column | **Removed** from results output; noted for later cleanup |
| 9 | Verify matcher impact | **Verified** — 0 correct→wrong flips; 3 old-misses now correctly matched; full-alias audit script provided |
| 10 | Metric not understood | Explained — MCQ-only; moot for current benchmark set |

All test suites re-run after the reversals (#7, #8): **compile OK · verify_rubric PASS ·
54/54 prompt goldens · 1,410/1,410 extraction goldens · all mock suites PASS.**
