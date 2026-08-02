# Conclusions Derivable from metrics_summary (rerun slice, 150-row cells)

**Scope caveat:** this summary covers the 3 standard models only (Gemma2-9B, Llama3.1-8B, Qwen2.5-7B)
× 4 datasets, from the **rerun** (top-20) files. Numbers differ slightly from the originals slice used
for the draft's headline claims (see RESULTS_VERIFICATION.md). Everything below is computed from the
12 pooled 150-row cells; per-seed rows were used only for conclusion #11.

---

1. **LegalBench is a dead zone for every confidence signal, in every model.** Across all 14 AUROC
   columns and all 3 models, the best LegalBench value is 0.552 (Llama, seq-sum); Gemma and Qwen max
   out at 0.503 and 0.514, and many signals fall *below* 0.5 (minimums 0.383 and 0.380 — actively
   inverted). This is the strongest single piece of evidence for the paper's thesis: on some tasks
   there is currently *no* usable confidence signal, so signal choice cannot be made a priori.

2. **The direction of miscalibration is a model property, not a universal law.** Overconfidence
   (mean stated probability − accuracy): Gemma2 is overconfident on all 4 datasets (+0.002 to +0.110),
   **Llama3.1 is underconfident on all 4** (−0.079 to −0.236), Qwen2.5 is mixed. The literature's
   blanket "LLMs are overconfident" claim does not survive contact with rubric-elicited confidence —
    genuinely novel, citable nuance (contrast with Xiong et ala., 2024).

3. **Mean-level calibration can be perfect while the signal is useless — the cleanest dissociation
   example in our data.** Gemma2 on LegalBench has overconfidence of just **+0.002** (essentially
   perfectly calibrated on average) while its verbalized AUROC is **0.488** (no discrimination at
   all). Aggregate calibration and per-item usefulness are different quantities; this cell is the
   figure-ready illustration of why the paper insists on the discrimination/calibration distinction.

4. **A large share of the raw log-prob sum's discrimination comes from length, not probability.**
   Seq-sum beats the length-normalized geometric mean in 11/12 cells, and the gap is huge exactly
   where generations vary most in length: Llama TriviaQA **+0.255** (0.785 vs 0.530), Qwen GSM8K
   +0.169. Since the sum mechanically decreases with length, and longer answers are harder answers,
   part of "seq log prob is the best logit signal" is a length proxy. The paper should headline the
   geometric mean as the honest logit signal and report a bare token-count baseline as a control.

5. **Last-token confidence is an anti-signal.** The last-token geometric/top-20 columns are below
   0.5 in 8/12 cells and never exceed 0.578. The final token's certainty measures "I am sure the
   generation is complete," not "I am sure the answer is right." This justifies excluding last-token
   variants from the main table (footnote-worthy negative result).

6. **Richer internal signals buy nothing.** The top-20 entropy mean beats plain seq-sum in only 3/12
   cells (mean difference −0.055), and entropy-max and margin variants track the simple aggregates
   everywhere. A useful negative result: expensive per-position top-k logging does not improve over
   one-line log-prob aggregates, so practitioners can skip it.

7. **Content-token filtering is a no-op.** Max |geom − geom_content| across all 12 cells is 0.022.
   Restricting logit aggregation to "content" tokens changes nothing; one sentence in Methods, no
   analysis needed.

8. **The binary "more likely than not" self-verdict fails a trivial baseline almost everywhere.**
   Its F1 beats the always-say-Yes baseline (F1 = 2·acc/(1+acc)) in only **3/12 cells**. Worst case,
   Llama on LegalBench: mln F1 0.543 vs 0.855 baseline, and mln *accuracy* 0.473 — worse than
   answering "correct" every time. Binary self-verdicts as deployed add essentially no decision
   value; graded confidence is strictly the better elicitation. (Report the always-Yes baseline in
   the paper — without it, F1 ≈ 0.8–0.95 numbers look deceptively strong.)

9. **Verbalized and logit signals succeed on different cells — complementarity, not redundancy.**
   Verbalized never beats the best logit signal by more than +0.013 in this slice, and loses badly on
   GSM8K (Gemma: 0.568 vs 0.797). But on Llama TriviaQA the *reverse* failure occurs: geometric mean
   is at chance (0.530) while verbalized reaches 0.795. Neither family dominates; the disagreement
   pattern itself argues for testing signal fusion (rank-averaging the two) as future/extra analysis.

10. **GSM8K is the logit stronghold.** For all 3 models, logit aggregates (0.69–0.82) clearly beat
    verbalized confidence (0.57–0.70) on GSM8K — plausibly because arithmetic errors perturb token
    probabilities locally, while the model's verbal self-assessment stays anchored at "high." Another
    concrete instance of task identity determining signal quality.

11. **Per-seed slices are too small to interpret — pool them.** Across the 45-row seed slices, the
    verbalized-AUROC spread within the same model×dataset has median 0.126 and maximum 0.307 (e.g.,
    Qwen GSM8K: 0.485 at seed 33 vs 0.730 at seed 89). Any figure or claim built on a single seed
    slice is noise; only the pooled 150-row cells (and CIs derived from them) should appear in the
    paper. This also pre-answers a likely reviewer question about seed robustness.

12. **Low ECE mostly reflects the accuracy regime, not metacognitive skill.** The best ECE in the
    slice (Qwen GSM8K, 0.027) co-occurs with mediocre discrimination (verbalized AUROC 0.628): the
    model is 93% accurate and says ~91% confident on everything, so the averages align without the
    model knowing *which* items it missed. Corollary for the paper: never report ECE without AUROC
    alongside it.

13. **Where logit signals invert, models are *more* fluent when wrong.** The sub-0.5 logit AUROCs
    (Gemma and Qwen on LegalBench) mean higher token-probability answers are *more* likely to be
    incorrect there — consistent with a fluency trap: confidently pattern-matched legal boilerplate
    that is wrong. Hypothesis-generating; frame as discussion material, not a claim.

14. **Llama's underconfidence is a consistent fingerprint across signal types.** It shows up in the
    overconfidence column (negative on all 4 datasets), in mln recall (0.42–0.82, i.e., it says "No"
    to its own correct answers far more than Gemma/Qwen, whose recall is 0.88–1.00), and in its low
    stated confidence on wrong answers. Miscalibration *direction* being stable within a model family
    while varying across families supports RQ2's "consistency" framing — signal *rankings* vary by
    task, but a model's metacognitive bias travels with it.

---

**Files:** cleaned summary at
`~/Desktop/blobfish/reruns_consolidated_NEW_METRICS/metrics_summary_150only.csv` (12 rows, seed rows
removed; original untouched).
