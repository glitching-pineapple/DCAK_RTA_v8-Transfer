# Finer-Grain Analyses: Row-Level Results and What They Mean

**Date:** 2026-08-02. Computed from raw row-level CSVs (no reruns). Analyses A–D use all 6 models
(standard models from rerun files, thinking models from cons55 combined files — mixed slice, so treat
cell counts as provisional until the canonical slice is pinned). Analyses E–G use the 3 rerun standard
models, which share sampled questions per dataset (same seeds), enabling item-level cross-model joins.

The point of this document: the cell-level metrics (AUROC/ECE per model×dataset) describe *whether*
signals work. The analyses below describe *why*, *how*, and *what that means* — which is where the
thesis claims about metacognition and transferability actually live.

---

## A. The blinded critique's binary verdict is a noisy alarm, not a reliable detector

**What was measured.** Rows where the owned self-assessment said "correct" but the blinded critique
flipped to "incorrect" (399 flips pooled across 24 cells). Of those flagged answers, 39% were actually
wrong, against a 21% base wrong rate — a 1.8× lift. Reverse flips (owner doubted, blinded critic
rescued) were rarer (81) but more precise: 75% of rescued answers were actually correct.

**What it means.** The blinded critic is *informative but trigger-happy*: a flag roughly doubles the
probability of error, but most flags are still false alarms. Meanwhile its endorsements are
trustworthy — the critic is better at confirming good work than catching bad work. Model heterogeneity
is extreme: Llama3.1 produces nearly half of all flips at barely-above-base precision (its critic is
noise), while Gemma4-31B flips almost nothing (its critique stage is inert — it rubber-stamps).

**So what for the paper.** This sharpens the "error filter" claim: the filtering value of the critique
does **not** come through the binary verdict channel. It comes through the graded channel (see B). If
a practitioner wants to use blinded critique, they should use the 1–10 rating shift, not the Yes/No flip.

## B. The confidence drop under blinding is *selective*, not uniform shyness — this is the evidence the critique carries information

**What was measured.** The rating drop (owned minus blinded) separately on wrong vs. correct answers.
Pooled: confidence drops **0.74 points on wrong answers vs. 0.35 on correct ones** — the critique
removes twice as much confidence from errors as from good answers. Selectivity is positive in 19/24
cells (the same 19/24 as the headline wrong-answer claim, now with its mechanism).

**What it means.** If blinding merely made models timid, the drop would be equal on both — it isn't.
The blinded reviewer is preferentially withdrawing confidence from answers that are actually wrong,
i.e., it *sees something* in bad reasoning that the owner-framed pass did not report. That is the
paper's cleanest evidence of authorship bias suppressing error signals: the information existed, and
removing ownership released it.

**But — and this is a finding, not a footnote — the capability is model-specific.** Qwen3 on TriviaQA
is near-perfectly selective (−2.20 on wrong, −0.03 on correct). Llama drops ~1.3–1.7 points on
*everything* (pessimism with mild selectivity). Gemma4's selectivity is negative in 3 of 4 datasets
(its critique is uninformative or backwards), and Gemma2 on LegalBench becomes *more* confident when
blinded. Selective self-critique appears to be an emergent, family-dependent skill — not a property
of the procedure alone.

## C. Even after blinded critique, half of all wrong answers still carry 9–10/10 confidence

**What was measured.** The "confidently-wrong rate": P(rating ≥ 9 | answer is wrong). Pooled, blinding
lowers it from 0.57 (owned) to 0.49 (blinded).

**What it means.** This is the deployment-facing translation of "critique doesn't fix calibration."
A user who trusts high-confidence answers still silently absorbs an error on roughly half of all
mistakes, even after the critique pass. Related: 66–73% of *all* rows sit in the top two rubric bands
— models use a 10-point scale as a ~3-point scale, which is *why* discrimination can improve while
ECE doesn't: reshuffling inside a compressed range changes ranking but not the stated probabilities.

**So what.** Use the confidently-wrong rate as a headline deployment metric in the results — it is far
more interpretable to readers than ECE deltas, and it makes the "sorts, doesn't estimate" thesis
concrete: sorting improved (A/B), the silent-failure rate barely moved.

## D. A confidence threshold does not mean the same thing on two tasks — magnitude non-transfer, operationalized

**What was measured.** The deployment rule "trust any answer rated ≥ 9" per model per dataset: the
error rate *inside the trusted bucket* ranges from 2–12% on GSM8K to 10–36% on LegalBench/StrategyQA
for the very same models (e.g., Qwen2.5: 7% on GSM8K → 36% on LegalBench).

**What it means.** "9/10 confident" is a different empirical claim depending on the task: an
acceptable risk policy tuned on one domain becomes a 1-in-3 error policy on another with no visible
warning. This is the concrete, one-table demonstration of the thesis sentence "magnitudes don't
transfer; thresholds must be recalibrated per task." Recommend this as a main-text table or heatmap.

## E. Confidence signals mostly measure question difficulty, not self-knowledge — the deepest reframe available in the data

**What was measured.** Because the three rerun models share sampled questions, one model's confidence
can be tested against *another model's* correctness. Self-AUROC vs. cross-AUROC: GSM8K 0.632 vs 0.594;
StrategyQA 0.604 vs 0.610 (cross ties self); TriviaQA 0.732 vs 0.692; LegalBench 0.493 vs 0.538
(cross *beats* self). The self-advantage is ≤ 0.04 everywhere and negative twice. Consistently, errors
are strongly correlated across models: all three models miss the same item 4–28× more often than
independence predicts.

**What it means.** Almost all of the predictive power in verbalized confidence is a *shared
item-difficulty signal* — "this question is hard" — which any model can read off the question, rather
than privileged introspective access to the model's own internal state. Genuine self-knowledge would
show self ≫ cross; it doesn't. This reframes what these signals are: not metacognition about "do *I*
know this," but perception of "is *this* hard."

**So what.** (a) It explains the thesis mechanistically: signal quality is task-bound because the
thing being measured — difficulty structure — is a property of the dataset, not the model. (b) It
sets up the blinded-critique result as the counterpoint: the *selectivity* in B is the one place the
data shows information beyond shared difficulty. (c) It is a novel, quotable finding ("cross-model
confidence transfers almost losslessly") that none of the current draft sections state.

## F. When the two signal families disagree, the verbalized rating usually wins — and agreement is itself a signal

**What was measured.** Median-split quadrants (logit geom × blinded rating), pooled per dataset over
the rerun trio. On tasks where signals work, agreement stratifies strongly (TriviaQA: both-high 84%
correct vs both-low 55%), and in disagreement cells the verbalized side is the better bet (GSM8K:
verb-high/logit-low 95% vs logit-high/verb-low 82%). On LegalBench all four quadrants sit at 66–74% —
flat: no combination of signals extracts anything.

**What it means.** The two families are partially complementary (disagreement is common and
resolvable), so a trivial ensemble — require both signals high — buys a cleaner trusted bucket at some
coverage cost, for free. And LegalBench's flat quadrants strengthen conclusion #1 from the summary
doc: it isn't that the wrong signal was chosen there; there is no signal to choose.

## G. Answer length alone is a competitive confidence signal — and part of seq-sum's crown is inherited from it

**What was measured.** AUROC of the bare token count (shorter = more confident), rerun cells. −n_tok
alone reaches 0.70–0.76 on TriviaQA/GSM8K — on Qwen2.5 GSM8K (0.739) it *beats* both the log-prob sum
(0.685) and the geometric mean (0.518); on Llama TriviaQA it (0.757) nearly matches the sum (0.785)
while the length-normalized geom sits at chance (0.530).

**What it means.** Models *flail when unsure*: struggling produces longer generations, so length is a
behavioral uncertainty signal in its own right — free, model-agnostic, and never below 0.45. It also
means the paper cannot present the raw log-prob sum as "the internal-probability signal" without
noting that a large share of its power is length in disguise. Honest presentation: geometric mean as
the probability signal, n_tok as an explicit cheap baseline, sum as their entanglement.

---

## Is "only overall metrics" limiting? Yes — here is the boundary

Aggregate AUROC/ECE answer "does signal X work on cell Y." They cannot answer any of: does the
critique *know* something (B), is confidence introspection or difficulty perception (E), do thresholds
transfer (D), what is the user-facing failure rate (C), are signals redundant (F). Those are the
questions the thesis actually poses — and all of them were answerable from data already on disk.

**Still unexplored, in rough priority order:**
1. **Risk–coverage curves** (accuracy vs. fraction answered, per signal per cell) — the figure-ready
   generalization of D; standard in selective-prediction literature.
2. **Variance decomposition** on the 24-cell AUROC table (share of variance explained by dataset vs.
   signal vs. model) — turns the thesis sentence into a single number.
3. **Formal fusion test** — rank-average of geom + blinded rating vs. each alone, with bootstrap CIs
   (F suggests it wins; needs quantification).
4. **Critique text mining** — `two_pass_critique` free text is stored for every row; classify whether
   the critic *articulated* a specific error on true-flag vs. false-flag rows (does it find the bug or
   just vibe?). Would connect A/B to interpretable behavior.
5. **Item-level difficulty regression** — predict correctness from question length/type across all
   models jointly; quantifies how much of E's shared axis is recoverable from surface features.
6. **Cross-seed stability of the flip/selectivity effects** (B) using the seed column in rerun files —
   cheap robustness check before any of A–C goes in the paper.
