# AI Main-Track Conference Paper Blueprint (8–10 Pages)

**Target venues:** NeurIPS / ICML / ICLR main track (also suitable for ACL/EMNLP main with minor reframing)
**Working title pattern:** *"Do Models Know When They're Wrong? A Multi-Signal, Cross-Model Study of LLM Confidence Calibration with Blinded Self-Critique"*
**Page budget:** 8–9 pages of main text + unlimited references + appendix. Camera-ready typically grants +1 page.

This blueprint is calibrated to the LLM confidence-telemetry study: 6 instruct models (Qwen2.5-7B, Qwen3-30B-A3B, Llama-3.1-8B, Gemma-2-9B, Gemma-4-31B, GPT-OSS-20B), 4 benchmarks (GSM8K, StrategyQA, TriviaQA, LegalBench), 7 confidence-signal families, and the three-generation elicitation architecture. Adapt the concrete numbers to your final runs; keep the structure.

---

## 0. Pre-Writing Decisions (Lock These Before Drafting)

- **The one-sentence claim.** Every main-track paper needs a falsifiable headline claim. Candidate: *"Blinded self-critique (Gen 3) produces systematically better-calibrated verbalized confidence than own-work self-rating (Gen 2), and the gap is largest for thinking models."* Write it on an index card; every section must serve it.
- **Contribution triage.** Main-track reviewers want 2–3 crisp contributions, not 4 diffuse ones. Recommended triage from the handoff's four contributions:
  1. **Method:** the three-generation elicitation architecture (reasoning → own-work self-rating → blinded critique) that decouples reasoning from self-evaluation and neutralizes sunk-cost inflation.
  2. **Empirical:** a 6-model × 4-task × 7-signal calibration matrix (ECE, AUROC, reliability diagrams) covering thinking and standard models.
  3. **Analysis:** signal-complementarity findings — where logit, verbalized, and semantic-entropy signals agree, diverge, and which to trust per task type.
  - Demote the engineering observations (harmony-envelope parsing, repetition-loop guards, forced-answer recovery) to a subsection of Experimental Setup + Appendix. They are *credibility material*, not contributions.
- **Statistics gate.** Main track requires: multiple seeds per (model × dataset) cell, standard deviations or 95% CIs on every headline number, and paired significance tests for the Gen-2-vs-Gen-3 claim (Wilcoxon signed-rank on per-item confidence deltas; bootstrap CIs on ECE differences with $B = 10{,}000$ resamples). If current cell sizes are 40–200 items, target ≥500 items per cell before submission — reviewers will compute the ECE standard error themselves ($\text{SE}_{\text{ECE}} \approx \mathcal{O}(1/\sqrt{n})$ per bin) and 40-item cells will not survive.
- **Blockers to clear before any SE-dependent claim:** restore `SE_NUM_SAMPLES = 5` and `SKIP_NLI_CLUSTERING = False`; exclude all pre-2026-05-10 (pre-rubric-harmonization) data from cross-dataset comparisons.

---

## 1. Abstract (150–250 words, single paragraph)

Exact information sequence (one sentence each unless noted):

1. **Context sentence:** LLMs are increasingly deployed in settings (legal, medical, agentic routing) where knowing *when the model is wrong* matters as much as accuracy.
2. **Gap sentence:** Verbalized confidence is known to be inflated, but prior elicitation studies conflate reasoning and self-assessment in a single forward pass and largely predate thinking models.
3. **Method sentence (2 sentences):** Introduce the three-generation architecture — Gen 1 reasoning/answer only, Gen 2 own-work-aware self-rating on a 10-class verbal-probability rubric, Gen 3 blinded critique framed as reviewing "someone else's" solution. Name the signal families compared: four logit aggregates, verbalized confidence, binary P(correct) judgment, and semantic entropy (Kuhn et al., 2023).
4. **Scale sentence:** 6 open-weights instruct models (7B–31B, including three thinking models) × 4 task types (arithmetic, multi-hop commonsense, factual recall, legal reasoning).
5. **Findings sentences (2–3):** State the two or three strongest quantified results, e.g., "Blinding reduces ECE by X.X points on average (p < 0.001), with the largest gains on models emitting `<think>` chains"; "Verbalized confidence and mean token log-probability diverge most on [task], where AUROC differs by X."
6. **Takeaway sentence:** Practical guidance — which signal to trust for which task family, at zero fine-tuning cost.

Do **not** put dataset names, model names, or citations in the abstract beyond what's needed; do put at least two concrete numbers.

---

## 2. Section 1 — Introduction (≈1 page, ending on page 2 with Figure 1 already visible)

### Content blocks, in order

- **¶1 Hook (4–5 sentences):** Deployment framing. A model routing hard questions to human review needs calibrated confidence; an agent deciding whether to retry needs it. Cite Guo et al. (2017) for the classical calibration frame and Kadavath et al. (2022) for LLM self-knowledge.
- **¶2 The problem (4–5 sentences):** Single-pass confidence elicitation is structurally biased: the model rates its answer immediately after producing a coherent reasoning chain (sunk-cost inflation, cite Mielke et al. 2022), and for thinking models the `<think>` block consumes 1,000–3,000+ tokens before structured output, so the confidence request is often truncated away entirely. State both failure modes explicitly — one is cognitive, one is mechanical, and the architecture fixes both.
- **¶3 Our approach (5–6 sentences):** Describe the three generations in one sentence each. Emphasize the key design element: Gen 3 receives the Gen-2 self-assessment as context ("The respondent self-assessed: Confidence X/10") so the blinded critic can endorse or push back — a within-item paired design that directly measures authorship bias.
- **¶4 Scope sentence + findings preview (4–5 sentences):** Enumerate the grid (6 models, 4 tasks, 7 signals) and preview the top three findings with numbers. Each previewed finding must point to a specific figure/table ("(Fig. 3, Table 2)").
- **Contribution bullet list (3 bullets, bolded lead-ins):** exactly the triaged contributions from §0. Each bullet ≤2 lines.

### Visual asset: **Figure 1 (Teaser) — top of page 2, full column width (or full page width if two-column)**

- **Content:** A two-panel composite.
  - **Panel (a):** Schematic of the three-generation pipeline as three boxes left-to-right: `Gen 1: reason + answer (8,192 tok)` → `Gen 2: "YOUR OWN reasoning" self-rating (512 tok)` → `Gen 3: blinded critique, "someone else's solution" (4,096 tok)`. Annotate what each pass emits (`Answer: X` + token log-probs; `Confidence: N`; revised `Confidence: N` + `Correct: Yes/No`). Use a lock/eye icon on Gen 3 to signal blinding.
  - **Panel (b):** One headline result as a paired-dot or slope chart: per-model Gen-2 vs Gen-3 mean confidence (or ECE), with arrows pointing down where blinding deflates overconfidence. This panel must make the paper's claim legible in <10 seconds.
- **Caption:** 3–4 sentences; must be self-contained (state models, n, and the effect size). Area chairs skim Figure 1 + abstract before anything else — this figure carries the accept/reject prior.

---

## 3. Section 2 — Related Work (0.75–1 page; may move to after Method if the method is the star)

Organize as 4 titled paragraphs (bolded run-in headers, no subsection numbers to save space). Each paragraph: 3–5 works, ending with a one-sentence contrast ("Unlike X, we …").

- **Calibration of neural networks.** Guo et al. 2017 (ECE, temperature scaling); Niculescu-Mizil & Caruana 2005; Naeini et al. 2015 (BBQ); Lakshminarayanan et al. 2017 (deep ensembles — note cost-prohibitive at LLM scale, motivating single-model multi-signal approach); Hendrycks & Gimpel 2017 (AUROC as discrimination metric). Contrast: we apply this measurement framework to *prompt-elicited* confidence, not post-hoc recalibration.
- **LLM self-knowledge and verbalized confidence.** Kadavath et al. 2022 (P(True)); Tian et al. 2023 (verbal-label rubrics beat raw numerics — our 10-class rubric implements this); Lin et al. 2022 (trained linguistic calibration — contrast: we are zero-shot); Xiong et al. 2023/ICLR-24 (elicitation-method survey; does not cover thinking models); Huang et al. 2023 (logit beats verbalized on factual QA — motivates carrying both). Contrast: 10-point rubric + authorship framing + blinded second pass; thinking models included.
- **Sampling-based uncertainty.** Kuhn et al. 2023 (semantic entropy — primary methodology citation; state your implementation matches: temperature 0.5, DeBERTa-large-MNLI bidirectional entailment at threshold 0.5, N=5); Farquhar et al. 2024 (SE for hallucination detection); SelfCheckGPT (Manakul et al. 2023); self-consistency (Wang et al. 2022). Contrast: SE evaluated side-by-side against logit and verbalized signals on binary and numeric tasks, not just open-ended QA.
- **Self-critique and multi-pass evaluation.** Saunders et al. 2022; Self-Refine (Madaan et al. 2023 — critique used to revise answers; we use it only to re-rate confidence); Reflexion (Shinn et al. 2023); McAleese et al. 2024 (separate critic model; we use the *same* model blinded — cheaper, and isolates authorship bias as the manipulated variable).

**Table 1 (optional, appendix if space-tight): methodological-contrast table** — rows = prior work threads, columns = {multi-signal, multi-model, thinking models, blinded critique, binary+numeric tasks}, checkmarks. This is the §10 table from the related-works handoff, compressed. If kept in main text, place at the bottom of the Related Work page.

---

## 4. Section 3 — Method: The Three-Generation Elicitation Architecture (1.25–1.5 pages)

This is the paper's core intellectual property. Structure:

### 3.1 Problem setup and notation (1 short paragraph + notation block)

- Define: input $x$, model $M$ with parameters $\theta$, generated response $y = (y_1, \dots, y_T)$, correctness indicator $c \in \{0,1\}$, and a confidence signal $s: (x, y) \to \mathbb{R}$.
- Define calibration targets formally:
  $$\text{ECE} = \sum_{b=1}^{B} \frac{|I_b|}{n} \left| \text{acc}(I_b) - \text{conf}(I_b) \right|, \quad B = 10 \text{ equal-width bins}$$
  and AUROC of $s$ as a correct-vs-wrong discriminator (cite Hendrycks & Gimpel 2017). State the distinction explicitly: ECE measures scale alignment, AUROC measures ranking quality; a signal can ace one and fail the other — this distinction structures the entire results section.

### 3.2 Confidence signal families (compact enumerated list + one formula each)

- **Logit aggregates** (computed on the Gen-1 greedy pass from `outputs.scores`; note the clean teacher-forced re-derivation when decoding guards are active, one sentence):
  - $s_{\text{mean-lp}} = \frac{1}{T}\sum_{t=1}^{T} \log p_\theta(y_t \mid y_{<t}, x)$ (primary logit signal)
  - $s_{\text{geom}} = \exp(s_{\text{mean-lp}})$; $s_{\text{amean}} = \frac{1}{T}\sum_t p_\theta(y_t)$; $s_{\min} = \min_t p_\theta(y_t)$
  - Cite Malinin & Gales 2021 (sequence-level grounding) and Fomicheva et al. 2020 (aggregation strategies).
- **Verbalized confidence** $s_{\text{verb}} \in \{1,\dots,10\}$ on the harmonized rubric mapping each integer to a verbal label + explicit probability decile ("6 = Better than even, 50–60%"). Reproduce the full rubric in the appendix, not here; here give one example row and cite Tian et al. 2023 for verbal-label grounding.
- **Binary self-judgment** $s_{\text{MLTN}} \in \{0,1\}$ ("more likely correct than not") — the P(True) analogue (Kadavath et al. 2022) at a fixed 0.5 threshold.
- **Semantic entropy** $\text{SE}(x) = -\sum_c p(c\mid x)\log p(c \mid x)$ over NLI-entailment clusters $c$ of $N{=}5$ temperature-0.5 samples; cluster membership requires bidirectional entailment $> 0.5$ under DeBERTa-large-MNLI. One sentence on `predictive_entropy` as the non-clustered ablation.

### 3.3 The three generations (the centerpiece — one paragraph per pass + design-rationale sentences)

For each pass specify, in running text: prompt framing, token budget, and what is stored.

- **Gen 1 — reasoning and answer only.** No confidence rubric present (`include_confidence=False`); 8,192-token budget sized for `<think>` chains; `<think>…</think>` stripped (and GPT-OSS harmony envelope split on `assistantfinal`) before answer extraction; all logit signals computed here. *Rationale sentence:* removing the rubric from Gen 1 prevents confidence anchoring from contaminating the reasoning pass and guarantees the answer never competes with the confidence request for budget.
- **Gen 2 — own-work-aware self-rating.** "The following is YOUR OWN reasoning chain…" framing; reasoning trimmed to 3,000 chars; 512-token budget that cannot be exhausted before `Confidence: N` is emitted. *Rationale:* explicit first-person framing activates self-reflection; separate pass guarantees completion. Outputs `single_pass_confidence`, `single_pass_correct`.
- **Gen 3 — blinded critique.** "You are reviewing a solution submitted by **someone else**…"; reasoning trimmed to 2,000 chars; the Gen-2 score is shown to the critic as a prior to endorse or contest; thinking disabled on this pass (`TWO_PASS_DISABLE_THINKING`); 4,096-token budget; fallback to Gen-2 values on extraction failure (report the fallback rate in the appendix). *Rationale:* blinding removes authorship bias while holding the model, the reasoning text, and the rubric fixed — so any Gen-2→Gen-3 shift identifies the authorship effect.

### 3.4 Acknowledged limitation (2–3 sentences, in the method — not buried)

Gen 2/3 read the reasoning *as text*, not from the Gen-1 hidden states; activation-level uncertainty is not retrospectively accessible. The logit signals compensate by capturing internal state during generation — this is the argument for the three signal classes being complementary rather than redundant. Putting this here, voluntarily, defuses the most predictable reviewer objection.

### Visual asset: **Figure 2 — System architecture, top of the Method page, full width**

- Detailed version of Figure 1(a): three vertical lanes (one per generation), each lane showing prompt template snippet (2–3 lines, monospace), token budget, and output fields. A horizontal band underneath maps each of the 7 signals to the pass it comes from (logit signals ← Gen 1; `single_pass_*` ← Gen 2; `verbalized_confidence`, `more_likely_than_not` ← Gen 3; SE ← separate sampling loop). Color-code signal families consistently — these colors must reappear in every results figure (same hue for "verbalized" everywhere).
- Include the standard-flow vs reasoning-flow fork as a small branch annotation: standard models (Qwen2.5, Llama-3.1-8B, Gemma-2-9B) skip the `<think>` handling but still receive Gen 2/3.

---

## 5. Section 4 — Experimental Setup (0.75 page + heavy appendix offload)

### 4.1 Models (compact table)

**Table 1 — Model lineup.** Columns: Model | HF ID | Params (total/active) | Flow (standard / thinking) | Precision. Rows: Qwen2.5-7B-Instruct (7B, standard, fp16); Qwen3-30B-A3B (30B/~3B MoE, thinking, bf16); Llama-3.1-8B-Instruct (8B, standard, fp16); Gemma-2-9B-IT (9B, standard, fp16); Gemma-4-31B-IT (31B, thinking, fp16); GPT-OSS-20B (20B, thinking via harmony channels, bf16). One footnote: greedy decoding for the main pass (reproducibility + logit-metric consistency, cite Holtzman et al. 2020 for the degeneration context); `no_repeat_ngram_size=3` guard for GPT-OSS with the inertness argument (the guard only fires on repeated 3-grams, so non-looping generations are byte-identical) in the appendix.

### 4.2 Tasks (2–3 sentences each, or a second compact table)

GSM8K (test; exact numeric match), StrategyQA (test; Yes/No), TriviaQA (validation, `rc.nocontext`; alias-set matching via `normalized_aliases`), LegalBench (`hearsay` + `consumer_contracts_qa` pooled with subset labels; Yes/No). One sentence on why the mix matters: near-deterministic ground truth (GSM8K) through genuinely ambiguous expert judgment (LegalBench) spans the difficulty range calibration must survive.

### 4.3 Protocol details that reviewers will probe (one short paragraph each; full detail → appendix)

- **Seeds and sample sizes:** #seeds per cell, dedup-on-`idx` accumulation, final n per (model × dataset) cell — point to the full matrix in Appendix Table A1.
- **Forced-answer recovery:** when strict `Answer:` extraction fails, a short forced-commit call (8–32 token budget by dataset) recovers an answer; `was_forced` rows are *retained* because dropping them is non-random dropout that biases calibration toward easy items — and a forced guess *should* carry low confidence, making it a calibration probe in itself. This retention argument is a reviewer-pleaser; make it in main text.
- **Refusal handling:** tail-scanned conservative pattern detector; refusals flagged and excluded from accuracy/calibration, reported as their own bucket.
- **Rubric harmonization:** all reported data postdates the 2026-05-10 uniform rubric; earlier data excluded.

### 4.4 Metrics

ECE (10 bins), reliability diagrams, AUROC per signal, Brier score (add it — cheap and standard), Spearman rank correlation between signals, and the paired Gen-2/Gen-3 delta tests (Wilcoxon + bootstrap CI). For verbalized confidence, map rubric class $k$ to the midpoint of its probability decile ($k \to (k-0.5)/10$) before computing ECE; state this mapping explicitly or reviewers will ask.

---

## 6. Section 5 — Results (2.5–3 pages; the bulk of the paper)

Order results by claim strength, not by chronology of experiments. Recommended arc: (1) main calibration matrix → (2) blinding effect → (3) signal comparison → (4) thinking-model analysis → (5) task-specific findings.

### 5.1 Main calibration matrix

**Table 2 — the headline table, top of first results page, full width.**
- Rows: 6 models grouped by family (rule between thinking and standard models). Columns grouped by dataset: Accuracy (%) ↑ | ECE_verb ↓ | ECE_logit ↓ | AUROC_verb ↑ | AUROC_SE ↑ (pick the 4–5 most informative metric columns; full grid in appendix).
- **Formatting rules:** direction arrows in the header (↑/↓); mean ± std over seeds; **bold** the best value per column, underline second-best; shade cells of thinking-model rows lightly so family effects pop; per-cell n in a footnote or gray subscript.
- Every claim in the running text must name a cell: "Gemma-4-31B attains 95.9% on GSM8K yet ECE_verb of X …".

**Figure 3 — Reliability diagrams, 4×2 or 4×3 small-multiples panel.**
- One column per dataset, rows = signal type (verbalized Gen 3, logit mean-lp, SE-derived confidence). Diagonal = perfect calibration; bars or dots per bin with a translucent histogram of bin mass behind (reviewers *will* look for bin-population evidence); one line per model, consistent model colors across the entire paper.
- The visual story to engineer: GSM8K curves hugging the diagonal at the top-right (near-ceiling accuracy, high confidence) vs LegalBench curves sagging below the diagonal (overconfidence in the ambiguous domain).

### 5.2 The blinding effect (Gen 2 vs Gen 3) — the paper's novelty payload

**Figure 4 — Gen-2 vs Gen-3 paired analysis, two panels.**
- **(a)** Per-model distribution of per-item deltas $\Delta_i = s^{(3)}_{\text{verb},i} - s^{(2)}_{\text{verb},i}$ (violin or half-eye plots), split by correctness: the money pattern is deltas shifting negative on *wrong* answers (blinding deflates unjustified confidence) while staying ~0 on correct ones.
- **(b)** ECE before/after blinding per (model × dataset), as a dumbbell/slope chart with significance stars from the bootstrap test.
- Caption states the mechanism claim and the test: "Wilcoxon signed-rank on paired per-item scores; ** p<0.01."
- In text, also report the binary version: Gen-2 vs Gen-3 `more_likely_than_not` flip rates conditioned on correctness — a clean 2×2 that directly quantifies sunk-cost bias.

### 5.3 Signal comparison and complementarity

**Figure 5 — Signal-agreement matrix + AUROC bar panel.**
- **(a)** Spearman correlation heatmap among the 7 signals (pooled, then per-dataset small versions in appendix). Annotate the key off-diagonal: verbalized vs mean-lp divergence.
- **(b)** Grouped bar chart: AUROC per signal per dataset, error bars = bootstrap 95% CI. This answers open question #2 (does SE beat logits on binary/numeric tasks?) visually.
- Text finding to feature: logit signals inter-correlate strongly but diverge from verbalized confidence exactly on hard items where the model *sounds* confident at low token probability — give 1 qualitative example inline (2–3 lines, monospace), full examples in appendix.

### 5.4 Thinking vs standard models

- Paired comparison at matched scale where possible (Qwen3-30B-A3B ~3B active vs the 7–9B standard models — address the active-vs-total parameter confound in one honest sentence).
- Answer discussion-question #3 with data: does longer reasoning improve calibration or just confidence? Plot ECE (or over-confidence gap $\text{conf} - \text{acc}$) vs `<think>`-length quartiles.

### 5.5 Task-specific findings (one tight paragraph each, each anchored to a panel)

- **GSM8K:** calibration on errors at 95%+ accuracy — are the rare mistakes flagged by low confidence? Report the mean confidence on wrong answers vs correct.
- **StrategyQA:** the Yes-bias asymmetry. **Table 3 (small):** confusion matrix + per-polarity calibration (accuracy and mean confidence for Yes-answers vs No-answers). Finding: models are better calibrated on "No."
- **TriviaQA:** SE's home turf — high-SE rows as the "knows it's guessing" frontier.
- **LegalBench:** cross-subtask calibration (hearsay = rule application vs consumer-contracts = language interpretation) within one expert domain.

### 5.6 Ablations (compact table or 4-panel figure; some can go to appendix with pointers)

Minimum ablation set a main-track reviewer expects here:
1. Gen-3 **without** the Gen-2 score shown (isolates anchoring on the self-assessment from blinding itself) — this is the single most important ablation; run it if it doesn't exist yet.
2. Gen 2 without the "YOUR OWN" authorship framing (neutral framing control).
3. Rubric ablation: 10-class verbal rubric vs bare 0–100 numeric elicitation (connects to Tian et al. 2023).
4. SE ablation: NLI clustering vs `predictive_entropy` (no clustering); N=5 vs smaller sample counts.
5. Reasoning-trim length (3,000/2,000 chars) sensitivity — appendix is fine.

---

## 7. Section 6 — Discussion & Limitations (0.5 page)

- **Mechanism discussion:** why blinding works — authorship-bias removal vs mere re-asking; use the no-Gen-2-score ablation to separate them.
- **Cross-model comparability of verbalized scores:** is a "7" from Qwen3 a "7" from GPT-OSS? Use logit signals as the objective anchor (open question #4).
- **Practical guidance paragraph:** a small decision recipe — e.g., "for arithmetic, trust logit signals; for expert domains, blinded verbalized confidence; route to human review when signals disagree." Reviewers reward actionable synthesis.
- **Limitations (bulleted, honest, specific):** text-based (not activation-based) retrospection; open-weights ≤31B only, no frontier API models; greedy decoding scope; English-only; 3× inference cost of the pipeline (quantify: ~$\mathcal{O}(3)$ forward passes per item, but Gen 2/3 budgets are 512/4,096 tokens vs Gen 1's 8,192, so real cost ≈ 1.5–1.7× tokens — compute the true ratio from your logs and state it).
- **Broader impact (2–3 sentences or checklist-only):** calibrated confidence reduces silent failure in deployment; miscalibration signals could be misused to game oversight — standard framing suffices.

---

## 8. Section 7 — Conclusion (1 short paragraph)

Restate the claim with the strongest number, one sentence on generality, one forward sentence (activation-level confidence reading, fine-tuning on blinded-critique targets).

---

## 9. Appendix Plan (unlimited pages — use them)

- **A. Full reproducibility matrix:** per-cell n, seeds, dates, rubric version, decoding config table (the ngram/rep-penalty/stop-strings matrix per model variant), hardware (GPU type, `device_map="auto"` sharding), total GPU-hours.
- **B. Complete prompts:** all three generation prompts verbatim, the full 10-class rubric, forced-answer prompts (instruct + base Q&A variants), extraction regex cascade description.
- **C. Full metric grids:** every (model × dataset × signal × metric) cell; per-dataset correlation heatmaps; Brier decompositions.
- **D. Failure-mode gallery:** repetition loops, harmony-envelope parsing, truncation/forced-answer statistics (`was_forced` rates per cell), refusal counts, Gen-3 extraction-failure/fallback rates.
- **E. Qualitative examples:** 6–10 full items showing Gen-1 reasoning, Gen-2 rating, Gen-3 critique text side by side — include at least two where Gen 3 correctly pushes back and one where it wrongly capitulates.
- **F. License/asset table:** dataset and model licenses (NeurIPS checklist requirement).

---

## 10. Reviewer Alignment — What Main-Track Acceptance Requires

- **Novelty bar:** the three-generation architecture + blinding is the novelty; defend it against "it's just self-consistency/self-refine" with the contrast table and the no-revision design point (critique re-rates confidence, never edits the answer).
- **Rigor bar (the usual rejection reasons, pre-empted):**
  - Multiple seeds + CIs on *every* number. No CI → "results may be noise" → reject.
  - Paired statistical tests for the central Gen-2/Gen-3 claim.
  - The anchoring ablation (5.6.1). Without it, a reviewer writes "the Gen-3 shift may be regression to the mean from re-asking" and you have no reply.
  - Cell sizes ≥ a few hundred; report ECE bin populations.
- **Completeness bar:** all 6 models × all 4 datasets fully populated. A grid with holes ("GPT-OSS TriviaQA pending") reads as work-in-progress → workshop, not main track.
- **Reproducibility checklist:** code + configs + seeds released (state the repo will be public); exact HF model IDs; the NeurIPS paper checklist answered honestly.
- **Anticipated reviews and your prepared rebuttals:**
  - *"Only open-weights ≤31B"* → framing: logit-level signals require white-box access; API models can't provide the logit arm of the comparison.
  - *"3× cost"* → real token-ratio number + the routing use-case where a 1.6× cost is trivial against human-review cost.
  - *"Verbalized confidence differences could be prompt-idiosyncratic"* → rubric ablation + harmonized-rubric protocol.
  - *"Is Gen 3 really blinded? The model may recognize its own style"* → acknowledge; report any style-recognition probe if available; frame as "pseudo-blinding" honestly if not.
- **Writing discipline:** every claim in intro/abstract must be traceable to a numbered figure/table; every figure caption self-contained; consistent signal colors and model colors across all figures; metrics always with direction arrows; no result appears only in text without a visual/tabular anchor.

---

## 11. Page-Budget Map (9-page target)

| Pages | Content |
|---|---|
| 1.0–2.0 | Title/abstract/intro + Figure 1 |
| 2.0–3.0 | Related work (+ contrast table) |
| 3.0–4.5 | Method + Figure 2 |
| 4.5–5.25 | Experimental setup + Table 1 |
| 5.25–8.0 | Results: Table 2, Figures 3–5, Table 3, ablations |
| 8.0–8.75 | Discussion + limitations |
| 8.75–9.0 | Conclusion |
