# AI Workshop / Extended Abstract Blueprint (4–6 Pages)

**Target venues:** NeurIPS/ICML/ICLR workshops on uncertainty quantification, trustworthy ML, reliable/safe LLMs, or reasoning (e.g., "UQ for LLMs", "Trustworthy NLP", "Reliable and Responsible FMs"); also fits an ACL Findings-adjacent short paper with format changes.
**Working title pattern:** *"Blinded Self-Critique Deflates LLM Overconfidence: Early Evidence from a Three-Generation Elicitation Pipeline"*
**Page budget:** 4 pages of main text (some workshops allow 5–6) + unlimited references; appendix usually allowed but reviewers are *not* obligated to read it.

This blueprint is calibrated to the same confidence-telemetry study as the main-track blueprint, but re-scoped for the workshop contract: **one sharp idea, honest preliminary evidence, and material that sparks discussion** — not a full reproducibility matrix. The current data (40–200 items per cell, Qwen3 + Gemma-4 cells most complete, SE temporarily on N=1) is *already sufficient* for this format if framed correctly.

---

## 0. Pre-Writing Decisions

- **Pick ONE claim, cut the rest.** A workshop paper is a single finding with a spotlight on it. Recommended: *"Separating self-assessment from reasoning — and blinding the assessor to authorship — measurably changes verbalized confidence, most strongly on wrong answers."* The full 7-signal × 6-model matrix is main-track material; here it appears only as context.
- **Scope reduction that keeps the story intact:** 2–3 models (Qwen3-30B-A3B as the primary thinking model, Gemma-4-31B as the second family, optionally Qwen2.5-7B as the standard-flow contrast) × the 4 datasets × 3 signals (Gen-2 verbalized, Gen-3 verbalized, mean token log-prob as the objective anchor). Explicitly defer SE if `SE_NUM_SAMPLES` is still 1 — a footnote "semantic entropy deferred to the full study" is respectable; a claim built on N=1 sampling is a desk-reject conversation in the hallway.
- **What "preliminary" licenses you to do:** report cells of 100–200 items with binomial CIs, mark incomplete cells as "in progress" in gray, and pose open questions *as questions*. What it does not license: missing error bars, cherry-picked cells presented as the grid, or hidden failure rates.
- **Positioning sentence to write first:** "This workshop paper presents the architecture and first evidence; the full cross-model calibration study is ongoing." Reviewers reward this honesty and it converts the venue's expectations in your favor.

---

## 1. Abstract (100–150 words)

Five sentences, no more:

1. Problem: single-pass verbalized confidence is inflated by sunk-cost bias and, for thinking models, mechanically broken by `<think>` budget exhaustion.
2. Idea: a three-generation pipeline — reason (Gen 1), rate your own work (Gen 2), critique it blinded as "someone else's" with the self-rating shown as a contestable prior (Gen 3).
3. Setup: 2–3 open-weights models including thinking models, 4 task types spanning arithmetic to legal reasoning.
4. Finding with a number: "Blinding shifts confidence down by X points on incorrect answers vs Y on correct ones (n=…, p=…)."
5. Stakes/discussion hook: implications for confidence-based routing and for whether models can audit themselves.

---

## 2. Section 1 — Introduction (0.75 page, compressed)

- **¶1 (3–4 sentences):** Deployment need for calibrated confidence + the two failure modes of single-pass elicitation (cognitive: sunk-cost inflation, cite Mielke et al. 2022, Kadavath et al. 2022; mechanical: thinking models spend 1,000–3,000+ tokens on `<think>` before any structured output, so confidence requests get truncated). The mechanical point is your freshest motivation — thinking models are new enough that most prior elicitation work (Tian et al. 2023; Xiong et al. 2023) simply predates them. Lead with it.
- **¶2 (3–4 sentences):** The three-generation idea in three sentences, one per pass. Include the key design detail even at this length: Gen 3 sees the Gen-2 score and may push back — giving a paired, within-item measurement of authorship bias.
- **¶3 (2–3 sentences):** What this paper shows (one number) and what it deliberately defers (full grid, SE arm, ablations) — with a pointer sentence "we outline the open questions this raises in §5."
- **Contribution list:** either 2 bullets or skip entirely (workshop papers may fold contributions into ¶3; saves 6–8 lines).

### Visual asset: **Figure 1 — pipeline + headline result, top-right of page 1 (wrap text around it) or spanning the top of page 2**

- Single combined figure, because you cannot afford two: left 60% = the three-generation pipeline as three compact boxes with token budgets and prompt-framing quotes ("YOUR OWN reasoning…" / "submitted by someone else…"); right 40% = one paired plot of Gen-2 vs Gen-3 confidence split by correct/wrong (dumbbell per model, or a 2×2 mean-confidence grid).
- Caption ≤3 sentences but self-contained. At a workshop poster session this figure IS the paper — design it to survive being screenshotted into a slide.

---

## 3. Section 2 — Method (0.75–1 page)

Compression strategy: formal notation only where it earns its space; prompts by quotation, not reproduction.

- **Signals (one compact paragraph, not subsections):** define only the three reported signals. Verbalized confidence $s_{\text{verb}} \in \{1,\dots,10\}$ on a harmonized rubric mapping each class to a verbal label + probability decile (one example: "6 = Better than even, 50–60%"); mean token log-probability $\frac{1}{T}\sum_t \log p_\theta(y_t \mid y_{<t}, x)$ from the greedy Gen-1 pass as the white-box anchor (Malinin & Gales 2021); the binary `more_likely_than_not` judgment as the P(True) analogue. Footnote the remaining signal families as "collected but deferred."
- **The three generations (the core — one short paragraph each):**
  - **Gen 1:** reasoning + answer, *no confidence rubric present*, 8,192-token budget, `<think>` stripped before extraction (GPT-OSS harmony envelope split on `assistantfinal` — one clause, appendix for detail).
  - **Gen 2:** "YOUR OWN reasoning" framing, 512-token dedicated budget → the confidence call can never be truncated away.
  - **Gen 3:** blinded "someone else's solution" framing, Gen-2 score provided as a contestable prior, thinking disabled on this pass, fallback to Gen-2 on extraction failure (report the fallback rate — one number, main text).
- **Metrics (2–3 sentences):** ECE with 10 bins (map rubric class $k$ to decile midpoint $(k-0.5)/10$), AUROC for correct-vs-wrong discrimination (Hendrycks & Gimpel 2017), and the paired per-item delta $\Delta_i = s^{(3)}_i - s^{(2)}_i$ tested with Wilcoxon signed-rank. State the ECE-vs-AUROC distinction in one sentence — scale vs ranking.
- **Setup (one dense paragraph or a 5-row table):** models with parameter counts (Qwen3-30B-A3B: 30B total / ~3B active MoE, thinking; Gemma-4-31B-IT: thinking; Qwen2.5-7B-Instruct: standard), greedy decoding, 4 datasets with answer formats and n per cell, forced-answer retention policy in one sentence (dropping truncated hard items would bias calibration toward easy samples — keep this sentence, reviewers like it), refusals flagged and excluded.

**Related work goes here, compressed to one paragraph (5–7 sentences)** rather than its own section: Kadavath et al. 2022 (P(True)); Tian et al. 2023 (verbal-label rubrics); Xiong et al. 2023 (elicitation survey, pre-thinking-models); Kuhn et al. 2023 (SE, deferred arm); Saunders et al. 2022 / Madaan et al. 2023 (critique-to-revise vs our critique-to-re-rate); one contrast clause per citation. Workshop reviewers accept a related-work paragraph; they do not accept a missing Kadavath/Tian citation.

---

## 4. Section 3 — Preliminary Results (1.25–1.5 pages)

Two figures + one table maximum. Every number carries a CI or is visibly n-labeled.

### Table 1 — the setup-and-headline table (top of the results page)

- Rows: (model × dataset) cells actually run. Columns: n | Accuracy % | mean conf Gen-2 | mean conf Gen-3 | ECE Gen-2 ↓ | ECE Gen-3 ↓ | Δ significant?
- Formatting: direction arrows in headers; **bold** the better of the Gen-2/Gen-3 ECE pair per row; ± binomial/bootstrap CI on accuracy and ECE; gray italic "(in progress)" rows for planned-but-unfinished cells — showing the intended grid signals the full-paper trajectory without overclaiming.
- Use the real observed anchors where available: Qwen3 GSM8K ~95.2% / StrategyQA ~74.6% / TriviaQA ~77.3% / LegalBench ~65.5%; Gemma-4 GSM8K ~95.9% / StrategyQA ~85.7%.

### Figure 2 — the blinding effect (the paper's one real result figure)

- **Panel (a):** per-item delta distributions $\Delta_i$ split by correctness (correct vs wrong), one pair of violins/half-eyes per model. Target pattern: wrong-answer deltas shifted negative, correct-answer deltas centered at 0. Annotate medians and the Wilcoxon p-value.
- **Panel (b):** one reliability diagram overlaying Gen-2 (dashed) and Gen-3 (solid) curves for the most complete cell (e.g., Qwen3 × StrategyQA), with bin-mass histogram behind the curves. One cell is enough at a workshop; the grid version is future work.
- Consistent colors: one hue per model, dashed=Gen-2 / solid=Gen-3 everywhere.

### Findings paragraphs (one per finding, 3–5 sentences each, each anchored to the table/figure)

1. **Blinding deflates unjustified confidence:** the Δ-by-correctness asymmetry, with the flip-rate version from `more_likely_than_not` (Gen-2 "Yes" → Gen-3 "No" conditioned on wrong answers) as a second, threshold-level confirmation.
2. **The effect is largest on thinking models / hedged reasoning:** Gen-3 is more conservative than Gen-2 exactly where the Gen-1 chain hedged (report the observation qualitatively with one 2–3 line quoted example inline — workshops love a concrete critique excerpt where the blinded reviewer pushes back on "the respondent self-assessed 9/10").
3. **Task texture (2–3 sentences each, only where you have data):** GSM8K near-ceiling accuracy makes *error* calibration the question; StrategyQA shows a systematic Yes-bias with better calibration on "No" answers; LegalBench's ~65% regime is where verbalized and logit signals diverge most. Frame each as an observation + open question, not a settled claim.

### What NOT to include

- No ablation section (name the needed ablations in §5 instead — especially Gen-3-without-the-Gen-2-score, which separates blinding from anchoring).
- No cross-signal correlation heatmap, no SE results at N=1, no base-model arm, no engineering war stories (repetition loops, harmony parsing → one appendix paragraph if the workshop allows appendices, else cut).

---

## 5. Section 4 — Discussion & Open Questions (0.5–0.75 page) — *the section workshops actually select for*

Structure as a short mechanism paragraph + an explicit numbered open-question list. Workshop PCs pick papers that will generate poster-session conversation; hand them the conversation:

- **Mechanism paragraph:** two rival explanations for the Gen-2→Gen-3 shift — genuine authorship-bias removal vs regression-to-the-mean from re-asking with a visible prior. State plainly that the deciding ablation (blinded critique *without* the Gen-2 score shown) is the next experiment. Naming your own confound before reviewers do is the strongest move available in a 4-pager.
- **Open questions (numbered, 1–2 sentences each):**
  1. Is a verbalized "7" comparable across model families, or do families have different rubric dialects? (Logit signals as the objective anchor.)
  2. Do thinking models calibrate better, or just commit harder? (`<think>`-length vs overconfidence gap.)
  3. Does semantic entropy's advantage on open-ended QA (Kuhn et al. 2023) survive binary and numeric formats? (The deferred arm.)
  4. Can Gen-3-style blinded critique supervise a *fine-tuned* calibration head, converting the 3-pass inference cost into a train-time cost?
- **Practical stakes sentence:** confidence-based routing to human review is the deployment story; state the cost honestly (extra passes are 512 + 4,096-token budgets against an 8,192-token Gen 1 — roughly 1.5–1.7× tokens, not 3×).
- **Limitations (3–4 bullet lines):** preliminary cell sizes; text-based retrospection (Gen 2/3 read reasoning as text, not activations); open-weights ≤31B; pseudo-blinding (the model might recognize its own style — unverified).

---

## 6. Section 5 — Conclusion (3–4 sentences, or merge into §4)

One sentence restating the finding with its number, one on the architecture as the reusable artifact, one on the full-study roadmap (complete 6×4 grid, SE arm at N=5, anchoring ablation).

---

## 7. References & Appendix

- **References (unlimited):** the ~12 load-bearing citations: Guo 2017; Kadavath 2022; Tian 2023; Xiong 2023; Kuhn 2023; Mielke 2022; Saunders 2022; Madaan 2023; Hendrycks & Gimpel 2017; Malinin & Gales 2021; dataset papers (Cobbe 2021, Geva 2021, Joshi 2017, Guha 2023); model reports for the models actually shown. Skip the long-tail citations the main-track version carries.
- **Appendix (only if the venue allows, and keep it to 2–3 pages):** (A) full prompts + the 10-class rubric verbatim; (B) per-cell n/seed table; (C) 2–3 full qualitative examples of Gen-1 reasoning → Gen-2 rating → Gen-3 critique. Do not put load-bearing results in the appendix — workshop reviewers won't read it.

---

## 8. Reviewer Alignment — What Workshop Acceptance Requires

Workshop review is 1–2 reviewers, light-touch, selecting for **relevance, novelty-of-idea, honesty, and discussion value** — not completeness. Optimize accordingly:

- **Fit paragraph:** mirror the workshop CFP's language in your intro (if the CFP says "uncertainty quantification for foundation models," those words appear in ¶1). Workshop PCs triage on topical fit first.
- **The idea must be legible in 90 seconds:** abstract + Figure 1 must carry the whole story. Reviewers here genuinely do read only that far before forming a verdict.
- **Honest-preliminary beats fake-complete:** CIs on everything, "(in progress)" cells shown, deferred arms footnoted. The fastest way to lose a workshop reviewer is a 4-pager cosplaying as a finished study.
- **Acceptance criteria checklist (typical workshop rubric):**
  - Novel or timely idea? → the blinded-critique architecture + thinking-model motivation (timeliness is your ace: `<think>`-budget breakage of single-pass elicitation is a 2025–2026 problem).
  - Technically sound as far as it goes? → paired tests, CIs, forced-answer retention rationale.
  - Sparks discussion? → §4's named confound + numbered open questions.
  - Within scope/pages? → hard 4-page discipline; workshops desk-reject on page violations more mechanically than main tracks.
- **Non-archival strategy note:** most workshops are non-archival — this paper can and should be the trailer for the main-track submission. Do not burn the full result grid here; show the architecture, the single strongest effect, and the roadmap. Anything published here as a headline number is spent as a novelty claim later, but an architecture + preliminary-evidence workshop paper strengthens (not blocks) the subsequent main-track submission and collects free reviews.
- **Poster derivative:** design Figure 1 and Figure 2 at vector quality and ≥18pt equivalent labels from the start — they will be reused directly on the poster, and workshop acceptance is effectively a poster commitment.

---

## 9. Page-Budget Map (4-page target)

| Pages | Content |
|---|---|
| 0.0–0.15 | Title + abstract |
| 0.15–1.0 | Introduction + Figure 1 |
| 1.0–2.0 | Method (signals, three generations, setup, related-work paragraph) |
| 2.0–3.4 | Results: Table 1, Figure 2, three findings paragraphs |
| 3.4–3.9 | Discussion, open questions, limitations |
| 3.9–4.0 | Conclusion |
| — | References (uncounted) + optional short appendix |
