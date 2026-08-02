# Results Verification — draft claims vs. raw CSVs

**Date:** 2026-08-01. Computed directly from the raw result CSVs on Desktop (`blobfish/`), no reruns.
**Canonical slice that reproduces the draft numbers:** `AI Research (cons55)` consolidated 150-files,
instruct models only, **deduplicated by question within each model×dataset cell**, refusals excluded.
Standard models = the ORIGINAL runs (not the `Raw_Reruns…top20` reruns).

## Claim-by-claim verdict

| Claim (from PAPER_README / draft PDF) | Computed | Verdict |
|---|---|---|
| Pooled N = 3,388 | **3,390** (question-deduped; ±2 = dedup-rule detail) | ✅ |
| Verbalized (primary/blinded): mean AUROC 0.673, floor 0.594 | **0.673 / 0.594** | ✅ exact |
| Seq log prob: mean AUROC 0.680, floor 0.566 | **0.678 / 0.563** | ✅ (rounding) |
| Blinded critique beats owned self-assessment on AUROC in 16/24 cells | **16/24** | ✅ exact |
| Blinded critique lowers wrong-answer confidence in 19/24 cells | **19/24** | ✅ exact |
| Blinded critique improves ECE in only 7/24 cells | **7/24** | ✅ exact |
| Separation: TriviaQA 2.93, StrategyQA 0.80 (Qwen3 §4 deep-dive) | **2.926 / 0.804** (Qwen3-only, owned conf) | ✅ exact |
| Qwen3 deep-dive n = 602 | **602** (150+150+152+150, undeduped combined files) | ✅ exact |
| LegalBench seq AUROC ≈ 0.566, near chance | **0.563** (macro) | ✅ |
| GSM8K / TriviaQA seq AUROC ≈ 0.78 | **0.787 / 0.773** (macro) | ✅ |
| Row-level wrong-answer confidence drop, p ≈ 5.7e-22 | direction confirmed; **sign test p = 3.8e-17** (lower on 304 vs higher on 130 of 434 non-tied wrong rows) | ⚠️ direction ✅, exact p depends on test choice (theirs likely Wilcoxon signed-rank). Re-derive and state the test in Methods. |

**Aggregation method that reproduces the means/floors:** macro-averaging — compute AUROC per
model×dataset cell, average the 6 model cells per dataset, then take mean / min across the 4 datasets.
(Pooling all rows per dataset gives different numbers: verb mean 0.688. The paper must state which.)

## Master table (canonical originals slice, deduped)

| model | dataset | n | acc | AUROC seq-sum | AUROC verb (blinded) | AUROC owned (Gen2) | ECE verb | ECE owned | wrong-conf verb | wrong-conf owned |
|---|---|---|---|---|---|---|---|---|---|---|
| GPT-OSS-20B | gsm8k | 146 | .938 | .849 | .791 | .769 | .047 | .038 | 8.56 | 9.11 |
| GPT-OSS-20B | legalbench | 134 | .791 | .573 | .606 | .495 | .151 | .201 | 6.11 | 7.61 |
| GPT-OSS-20B | strategyqa | 140 | .779 | .611 | .614 | .597 | .138 | .117 | 7.27 | 7.73 |
| GPT-OSS-20B | triviaqa | 146 | .685 | .927 | .868 | .852 | .073 | .088 | 4.71 | 5.33 |
| Gemma2-9B | gsm8k | 146 | .849 | .738 | .610 | .556 | .079 | .096 | 8.64 | 9.64 |
| Gemma2-9B | legalbench | 127 | .764 | .510 | .491 | .530 | .179 | .160 | 7.55 | 7.00 |
| Gemma2-9B | strategyqa | 140 | .714 | .584 | .609 | .689 | .159 | .101 | 7.65 | 6.93 |
| Gemma2-9B | triviaqa | 147 | .694 | .694 | .726 | .725 | .114 | .092 | 6.56 | 7.84 |
| Gemma4-31B | gsm8k | 146 | .959 | .852 | .667 | .575 | .032 | .032 | 9.17 | 9.33 |
| Gemma4-31B | legalbench | 134 | .813 | .549 | .546 | .533 | .143 | .131 | 8.96 | 9.00 |
| Gemma4-31B | strategyqa | 140 | .814 | .631 | .569 | .588 | .127 | .120 | 9.62 | 9.27 |
| Gemma4-31B | triviaqa | 147 | .837 | .770 | .680 | .718 | .072 | .062 | 7.33 | 6.38 |
| Llama3.1-8B | gsm8k | 146 | .815 | .777 | .778 | .594 | .110 | .103 | 6.00 | 7.75 |
| Llama3.1-8B | legalbench | 127 | .693 | .529 | .566 | .594 | .200 | .088 | 5.31 | 6.72 |
| Llama3.1-8B | strategyqa | 138 | .696 | .542 | .537 | .620 | .219 | .123 | 6.26 | 6.26 |
| Llama3.1-8B | triviaqa | 147 | .782 | .585 | .698 | .608 | .148 | .043 | 5.69 | 7.91 |
| Qwen2.5-7B | gsm8k | 146 | .890 | .733 | .744 | .519 | .040 | .035 | 8.56 | 9.06 |
| Qwen2.5-7B | legalbench | 134 | .716 | .537 | .646 | .600 | .169 | .116 | 5.84 | 6.63 |
| Qwen2.5-7B | strategyqa | 143 | .692 | .583 | .708 | .657 | .073 | .136 | 6.05 | 7.39 |
| Qwen2.5-7B | triviaqa | 147 | .694 | .738 | .735 | .718 | .135 | .147 | 6.02 | 8.09 |
| Qwen3-30B | gsm8k | 146 | .959 | .770 | .704 | .720 | .060 | .049 | 7.83 | 8.67 |
| Qwen3-30B | legalbench | 134 | .776 | .678 | .708 | .636 | .139 | .047 | 5.73 | 7.50 |
| Qwen3-30B | strategyqa | 142 | .725 | .592 | .623 | .650 | .125 | .146 | 7.31 | 8.03 |
| Qwen3-30B | triviaqa | 147 | .830 | .926 | .925 | .909 | .062 | .071 | 4.00 | 6.20 |

Definitions: AUROC seq-sum = raw sequence log-prob sum (`seq_confidence_mean` column, which holds the
SUM in these files). verb = primary verbalized (blinded critique w/ owned fallback). owned = Gen-2
`single_pass_confidence`. ECE uses rubric band-midpoints p=(c−0.5)/10, one bin per rating. wrong-conf =
mean 1–10 rating on incorrect answers.

## Slice sensitivity (IMPORTANT — pin before submission)

The 3 standard models also exist as **reruns** (`Raw_Reruns (150 consolidated)`, top-20 columns).
Swapping reruns in for originals (thinking models unchanged) shifts the headline counts:

| Metric | Originals slice (matches drafts) | Reruns slice |
|---|---|---|
| AUROC blinded > owned | 16/24 | 15/24 |
| wrong-answer conf lower | 19/24 | 20/24 |
| ECE better | 7/24 | 10/24 |
| verb macro mean / floor | 0.673 / 0.594 | ~0.668 / 0.586 |

Every qualitative claim survives both slices (good — the thesis is robust), but the exact numbers
don't. **Decide which slice is canonical, write it in Methods, and regenerate every table from it once.**

## Data-quality findings from this audit

1. **Qwen2.5 StrategyQA combined file contains a duplicated chunk** — 300 raw rows collapse to 143
   unique questions. Dedupe before any analysis of that cell.
2. **Cross-seed question overlap is real and expected** (LegalBench pool is small: ~15 duplicated
   questions per cell; TriviaQA/GSM8K ~3–4). The draft's N=3,388 implies question-level dedup was
   applied. State the dedup policy in Methods (repeated samples are near-identical under greedy, so
   deduping is the right call — see #4).
3. **GPT-OSS TriviaQA geometric-mean AUROC = 0.335** — an *inverted* signal, far below chance. Likely a
   harmony-envelope/token-alignment artifact for that family. Exclude or footnote geom for GPT-OSS;
   don't let it silently drag "logit signals" aggregates.
4. **Greedy determinism, measured:** duplicated questions *within* the same run era agree on the final
   answer ~100% (0–1 flips per cell; occasional 1-point verbalized-rating differences). Overlapping
   questions *across* run eras (originals vs reruns) disagree ~13% (16/119; the famous 46-overlap cell
   is Qwen2.5 LegalBench: 6/46 answers differ). Cause: different batch composition + bf16 kernel
   nondeterminism + code-era differences. Paper should claim within-run determinism only.
