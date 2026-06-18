# Results_Draft — Related Works & Paper Framing Handoff

**Created:** 2026-06-18  
**Purpose:** Comprehensive related-works context for the LLM confidence telemetry paper. Covers every major prior-work thread the paper engages with, organized by theme. Provide this document to any new session working on the paper draft.

---

## 0. Paper Identity (What This Paper Is)

This paper studies how well large language models can communicate their own uncertainty. It compares multiple classes of confidence signal — **logit-based**, **verbalized (self-rated)**, and **semantic entropy** — across six model families and four diverse task types (arithmetic reasoning, factual recall, commonsense reasoning, legal reasoning), using a novel **three-generation elicitation architecture** that separates reasoning, self-assessment, and blinded critique into independent forward passes.

The core contributions are:
1. A systematic, multi-signal, cross-model comparison of LLM confidence calibration on open-ended and binary benchmark tasks.
2. A three-generation pipeline (reasoning → own-work-aware self-rating → blinded critique) that separates reasoning from self-evaluation and mitigates sunk-cost inflation in verbalized confidence.
3. Empirical findings on how model family, model size, and thinking-model architecture (chain-of-thought reasoning tokens) affect calibration quality across signal types.
4. Practical engineering observations: prompt design effects on verbalized confidence, repetition-loop failure modes in base models, and the harmony-envelope parsing challenge for GPT-OSS.

---

## 1. Calibration in Neural Networks (Foundational)

### 1.1 Core Calibration Concepts

**Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). "On Calibration of Modern Neural Networks." ICML.**  
The canonical modern reference for neural network calibration. Introduces Expected Calibration Error (ECE), reliability diagrams, and the surprising finding that post-2012 deep networks are overconfident. Proposes temperature scaling as a simple post-hoc fix. Our paper uses ECE and reliability diagrams as primary calibration evaluation tools, directly building on this work's framework.

**Niculescu-Mizil, A., & Caruana, R. (2005). "Predicting good probabilities with supervised learning." ICML.**  
Shows that different classifiers are miscalibrated in different ways (SVMs are underconfident, naive Bayes overconfident) and evaluates calibration methods including isotonic regression and Platt scaling. Provides the conceptual foundation for post-hoc calibration methods that our paper implicitly contrasts against intrinsic (prompt-elicited) calibration.

**Naeini, M. P., Cooper, G., & Hauskrecht, M. (2015). "Obtaining well-calibrated probabilities using Bayesian binning into quantiles." AAAI.**  
Proposes BBQ (Bayesian Binning into Quantiles) for calibration. Alternative to ECE. We use ECE throughout but this represents the broader calibration measurement literature.

**Lakshminarayanan, B., Pritzel, A., & Blundell, C. (2017). "Simple and scalable predictive uncertainty estimation using deep ensembles." NeurIPS.**  
Demonstrates that ensembles of independently trained networks produce well-calibrated uncertainty estimates. Our paper does not use ensembles (cost-prohibitive at LLM scale), but this work establishes the calibration baseline that single-model uncertainty estimation must exceed to be useful — directly motivating why we study multiple complementary signals rather than relying on any one.

**Gal, Y., & Ghahramani, Z. (2016). "Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning." ICML.**  
Connects MC Dropout to variational Bayes for uncertainty estimation. Established the Monte Carlo sampling approach to uncertainty that later influenced semantic entropy's sampling-based design. Conceptually upstream of the SE method we implement.

### 1.2 AUROC as a Calibration / Uncertainty Quality Metric

**Hendrycks, D., & Gimpel, K. (2017). "A Baseline for Detecting Misclassified and Out-of-Distribution Examples in Neural Networks." ICLR.**  
Proposes using AUROC to measure how well a model's confidence score discriminates correct from incorrect predictions — specifically, how well the maximum softmax probability distinguishes in-distribution correct answers from misclassified examples. This is the conceptual foundation for our AUROC-based evaluation: treating each confidence signal as a binary discriminator (correct vs. wrong) and computing AUROC to measure its discriminative quality independently of calibration scale.

**Ovadia, Y., Fertig, E., Ren, J., Nado, Z., Sculley, D., Nowozin, S., ... & Ghahramani, Z. (2019). "Can you trust your model's uncertainty? Evaluating predictive uncertainty under dataset shift." NeurIPS.**  
Studies how well uncertainty estimates hold under distribution shift. Motivates why we evaluate across diverse task domains — calibration that holds on one benchmark may not transfer.

---

## 2. LLM-Specific Uncertainty and Calibration

### 2.1 Self-Knowledge and P(True)

**Kadavath, S., Conerly, T., Askell, A., Henighan, T., Ganguli, D., Hernandez, D., ... & Clark, J. (2022). "Language Models (Mostly) Know What They Know." arXiv:2207.05221.**  
One of the most directly related papers. Shows that large language models can estimate the probability of their own answers being correct when asked directly — referred to as P(True). Demonstrates that P(True) is better calibrated than raw token probabilities for open-ended generation, and that calibration scales with model size. Our single-pass verbalized confidence (Gen 2 in our three-generation pipeline) is conceptually an instantiation of P(True) elicitation, extended with explicit probability-range rubrics and applied with explicit authorship framing ("YOUR OWN reasoning chain") to reduce anchoring effects.

**Key distinction from our work:** Kadavath et al. ask models whether they know a *particular answer* — a binary know/don't-know framing. We extend this to a 10-point scale with explicit probability labels, apply it across multiple model families including thinking models with explicit chain-of-thought, and introduce a *blinded* second evaluator (Gen 3) to test whether models revise their self-assessment when the authorship bias is removed.

### 2.2 Verbalized Confidence Elicitation

**Tian, K., Mitchell, E., Zhou, A., Sharma, A., Rafailov, R., Yao, H., ... & Finn, C. (2023). "Just Ask for Calibration: Strategies for Eliciting Calibrated Confidence Scores from Language Models on Free-Form Questions." EMNLP.**  
Directly compares multiple verbalized confidence prompting strategies on free-form QA tasks: asking for a probability, asking for a linguistic hedge ("I'm certain / I think / I'm not sure"), asking for a 0–10 scale, etc. Finds that prompting models to express confidence in verbal labels (rather than numeric probabilities) often produces better-calibrated outputs. Our 10-class verbal rubric ("Almost no chance" through "Almost certain") implements this finding — the rubric grounds each number in a verbal label and an explicit probability range to reduce the arbitrary numeric interpretation.

**Mielke, S. J., Szlam, A., Dinan, E., & Boureau, Y. L. (2022). "Reducing conversational agents' overconfidence through linguistic calibration." TACL.**  
Shows that conversational agents systematically overstate certainty, and proposes training-based and prompting-based approaches to reduce this. Documents the "sunk-cost" self-confidence inflation that motivates our Gen 3 blinded critique: after producing a coherent response, models tend to endorse it highly even when it is wrong. Our three-generation design directly addresses this by inserting a blinded evaluation step.

**Lin, Z., Trivedi, S., & Sun, J. (2022). "Teaching models to express their uncertainty in words." TMLR.**  
Focuses on training language models to produce well-calibrated linguistic confidence expressions (hedges, certainty markers). Shows that fine-tuned models can learn to distinguish "I'm not sure" from "I know" more reliably. Relevant as a contrast to our zero-shot elicitation approach — we study out-of-the-box calibration using prompting alone.

**Xiong, M., Hu, Z., Lu, X., Li, Y., Fu, J., He, J., & Hooi, B. (2023). "Can LLMs Express Their Uncertainty? An Empirical Evaluation of Confidence Elicitation Methods for LLMs." ICLR 2024.**  
Comprehensive empirical study of multiple confidence elicitation methods (verbalized, chain-of-thought, consistency-based) on factual QA. Key finding: consistency-based methods (sampling multiple responses and measuring agreement) often outperform single-pass verbalized confidence. Relevant to our semantic entropy implementation and to the comparison between our confidence signals. This paper does not study thinking models or the three-generation critique architecture.

**Huang, J., Shao, H., & Chang, K. C. C. (2023). "Look Before You Leap: An Exploratory Study of Uncertainty Measurement for Large Language Models." arXiv.**  
Evaluates logit-based and verbalized uncertainty measures on open-domain QA. Finds that logit-based measures are often better calibrated than verbalized confidence on factual questions. Motivates our inclusion of both logit-based and verbalized signals as complementary rather than competing.

### 2.3 Calibration Across Model Scales and Families

**Achiam, J., Adler, S., Agarwal, S., Ahmad, L., Akkaya, I., Aleman, F. L., ... & Ziegler, D. (2023). "GPT-4 Technical Report." arXiv.**  
Reports GPT-4's calibration on various benchmarks and notes that training with RLHF shifts calibration in complex ways (typically toward overconfidence on questions the model knows). Our multi-model comparison (Qwen, Llama, Gemma, GPT-OSS) empirically examines whether similar patterns hold across the 2024–2025 generation of open-weights models.

**Jiang, Z., Xu, F. F., Gao, J., Sun, Z., Liu, Q., Dwivedi-Yu, J., ... & Neubig, G. (2021). "How Can We Know When Language Models Know? On the Calibration of Language Models for Question Answering." TACL.**  
Systematically evaluates whether LM calibration on multiple-choice QA generalizes to open-domain QA. Identifies conditions under which token probabilities are and are not reliable calibration signals. Directly motivates our inclusion of verbalized confidence alongside logit-based metrics — the paper shows that raw token probs are insufficient alone.

---

## 3. Semantic Entropy

**Kuhn, L., Gal, Y., & Farquhar, S. (2023). "Semantic Uncertainty: Linguistic Invariances for Uncertainty Estimation in Natural Language Generation." ICLR 2023.**  
**This is the primary methodology paper for our semantic entropy implementation.** Proposes computing uncertainty over semantic equivalence classes rather than over surface-form strings — clustering multiple sampled answers by bidirectional NLI entailment and computing entropy over cluster probability mass. The key insight: paraphrases of the same answer ("United States" / "the US" / "America") should not count as distinct hypotheses in the uncertainty estimate. We implement SE exactly as described (temperature 0.5 sampling, DeBERTa-large-MNLI as NLI model, bidirectional entailment threshold 0.5, N=5 samples). Semantic entropy is the only non-logit, non-verbalized signal in our study that does not require the model to explicitly rate its own confidence.

**Farquhar, S., Kossen, J., Kuhn, L., & Gal, Y. (2024). "Detecting Hallucinations in Large Language Models Using Semantic Consistency." Nature.**  
Extends semantic uncertainty to hallucination detection. Shows that models hallucinate more when their semantic entropy is high — validating that SE captures genuine model uncertainty rather than surface-form variation. Provides empirical support for SE as a calibration signal beyond academic benchmarks.

**Chen, J., Mueller, J., & Goel, K. (2023). "Quantifying Uncertainty in Answers from any Language Model and Enhancing their Trustworthiness." arXiv.**  
Proposes consistency-based uncertainty estimation using multiple samples. Related to SE but uses simpler string-match clustering. Motivates the sampling-based approach as an alternative to logit-based methods.

---

## 4. Self-Critique, Two-Pass Evaluation, and Metacognition

### 4.1 Self-Evaluation and Critique

**Saunders, W., Yeh, C., Wu, J., Bills, S., Ouyang, L., Ward, J., & Leike, J. (2022). "Self-critiquing models for assisting human evaluators." arXiv.**  
Shows that models can be prompted to critique their own outputs with reasonable quality, and that these critiques help human evaluators identify errors. Demonstrates the feasibility of using a model to evaluate its own responses — conceptually upstream of our Gen 3 blinded critique. Our design makes a key modification: Gen 3 is framed as evaluating a *different* author's work, not the model's own, to remove authorship bias.

**Madaan, A., Tandon, N., Gupta, P., Hallinan, S., Gao, L., Wiegreffe, S., ... & Clark, P. (2023). "Self-Refine: Iterative Refinement with Self-Feedback." NeurIPS.**  
Demonstrates that iterative self-feedback (generate → critique → revise) improves task performance. Our three-generation pipeline shares the multi-pass structure but differs in purpose: we do not revise the answer, we instead use the critique pass to produce a more calibrated confidence estimate. The critique is not used to improve the answer but to improve the confidence signal.

**Shinn, N., Cassano, F., Berman, E., Gopalan, A., Narasimhan, K., & Yao, S. (2023). "Reflexion: Language Agents with Verbal Reinforcement Learning." NeurIPS.**  
Uses verbal self-reflection to guide agent behavior over multiple trials. Our blinded critique (Gen 3) is a single-trial version of this reflective mechanism, applied specifically to uncertainty estimation rather than answer revision.

**McAleese, N., Trebacz, M., Mikusch, R., Lindner, D., Wu, J., Saunders, W., ... & Irving, G. (2024). "LLM critics help catch LLM bugs." arXiv.**  
Shows that a separately prompted critic LLM can catch errors that the generator missed. This is the multi-agent version of our blinded critique — we use the same model in a blinded evaluator role rather than a separate model, making the design computationally tractable while still testing whether "blinding" the model to its own authorship changes its assessment.

### 4.2 P(True) and Binary Self-Judgment

The `more_likely_than_not` signal we collect (binary Yes/No judgment on whether the answer is more likely correct than not) is directly related to the P(True) framework in Kadavath et al. (2022). Specifically:
- **Single-pass `more_likely_than_not`** (Gen 2): the model's own judgment of its answer using the "YOUR OWN reasoning" framing
- **Blinded `more_likely_than_not`** (Gen 3): a pseudo-external evaluator's judgment of the same answer without knowing the model wrote it

The comparison between these two binary signals tests whether removing authorship bias shifts the model's probabilistic judgment — a direct empirical test of sunk-cost effects in LLM self-evaluation.

---

## 5. Chain-of-Thought Reasoning and Thinking Models

### 5.1 Chain-of-Thought Prompting

**Wei, J., Wang, X., Schuurmans, D., Bosma, M., Ichter, B., Xia, F., ... & Zhou, D. (2022). "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models." NeurIPS.**  
Demonstrates that prompting large models to reason step-by-step before answering substantially improves performance on reasoning tasks. Our study is partially motivated by the question: does step-by-step reasoning also improve *calibration*, not just accuracy? Do models that reason more carefully also become better at knowing when they're right?

**Wang, X., Wei, J., Schuurmans, D., Le, Q. V., Chi, E. H., Narang, S., ... & Zhou, D. (2022). "Self-Consistency Improves Chain of Thought Reasoning in Language Models." ICLR 2023.**  
Shows that sampling multiple reasoning chains and taking the majority-vote answer substantially improves accuracy. This is related to semantic entropy's sampling approach — both use multiple samples to reduce variance. Self-consistency uses the agreement among samples as an implicit confidence signal (high agreement = high confidence), but unlike SE does not compute an explicit entropy.

### 5.2 Thinking / Reasoning Models

**OpenAI. (2024). "Learning to Reason with LLMs." OpenAI Blog.**  
Introduces the o1 series of models that use "chain-of-thought" internally before producing an output. Directly relevant to our treatment of Qwen3, Gemma4-instruct, and GPT-OSS — all of which emit reasoning tokens (`<think>...</think>` blocks or harmony analysis channels) before committing to an answer. Our three-generation pipeline was specifically designed to handle thinking models where the reasoning pass cannot include confidence elicitation prompts (token budget constraints and sunk-cost bias in single-pass confidence elicitation are both exacerbated by long thinking blocks).

**DeepSeek-AI. (2025). "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning." arXiv.**  
Demonstrates that reasoning-model behavior (extended thinking chains before answering) can be achieved via RL training. Documents the `<think>...</think>` convention that Qwen3 and Gemma4 inherit. Contextualizes the thinking-model generation patterns our pipeline must parse and handle.

**Qwen Team. (2025). "Qwen3 Technical Report." arXiv.**  
Technical description of the Qwen3 model family, including the MoE architecture (Qwen3-30B-A3B has 30B total / ~3B active parameters), thinking mode, and `<think>` token conventions. Direct model reference for our primary reasoning model.

**Google DeepMind. (2024). "Gemma 2: Improving Open Language Models at a Practical Size." arXiv.**  
Technical description of Gemma-2-9B-IT, one of our standard-flow (non-thinking) baseline models.

**Google DeepMind. (2025). "Gemma 4 Technical Report."**  
Technical description of Gemma-4-31B-IT, our primary large-scale reasoning model from the Google family. Documents the `<think>` token convention for Gemma 4's thinking mode.

**Meta AI. (2024). "The Llama 3 Herd of Models." arXiv.**  
Technical description of the Llama-3.1 model family. Covers Llama-3.1-8B-Instruct, our compact instruct baseline. Documents training methodology, chat template format, and evaluation benchmarks. Contextualizes why Llama-3.1-8B-base behaves differently from instruct: the base model has no instruction-following training, making confidence elicitation via instruction-style prompts unreliable.

**Qwen Team. (2024). "Qwen2.5 Technical Report." arXiv.**  
Technical description of Qwen2.5-7B-Instruct, our compact Qwen baseline.

---

## 6. Hallucination Detection (Related Upstream)

**Manakul, P., Liusie, A., & Gales, M. J. (2023). "SelfCheckGPT: Zero-resource Black-Box Hallucination Detection for Generative Large Language Models." EMNLP.**  
Proposes detecting hallucinations by checking whether multiple sampled responses from the same model agree with each other — high disagreement signals uncertainty/hallucination. Conceptually similar to semantic entropy (both use sampling to estimate uncertainty), but operates on sentence-level consistency rather than full-response NLI clustering. Our SE implementation is a more principled version of this sampling-based idea with semantic clustering.

**Min, S., Krishna, K., Lyu, X., Lewis, M., Yih, W. T., Koh, P. W., ... & Hajishirzi, H. (2023). "FActScoring: Fine-grained Atomic Evaluation of Factual Precision in Long Form Text Generation." EMNLP.**  
Proposes atomic fact-level factuality scoring. Relevant background for why token-level calibration on short-answer tasks (our setup) is cleaner than on long-form generation where individual factual claims may be correct or wrong independently.

---

## 7. Benchmarks Used

### 7.1 GSM8K (Arithmetic Reasoning)

**Cobbe, K., Kosaraju, V., Bavarian, M., Chen, M., Jun, H., Kaiser, L., ... & Schulman, J. (2021). "Training Verifiers to Solve Math Word Problems." arXiv.**  
Introduces GSM8K — 8,500 grade-school math word problems requiring multi-step reasoning and arithmetic. Ground truth is a single numeric answer (after stripping commas/whitespace). We use the GSM8K test split. The near-deterministic correct answer makes it ideal for calibration studies: a well-calibrated model should be highly confident when it answers correctly (which it does frequently at 95%+ accuracy on current models) and less confident on the rare errors. The relatively high accuracy means the calibration analysis is dominated by correct-high-confidence rows, with errors concentrated at lower confidence levels.

### 7.2 StrategyQA (Commonsense Reasoning)

**Geva, M., Khashabi, D., Segal, E., Khot, T., Roth, D., & Berant, J. (2021). "Did Aristotle Use a Laptop? A Question Answering Benchmark with Implicit Reasoning Strategies." TACL.**  
Introduces StrategyQA — 2,780 Yes/No questions requiring implicit multi-step reasoning over world knowledge (e.g., "Was Aristotle alive before the invention of the printing press?"). Binary Yes/No format supports confusion-matrix analysis and Yes/No bias measurement. We observe a systematic Yes-bias in models on this benchmark (tendency to answer Yes more than warranted), which creates calibration asymmetries between Yes and No questions. This benchmark is particularly valuable for studying whether models recognize their own uncertainty on questions that require implicit knowledge retrieval and multi-hop inference.

### 7.3 TriviaQA (Factual Recall)

**Joshi, M., Choi, E., Weld, D. S., & Zettlemoyer, L. (2017). "TriviaQA: A Reading Comprehension Dataset Containing 650K Question-Answer Pairs From Wikipedia and the Web." ACL.**  
Introduces TriviaQA — a large-scale open-domain trivia QA dataset with multiple valid answer aliases. We use the `rc.nocontext` configuration (no supporting context; models must answer from parametric memory) from the validation split, loaded via `mandarjoshi/trivia_qa` on HuggingFace. The alias-based correctness evaluation (`answer.normalized_aliases`) handles paraphrase variability. TriviaQA is the benchmark where base-model repetition loops were most prevalent (16% loop rate for GPT-OSS), making it the primary stress test for our anti-loop decoding guards.

### 7.4 LegalBench (Legal Reasoning)

**Guha, N., Nyarko, J., Ho, D. E., Ré, C., Chilton, A., Chohlas-Wood, A., ... & Kreiman, G. (2023). "LegalBench: A Collaboratively Built Benchmark for Measuring Legal Reasoning in Large Language Models." NeurIPS.**  
Introduces LegalBench — 162 legal reasoning tasks built collaboratively by lawyers and NLP researchers. We use two subtasks: `hearsay` (evidentiary rules for what constitutes hearsay) and `consumer_contracts_qa` (consumer contract interpretation), both requiring binary Yes/No answers. The two-subtask structure enables cross-subtask calibration comparison within a single expert domain. Legal reasoning is particularly interesting for calibration because it combines factual recall (rules) with case-specific application (does this situation match the rule?), creating genuine ambiguity that a well-calibrated model should express.

---

## 8. Decoding and Generation Quality

**Holtzman, A., Buys, J., Du, L., Forbes, M., & Choi, Y. (2020). "The Curious Case of Neural Text Degeneration." ICLR.**  
Identifies repetition and incoherence as failure modes of greedy/beam-search decoding and proposes nucleus (top-p) sampling as a fix. Contextualizes why base models under greedy decoding degenerate into repetition loops — directly related to our use of `no_repeat_ngram_size=3` and stop-string guards for base models and GPT-OSS. We deliberately use greedy decoding for the main evaluation pass (for reproducibility and logit-metric consistency) but add minimal anti-loop guards to prevent pathological failure modes.

---

## 9. Logit-Based Uncertainty in Language Models

**Malinin, A., & Gales, M. (2021). "Uncertainty Estimation in Autoregressive Structured Prediction." ICLR.**  
Proposes methods for uncertainty estimation in sequential prediction, connecting per-step log-probabilities to sequence-level uncertainty. Provides theoretical grounding for the `seq_confidence_mean` signal (mean log-probability of generated tokens) as an uncertainty estimator. This signal is the primary logit-based confidence metric in our study.

**Fomicheva, M., Sun, S., Yankovskaya, L., Blain, F., Guzmán, F., Fishel, M., ... & Specia, L. (2020). "Unsupervised Quality Estimation for Neural Machine Translation." TACL.**  
Shows that per-token probability scores can be aggregated (geometric mean, arithmetic mean, minimum) to produce sentence-level uncertainty estimates for MT. Our `logit_confidence_geom`, `logit_confidence_mean_prob`, and `logit_confidence_min` signals are direct instantiations of the aggregation strategies studied here, applied to confidence estimation rather than MT quality estimation.

---

## 10. Key Methodological Contrasts (What Makes Our Work Novel)

This section synthesizes how our approach differs from each prior-work thread.

| Prior Work | What They Do | Our Extension / Difference |
|---|---|---|
| Kadavath et al. (2022) | Binary P(True) self-evaluation | 10-point scale rubric; explicit authorship framing; blinded Gen 3 critique to test sunk-cost effects |
| Tian et al. (2023) | Compare verbalized confidence strategies on free-form QA | Multi-signal comparison (logit + verbalized + SE); multi-model; thinking models |
| Kuhn et al. (2023) | Semantic entropy for open-ended QA | Implement SE alongside logit and verbalized signals for side-by-side comparison; apply to binary and numeric tasks |
| Saunders et al. (2022) / Self-Refine | Self-critique to improve answers | Critique used only for confidence re-rating (Gen 3), not for answer revision; blinding is the key design |
| Xiong et al. (2023) | Empirical comparison of elicitation methods | Include thinking models (Qwen3, Gemma4, GPT-OSS) and their unique three-gen architecture; base vs. instruct comparison |
| Guo et al. (2017) | Calibration measurement with ECE | Apply ECE framework to LLM verbalized + logit + SE signals simultaneously |
| Mielke et al. (2022) | Reduce overconfidence via training | Prompting-only approach (zero-shot); test whether multi-pass critique achieves similar recalibration without fine-tuning |

---

## 11. Datasets and Models — Citation Checklist for Paper

### Datasets to Cite
- [ ] Cobbe et al. (2021) for GSM8K
- [ ] Geva et al. (2021) for StrategyQA
- [ ] Joshi et al. (2017) for TriviaQA
- [ ] Guha et al. (2023) for LegalBench

### Models to Cite
- [ ] Qwen Team (2024) for Qwen2.5
- [ ] Qwen Team (2025) for Qwen3
- [ ] Meta AI (2024) "Llama 3 Herd" for Llama-3.1
- [ ] Google DeepMind (2024) for Gemma 2
- [ ] Google DeepMind (2025) for Gemma 4
- [ ] OpenAI for GPT-OSS-20B (technical report / model card)

### Confidence Signals to Cite
- [ ] Kuhn et al. (2023) for semantic entropy methodology (DeBERTa-MNLI clustering, bidirectional entailment)
- [ ] Kadavath et al. (2022) for P(True) / self-evaluation conceptual framing
- [ ] Guo et al. (2017) for ECE and calibration measurement
- [ ] Hendrycks & Gimpel (2017) for AUROC as discrimination metric
- [ ] Malinin & Gales (2021) for seq log-prob uncertainty interpretation
- [ ] Fomicheva et al. (2020) for geometric/arithmetic mean aggregation strategies

### Architecture to Cite
- [ ] Saunders et al. (2022) or Madaan et al. (2023) for self-critique / multi-pass evaluation paradigm
- [ ] Wei et al. (2022) for chain-of-thought background (thinking models context)
- [ ] OpenAI o1 / DeepSeek-R1 for thinking-model behavioral context

---

## 12. Important Empirical Findings from Our Data (For the Paper)

These are the most significant calibration findings observed in the HTML visualizations and CSVs, which should anchor the results section.

### 12.1 Model Accuracy vs. Confidence Alignment

- **Gemma4-31B-instruct GSM8K:** ~95.9% accuracy, high verbalized confidence (predominantly 8–10). "Almost always right, *always* says so" — potential overconfidence pattern, but genuine accuracy backs it up.
- **Qwen3 GSM8K:** ~95.2% accuracy; calibration curves should show tight alignment between confidence bucket and accuracy for the 8–10 range.
- **Qwen3 StrategyQA:** ~74.6% accuracy with a systematic Yes-bias — the model answers Yes more frequently than warranted, and Yes-answers are better calibrated than No-answers. The confusion matrix asymmetry (more Yes→Wrong than No→Wrong) is a key finding.
- **Gemma4-31B-instruct StrategyQA:** ~85.7% accuracy; comparison with Qwen3 on the same benchmark shows cross-model calibration differences at similar accuracy levels.
- **Qwen3 LegalBench (combined):** ~65.5% accuracy across two subtasks, revealing expert-domain uncertainty and cross-subtask calibration asymmetry.
- **Qwen3 TriviaQA:** ~77.3% accuracy; factual recall shows a different confidence distribution shape than reasoning tasks (less peaked, more mid-range confidence).

### 12.2 Signal Comparison Patterns

From the correlation matrix (Qwen, cross-benchmark):
- `answer_token_entropy` and `chosen_answer_raw_prob` (MCQ-only, now archived) showed stronger rank-correlation with correctness under Spearman/Kendall than Pearson — suggesting non-linear relationship.
- Logit signals (`seq_confidence_mean`, `logit_confidence_geom`) tend to correlate with each other but may diverge from verbalized confidence on hard questions where the model generates a confident-sounding response but with low token probabilities.
- The `verbalized_confidence` (Gen 3 blinded) vs. `single_pass_confidence` (Gen 2 own-work) comparison tests whether blinding shifts scores down (expected if Gen 2 is inflated by sunk-cost).

### 12.3 Thinking Model Behavior

- **Three-generation architecture motivation:** Thinking models emitting `<think>` blocks consume 1,000–3,000+ tokens before structured output. Single-pass confidence elicitation fails because the token budget is exhausted. The three-generation design is not optional for these models — it is a technical necessity as well as a methodological improvement.
- **Gen 3 blinded critique:** Provides a second confidence estimate that has not been anchored by the model's own reasoning chain. Preliminary analysis suggests Gen 3 confidence is more conservative than Gen 2 on questions where the model's reasoning was hedging.

### 12.4 Base vs. Instruct Comparison

- Base models cannot reliably follow instruction-style confidence prompts. Our design includes a tiered prompt strategy: minimal Q&A format for base models, instruction-style for instruct.
- The logit-based signals (available for both variants) enable comparison even when verbalized signals are unreliable. This comparison is the primary use of base-model data in the paper.

---

## 13. Narrative Frames for Each Benchmark Section

Based on the editorial voice established in the HTML pages (§16 of confidence_telemetry_handoff.md), each benchmark section in the paper should lead with a specific data finding:

- **GSM8K:** The accuracy is near-ceiling (95%+), so calibration on *errors* is the story — do models know when they've made a mistake on a problem they usually get right?
- **StrategyQA:** The Yes-bias and its calibration asymmetry. Models are better calibrated when they say "No" than when they say "Yes," suggesting systematic overconfidence on affirmative answers.
- **TriviaQA:** The factual recall frontier — where models know they know vs. where they're guessing. High semantic entropy rows are the interesting cases.
- **LegalBench:** Cross-subtask calibration. The hearsay and consumer-contracts subtasks require different reasoning strategies (rule application vs. language interpretation); do models recognize this difference in their confidence?

---

## 14. Open Questions for the Discussion Section

1. **Why does the blinded critique (Gen 3) change confidence estimates?** The mechanism (authorship bias removal) is argued theoretically; empirical evidence from Gen 2 vs. Gen 3 delta distributions is the test.
2. **Does semantic entropy actually track correctness better than logit signals?** Kuhn et al. (2023) say yes on open-ended QA; our cross-benchmark data tests whether this holds on binary (StrategyQA, LegalBench) and numeric (GSM8K) tasks.
3. **Do thinking models calibrate better or worse?** The hypothesis (longer reasoning → better uncertainty tracking) is not obvious — more reasoning could also lead to more confident (but overconfident) commitment to wrong answers.
4. **How stable are verbalized confidence scores across model families?** Is a "7" from Qwen3 semantically equivalent to a "7" from GPT-OSS? The logit-based signals provide an anchor for cross-model comparison since they are on an objective log-probability scale.
5. **What is the practical implication for model selection?** Given a task type (arithmetic vs. factual vs. legal), which model's confidence signals can be most trusted for downstream decisions (e.g., routing hard questions to human review)?

---

*End of Results_Draft handoff document.*
