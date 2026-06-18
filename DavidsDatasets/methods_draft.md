# Methods Draft — LLM Confidence Telemetry Study

---

## 1. Datasets

We evaluate across four benchmark datasets spanning arithmetic reasoning, factual recall, commonsense reasoning, and legal question answering. Each dataset was selected to provide a distinct task type and answer format, allowing confidence signals to be assessed across a wide range of difficulty and domain.

| Dataset | Task Type | Answer Format | Split Used | Reason for Inclusion |
|---------|-----------|--------------|------------|----------------------|
| **GSM8K** | Arithmetic word problems | Open-ended numeric (integer or decimal) | Test | Grade-school math provides near-deterministic ground truth with graded difficulty; widely used calibration benchmark |
| **StrategyQA** | Multi-hop commonsense reasoning | Binary Yes/No | Test | Tests implicit decomposition; binary output supports confusion-matrix analysis and Yes/No bias measurement |
| **TriviaQA** | Open-ended factual recall | Free-form string | Validation | Tests recall breadth; alias-aware evaluation handles paraphrase variability |
| **LegalBench** | Legal reasoning (hearsay, consumer contracts) | Binary Yes/No | Test | Expert-knowledge domain; binary format with two distinct subtasks (hearsay rules, consumer contracts) enables cross-subtask calibration comparison |

### Dataset Notes

- **LegalBench** is drawn from two subtasks (`hearsay` and `consumer_contracts_qa`) pooled into a single evaluation cluster with subset labels to disambiguate idx collisions between subtasks.
- Sample counts vary by model and run; typical run sizes range from 40–200 items per (model × dataset) cell, accumulated across multiple random seeds and combined by deduplication on `idx`.
- *Previously included datasets (no longer in active use): MMLU-Pro (10-option MCQ) and MedQA/USMLE (4-option MCQ). These were removed from the study scope; their associated MCQ-only confidence signals (answer-token entropy, per-letter probabilities) are not reported.*

---

## 2. Models

The study focuses exclusively on instruct-tuned variants. All models are evaluated in greedy decoding mode (temperature = 1.0, `do_sample = False`) for the main generation pass.

| Model | HuggingFace ID | Parameter Count | Reasoning Capability | Reason for Inclusion |
|-------|---------------|-----------------|---------------------|----------------------|
| **Qwen2.5-7B-Instruct** | `Qwen/Qwen2.5-7B-Instruct` | 7B | Standard (no chain-of-thought scaffolding) | Compact, well-calibrated baseline; representative of mid-size instruction-tuned models |
| **Qwen3-30B-A3B** | `Qwen/Qwen3-30B-A3B` | 30B total / ~3B active (MoE) | Thinking model — emits `<think>…</think>` chain-of-thought block before committing to an answer | Large-scale reasoning model; primary test case for the three-generation architecture |
| **Llama-3.1-8B-Instruct** | `meta-llama/Llama-3.1-8B-Instruct` | 8B | Standard | Widely used open-weights baseline; enables comparison with Qwen at similar scale |
| **Gemma-2-9B-IT** | `google/gemma-2-9b-it` | 9B | Standard | Google's mid-size instruct model; provides a third family for cross-model comparison |
| **Gemma-4-31B-IT** | `google/gemma-4-31b-it` | 31B | Thinking model (`<think>…</think>`) | Large-scale Google reasoning model; tested in the three-generation pipeline |
| **GPT-OSS-20B-Instruct** | `openai/gpt-oss-20b` | 20B | Uses OpenAI harmony channel format: `analysis<reasoning>assistantfinal<final answer>` (no `<think>` tags) | OpenAI open-weights model; tests generalization of the confidence pipeline to a non-Transformers-standard output format |

*Previously included model (no longer in active use): Llama-4-Scout-17B-16E-Instruct (`meta-llama/Llama-4-Scout-17B-16E-Instruct`) — removed from the study scope.*

### Model Loading

Large models (Qwen3, Gemma4, GPT-OSS) are loaded with `device_map="auto"` for multi-GPU sharding. Qwen3 and GPT-OSS use `torch.bfloat16`; other models use `torch.float16`. All models are loaded from HuggingFace Hub.

---

## 3. Confidence Signals

Seven families of confidence signal are collected for each sample.

### 3.1 Logit-Based Confidence

All four logit metrics are computed from the token probabilities of the **main generation pass** (Gen 1 for reasoning-flow models; the single forward pass for standard models). Because the main pass uses greedy decoding, `outputs.scores` provides per-step vocabulary distributions.

For models that require repetition guards (base models and GPT-OSS), a clean teacher-forced forward pass over `outputs.sequences` is used to re-derive unwarped token probabilities and raw logit scores, so that the guard does not perturb the confidence metrics.

| Signal | Formula | Notes |
|--------|---------|-------|
| **`seq_confidence_mean`** | Mean log-probability of generated tokens: $\frac{1}{T}\sum_{t=1}^T \log p(x_t)$ | Length-normalized; negative values (less negative = higher confidence); primary logit-based signal |
| **`logit_confidence_geom`** | Geometric mean of per-token probabilities: $\left(\prod_{t=1}^T p(x_t)\right)^{1/T}$ | Equivalent to $\exp(\text{seq\_confidence\_mean})$; range (0, 1] |
| **`logit_confidence_mean_prob`** | Arithmetic mean of per-token probabilities: $\frac{1}{T}\sum_{t=1}^T p(x_t)$ | More sensitive to occasional low-probability tokens than geometric mean |
| **`logit_confidence_min`** | Minimum per-token probability: $\min_t p(x_t)$ | Captures the weakest link in the generation chain |

### 3.2 Answer Token Entropy (MCQ Only — Not Reported in This Study)

> **Note:** Answer Token Entropy was implemented for the multiple-choice datasets (MMLU-Pro and MedQA) that were previously in scope. Since both MCQ datasets have been removed from the study, this signal is not included in the reported results. The implementation remains in the pipeline for potential future use.

~~For multiple-choice datasets (MMLU-Pro, MedQA), a dedicated signal captures the model's uncertainty at the answer-commitment step specifically.~~

~~After extracting the `Answer:` token position in the generated sequence, the raw logit vector at that position (shape `[vocab_size]`) is used to compute the probability distribution over answer letters (A through J for MMLU-Pro; A through D for MedQA). Shannon entropy is then computed over that letter distribution:~~

$$H_{\text{ATE}} = -\sum_{k \in \{A,\ldots,J\}} p_k \log p_k$$

~~Low entropy indicates a peaked distribution (the model strongly preferred one letter); high entropy indicates near-uniform uncertainty across options.~~

~~Additional columns derived from the same logit snapshot: `chosen_answer_raw_prob`, `top_answer_letter`, and `prob_A` through `prob_J`.~~

### 3.3 Verbalized Confidence (1–10 Scale)

The model is asked to rate its own confidence on a 10-class scale with explicit probability ranges. The rubric, applied uniformly across all four active datasets (GSM8K, StrategyQA, TriviaQA, LegalBench) after harmonization on 2026-05-10, reads:

```
- 1  = "Almost no chance"  (0–10% likely correct)
- 2  = "Highly unlikely"   (10–20% likely correct)
- 3  = "Chances are slight" (20–30% likely correct)
- 4  = "Unlikely"          (30–40% likely correct)
- 5  = "Less than even"    (40–50% likely correct)
- 6  = "Better than even"  (50–60% likely correct)
- 7  = "Likely"            (60–70% likely correct)
- 8  = "Very good chance"  (70–80% likely correct)
- 9  = "Highly likely"     (80–90% likely correct)
- 10 = "Almost certain"    (90–100% likely correct)
```

Each prompt also includes a class-matching reminder to prevent label/number mismatches:

> "The Confidence number MUST match the class you selected — for example, if you select 'Better than even' you MUST write Confidence: 6, not any other number."

Verbalized confidence is extracted from the model's output via a three-priority regex pattern (`Confidence: N`, `Confidence N` without colon, or prose form `"confidence is N"`).

The value stored in **`verbalized_confidence`** is the primary verbalized signal. For standard-flow models it is sourced from the Gen-2 call (see §5). For reasoning-flow models it is the Gen-3 blinded critique output with fallback to Gen-2 if Gen-3 extraction fails.

### 3.4 More Likely Than Not (Binary)

Each confidence elicitation prompt also asks the model to make a binary probabilistic judgment:

> "State if you think your answer is more likely correct than not after 'Correct:' (Yes or No)."

This is extracted from the `Correct: Yes/No` line. The **`more_likely_than_not`** column stores this as a boolean. It represents a calibration-relevant threshold signal: a well-calibrated model should answer "Yes" on approximately 50%+ of items where it is actually correct.

### 3.5 Semantic Entropy

Semantic entropy (SE) follows Kuhn et al. (2023) "Semantic Uncertainty." N answers are sampled at temperature 0.5 from the main prompt. Each sampled response is parsed for its answer string. Semantically equivalent answers are clustered using a DeBERTa-large-MNLI model via bidirectional entailment: two answers are in the same cluster iff both A→B and B→A entailment probabilities exceed a threshold of 0.5.

SE is then computed over cluster probability mass:

$$\text{SE}(x) = -\sum_c p(c \mid x) \log p(c \mid x)$$

where $p(c \mid x) = \sum_{s \in c} p(s \mid x)$ is the length-normalized log-probability mass aggregated within cluster $c$.

Additional SE-related outputs: `predictive_entropy` (token-count-based Shannon entropy over the discrete answer distribution, without NLI clustering), `num_clusters`, `cluster_sizes`, `sampled_answers`, `se_extraction_failure_rate`.

**Current status:** `SE_NUM_SAMPLES = 1` and `SKIP_NLI_CLUSTERING = True` are set for speed during active development. Both must be restored to `SE_NUM_SAMPLES = 5` and `SKIP_NLI_CLUSTERING = False` before any SE-dependent analysis.

---

## 4. Three-Generation Architecture (Reasoning-Flow Models)

For thinking models — those that emit a reasoning block before committing to an answer (Qwen3, GPT-OSS, Gemma4-instruct) — a three-generation pipeline separates reasoning, self-assessment, and blinded critique into distinct forward passes. This architecture is controlled by the `USE_REASONING_FLOW` flag in `config.py`.

The key motivation is that a single-generation flow forces the model to rate its own confidence immediately after generating its own coherent reasoning chain, which produces inflated scores due to a sunk-cost effect. Additionally, thinking models' `<think>` blocks can consume 1,000–3,000+ tokens before any structured output is emitted, exhausting token budgets when confidence elicitation is attempted in the same pass.

### Gen 1 — Reasoning and Answer Only

- **Prompt content:** Task-specific question with answer-format instructions. The confidence rubric and `Confidence:/Correct:` output requirements are entirely absent (`include_confidence=False`).
- **Token budget:** 8,192 tokens (reasoning-model budget; thinking chains on hard MMLU-Pro math/physics problems regularly exceed 4,096 tokens).
- **Output:** The model produces its reasoning chain and commits to a final answer (`Answer: X`).
- **Post-processing:** The `<think>…</think>` block is stripped before answer extraction. For GPT-OSS, the harmony envelope (`analysis…assistantfinal`) is stripped to isolate the committed final channel. The stripped response and its token-level probabilities are used for all logit-based metrics and answer-token entropy.
- **Stored in:** `full_response` (think block stripped), `seq_confidence_mean`, `logit_confidence_*`, `answer_token_entropy`, `prob_*` columns.

### Gen 2 — Own-Work-Aware Verbalized Confidence

- **Prompt content:** "The following is YOUR OWN reasoning chain and final answer that YOU previously produced. Based on YOUR reasoning, how confident are you that YOUR answer is correct?" The full 10-class rubric is included. The reasoning is trimmed to 3,000 characters to fit comfortably in a short generation.
- **Token budget:** 512 tokens (tight budget dedicated solely to the confidence rating; cannot be truncated before emitting `Confidence: N`).
- **Design rationale:** Explicit first-person authorship framing activates self-reflection. The model reads its own reasoning as text (not from internal activations — a known limitation; see §4.1). Because Gen 2 is a separate forward pass, the confidence call can always complete within its budget.
- **Outputs:** `single_pass_confidence` (1–10 integer), `single_pass_correct` (boolean).
- **Fallback for non-reasoning-flow models:** Standard-flow models (Qwen2.5, Llama-3.1-8B, Gemma-2-9B) also receive the Gen-2 own-work framing from session 2026-06-08 onward. The response footer (`Confidence: N\nCorrect: Yes/No`) is stripped from the main pass before passing reasoning to Gen 2, so the confidence rating is independent.

### Gen 3 — Blinded Two-Pass Critique

- **Prompt content:** "You are reviewing a solution submitted by **someone else**…" — the model is not told this is its own work. The Gen-2 self-reported score is provided as context so the external reviewer can push back on it: "The respondent self-assessed: Confidence X/10, More likely correct: Yes/No." The reasoning is trimmed to 2,000 characters (blinded reviewer receives a summary rather than the full chain). The same 10-class rubric is included.
- **Token budget:** 4,096 tokens (accommodates `<think>` blocks on the critique pass for reasoning models; thinking is disabled via a `[THINK OFF]` instruction or model-specific mechanism when `TWO_PASS_DISABLE_THINKING = True`).
- **Design rationale:** Blinding removes authorship bias. Providing the Gen-2 score gives the critic a baseline to either endorse or push back against, approximating a second-opinion review.
- **Outputs:** `two_pass_critique` (raw text), `verbalized_confidence` (1–10, primary output), `more_likely_than_not` (boolean, primary output).
- **Fallback:** If Gen-3 extraction fails (empty response or no parseable `Confidence:` line), `verbalized_confidence` and `more_likely_than_not` fall back to Gen-2 values.

#### 4.1 Acknowledged Limitation: Internal State Abstraction

Gen 2 reads the model's own reasoning as text in a fresh forward pass, rather than from the internal hidden states that were active during Gen 1. Subtle uncertainty signals embedded in activations at generation time are not accessible retrospectively. The logit-based metrics (`seq_confidence_mean`, `logit_confidence_*`) partially compensate by capturing internal-state signals during Gen 1. The combination — logit metrics (internal, during generation) + Gen-2 verbalized confidence (semantic retrospective, separate pass) + Gen-3 blinded critique (external evaluator perspective) — is designed to provide three complementary, non-redundant signal types.

---

## 5. Confidence Rubric

A single, uniform rubric is applied across all four active datasets (GSM8K, StrategyQA, TriviaQA, LegalBench) and all three confidence elicitation contexts (Gen 1 single-pass prompt, Gen 2 own-work prompt, Gen 3 blinded critique). The rubric format uses a bulleted list with leading index, verbal label, and explicit probability range:

```
- 1  = "Almost no chance"   (0–10% likely correct)
- 2  = "Highly unlikely"    (10–20% likely correct)
- 3  = "Chances are slight" (20–30% likely correct)
- 4  = "Unlikely"           (30–40% likely correct)
- 5  = "Less than even"     (40–50% likely correct)
- 6  = "Better than even"   (50–60% likely correct)
- 7  = "Likely"             (60–70% likely correct)
- 8  = "Very good chance"   (70–80% likely correct)
- 9  = "Highly likely"      (80–90% likely correct)
- 10 = "Almost certain"     (90–100% likely correct)
```

This uniform format was finalized on 2026-05-10. Prior to that date, different datasets used different rubric formats (MMLU-Pro had no rubric at all; StrategyQA/TriviaQA used a different probability scale). All data produced before rubric harmonization should be treated as a distinct distribution and excluded from cross-dataset calibration comparisons.

---

## 6. Think Block Handling

Reasoning-flow models emit a `<think>…</think>` block containing their chain-of-thought before committing to a structured answer. Two handling decisions apply:

**Stripping before extraction:** A compiled regex `_QWEN3_THINK_RE = re.compile(r'<think>.*?</think>', re.DOTALL)` strips the think block from the response before answer and confidence extraction. This ensures that `Answer:` lines inside the reasoning block do not shadow the committed answer.

**GPT-OSS harmony format:** GPT-OSS uses a different convention: `analysis<reasoning>assistantfinal<final answer>`. The delimiter `"assistantfinal"` (no separator between the keyword and the following text) separates the analysis channel from the committed output. A helper `_strip_harmony_envelope(response)` applies `rsplit("assistantfinal", 1)[-1]` to extract the final channel. The pre-`assistantfinal` content is extracted as `reasoning_for_critique` (with the leading `analysis` channel marker trimmed) and passed to Gen 3 in place of the think block.

**Two-pass critique:** For reasoning-flow models, thinking is disabled on the Gen-3 critique pass via `TWO_PASS_DISABLE_THINKING = True` to prevent the think block from consuming the entire critique budget before `Confidence:` and `Correct:` lines are emitted.

---

## 7. Truncation Detection and Forced-Answer Recovery

### 7.1 Truncation Detection

`generate_with_logits` returns a 5-tuple `(text, token_probs, tokens, raw_scores, meta)` where `meta` is a dict produced by `_detect_truncation`:

- `finish_reason`: `"eos"` (EOS token reached), `"stop"` (stop string fired before budget), or `"length"` (hit `max_new_tokens` without EOS or stop string)
- `was_truncated`: `True` when `finish_reason == "length"`, or when a two-pass critique response is non-empty but structurally incomplete (missing `Confidence:` or `Correct:` markers)

Empty-EOS responses (a base model emitting EOS immediately on an out-of-distribution instruction prompt) are explicitly **not** classified as truncated, to avoid the contradictory combination `(finish_reason=eos, was_truncated=True)`.

### 7.2 Forced-Answer Fallback

Hard questions on reasoning-flow models regularly exhaust the 8,192-token budget before the model emits `Answer: X`. Rather than dropping these rows (which would introduce non-random dropout biasing calibration toward easy samples), the pipeline attempts a forced-answer recovery.

**Trigger condition:** `extract_model_answer_strict` (Priority-1 only; requires an explicit `Answer:` line anchored at line start) fails on the main response.

**Forced-answer call:** `get_forced_answer` is called with:
- **Instruct models:** An instruction-style prompt including up to 3,000 characters of the (possibly truncated) reasoning and a directive to commit to a single answer. Token budget is 8–32 tokens depending on dataset (8 for letter/Yes-No, 16 for GSM8K numbers, 32 for TriviaQA free-form).
- **Base models:** A minimal Q&A format (`Q: {question}\nA:`) matching the pretraining distribution, omitting the instruction framing that causes base models to emit EOS.

**Result columns:** `was_forced` (boolean — `True` iff the main response had no clean `Answer:` line), `forced_answer_response` (raw text of the forced call).

### 7.3 Preservation of Difficult Examples

Truncated and forced-answer rows are **retained** in the calibration dataset rather than excluded. The rationale:

1. Calibration analysis requires paired (answer, confidence) triples on every sample.
2. Dropout is non-random — only hard questions truncate — which would bias calibration curves toward easy items.
3. A forced answer is itself a calibration signal: a well-calibrated model should produce low verbalized confidence on forced guesses.

Partitioning rows by `was_forced` supports downstream analysis of "honest" vs. forced accuracy separately.

---

## 8. Post-Processing and Answer Extraction

### 8.1 Dataset-Specific Extraction Rules

Answer extraction applies a priority-ordered regex cascade in `data_utils.py::extract_model_answer`. All extractors apply harmony-envelope stripping and first-block truncation before pattern matching.

**First-block truncation** (`_truncate_to_first_block`): Cuts the response after the first `Correct: Yes/No` line or first restart marker (e.g. a second `Question:` block). This prevents base-model over-generation from causing the last-match parser to return an answer from a continuation block rather than the original response.

**Harmony stripping** (`_strip_harmony_envelope`): For GPT-OSS responses containing `"assistantfinal"`, takes the post-delimiter slice before running any pattern.

#### Multiple-Choice Parsing (MCQ — Not Active; MMLU-Pro and MedQA Removed)

> *The pipeline retains MCQ parsing logic for completeness but it is not exercised in the current active dataset set.*

~~Priority 1 (strict): `re.findall` with a start-of-line anchor `(?m)^[^a-zA-Z\n]*[Aa]nswer[^a-zA-Z\n]*:\s*([A-J])`, taking the **last** match.~~

~~Priority 2: `re.findall` with a standalone letter pattern `(?m)^[^a-zA-Z\n]*([A-J])\s*$`, last match.~~

#### Open-Ended Answer Normalization

**GSM8K:** Priority 1 (strict `Answer:` line, last match), Priority 2 (`The answer is: X` form). No Priority-3 last-number fallback — rows where no answer was committed receive `model_answer=None` and `answer_extraction_failed=True`.

**TriviaQA:** Priority-1 (strict `Answer:` line, last match, start-of-line anchor); Priority-1.5 (mid-line commit after punctuation separator: `[,;.…]\s*[Aa]nswer\s*:\s*(.+?)`); Priority-2 (commit phrases: `"My answer is:"`, `"The answer is:"`, `"Final answer:"`). All TriviaQA extractions apply: (a) sentence-boundary truncation before self-commentary clauses (`". I "`, `". My "`, etc.), (b) bare-number rejection (TriviaQA answers are never isolated integers), and (c) meta-commentary word blocklist (`"correct"`, `"incorrect"`, `"right"`, `"wrong"`, etc.).

**StrategyQA / LegalBench:** Pattern matching for `Yes` or `No` on an `Answer:` line; fall back to last standalone `Yes/No` in the response.

#### Strict vs. Lax Extraction

`extract_model_answer_strict` applies Priority-1 only (anchored `Answer:` line). Strict extraction determines whether a forced-answer call is needed: if strict extraction fails, `was_forced=True` is set and the forced call runs. The lax `extract_model_answer` (Priority-1 through Priority-2) is used on the forced-answer response.

### 8.2 Correctness Evaluation

- **GSM8K:** Exact numeric match after stripping commas and whitespace.
- **StrategyQA / LegalBench:** Case-insensitive `Yes/No` match.
- **TriviaQA:** `check_triviaqa_correct` uses three-tier alias matching: (1) exact normalized match against official TriviaQA alias list, (2) `model_lower in alias`, (3) `alias in model_lower`. The alias list is provided by the `mandarjoshi/trivia_qa` dataset's `answer.normalized_aliases` field.
- *MMLU-Pro and MedQA (exact letter match) are no longer active datasets.*

### 8.3 Refusal Detection

`is_refusal_response(response, extracted_answer)` returns `True` only when the extracted answer is empty or None **and** the response text matches one of ten conservative abstention patterns (e.g., `"I can't determine … without"`, `"please provide the terms"`, `"I don't have access"`). The patterns are scanned against the **tail** of the response only: the final 350 characters for responses longer than 400 characters, or the full response otherwise. This prevents casual mid-reasoning uncertainty phrases (`"I can't recall exactly who…"`) from triggering the detector, since genuine abstentions always close the response while such phrases appear in opening reasoning sentences.

Refusal rows are a strict subset of `answer_extraction_failed=True` rows. They are flagged rather than forced, and excluded from accuracy and calibration analysis.
