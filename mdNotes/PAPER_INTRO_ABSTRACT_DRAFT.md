# Abstract

Large language models (LLMs) are increasingly deployed in settings where knowing when a model is right matters as much as being right. Many confidence signals have been proposed — internal token probabilities, verbalized self-ratings, binary self-verdicts — but they are typically evaluated in isolation, extracted from the same generation that produced the answer, and rarely tested for whether their quality transfers across tasks and models. We present a systematic evaluation of confidence signals across six open-weight instruction-tuned models (7B–31B parameters, including three reasoning models) and four task domains, totaling roughly 3,400 scored responses. Our pipeline separates answering, self-assessment, and a blinded critique in which the model re-evaluates its own reasoning presented as someone else's, so that any change in confidence is attributable to authorship belief alone. We find that no confidence signal reliably dominates: the best logit-based and best verbalized signals are nearly indistinguishable on average (AUROC 0.68 vs. 0.67), and which dataset a signal is measured on explains more of its apparent quality than which signal it is — every signal approaches chance on legal reasoning (about 0.56–0.59) while exceeding 0.77 on factual recall. The blinded critique reduces confidence on incorrect answers in 19 of 24 model–dataset settings and improves discrimination in 16 of 24, but improves calibration error in only 7 of 24: removing authorship helps models sort right answers from wrong ones, not estimate probabilities. These results caution against choosing a confidence signal a priori and show that its reliability must be validated on the target task; our framework provides a standardized way to do so.

# Introduction

Large language models now perform strongly across mathematical, factual, commonsense, and legal reasoning tasks (Cobbe et al., 2021; Geva et al., 2021; Guha et al., 2023; Joshi et al., 2017). Yet they remain prone to producing fluent, confidently incorrect answers (Ji et al., 2023; Xiong et al., 2024), with documented consequences in high-stakes domains such as law, where hallucination rates on legal queries have been measured at 58–82% for popular models (Dahl et al., 2024). As deployment expands, estimating when a model is likely to be correct — so that unreliable outputs can be flagged, deferred, or discarded — has become as important as raw accuracy (Guo et al., 2017; Hendrycks & Gimpel, 2017).

Prior work offers two main families of confidence signals. Probability-based methods derive confidence from the model's own token probabilities (Hendrycks & Gimpel, 2017; Jiang et al., 2021; Kadavath et al., 2022) or from the semantic dispersion of multiple sampled answers (Farquhar et al., 2024; Kuhn et al., 2023). Verbalized methods instead ask the model to state its confidence directly, either as a probability or on a discrete scale (Lin et al., 2022; Tian et al., 2023; Xiong et al., 2024), or as a binary self-verdict on its own answer (Kadavath et al., 2022). Neither family is reliable in general: instruction tuning and reinforcement learning from human feedback degrade the calibration of token probabilities (OpenAI, 2023; Tian et al., 2023), while verbalized confidence is systematically overconfident (Xiong et al., 2024). Crucially, these signals are usually evaluated in isolation — on different models, tasks, and protocols — making direct comparison difficult (Geng et al., 2024). Moreover, confidence is typically extracted from the same generation that produces the answer, conflating reasoning with self-assessment. This matters because models demonstrably favor their own outputs when evaluating them (Panickssery et al., 2024; Zheng et al., 2023) and struggle to find their own reasoning errors (Huang et al., 2024). Recent work has begun to ask whether critique can improve confidence estimation (Yang et al., 2025), but no existing design isolates the effect of authorship — whether a model judges the same reasoning differently when it believes the work is its own.

We introduce a unified evaluation framework that measures multiple confidence signals under one protocol across six open-weight instruction-tuned models from four families — Gemma (Gemma Team, 2024, 2025), Llama (Grattafiori et al., 2024), Qwen (Qwen Team, 2024, 2025), and GPT-OSS (OpenAI, 2025) — including three reasoning ("thinking") models (Guo et al., 2025; OpenAI, 2024), on four benchmarks spanning arithmetic (GSM8K; Cobbe et al., 2021), multi-hop commonsense reasoning (StrategyQA; Geva et al., 2021), factual recall (TriviaQA; Joshi et al., 2017), and legal reasoning (LegalBench; Guha et al., 2023). For every question we collect logit-based confidence aggregates, a graded 1–10 verbalized confidence elicited against an explicit probability rubric (Yoon et al., 2025), and a binary more-likely-than-not self-verdict (Kadavath et al., 2022). A standardized extraction and scoring pipeline retains truncated and recovered responses rather than dropping them, since generation failures concentrate on difficult items and their removal would bias calibration estimates toward easy questions.

The central design choice is a three-stage generation pipeline. Stage one produces a reasoning trace and answer with no confidence request, so reasoning models can spend their token budget on reasoning. Stage two shows the model its own completed work, explicitly framed as its own, and elicits a rubric-based confidence. Stage three presents identical content framed as another model's submitted solution and elicits a fresh critique and confidence. Because the only difference between stages two and three is the claimed authorship, any systematic difference between them is attributable to authorship belief — a controlled test of self-preference in confidence estimation (Panickssery et al., 2024) that single-generation designs cannot perform, and a manipulation absent from prior critique-based calibration work (Yang et al., 2025).

Using this framework we ask three questions. First, do confidence signals generalize across datasets and reasoning domains? Second, are signal rankings consistent across model families and architectures, including reasoning models? Third, does separating and blinding self-assessment reduce overconfidence on incorrect answers?

Three results emerge. First, no signal reliably dominates: sequence log-probability attains the best mean discrimination (AUROC 0.68) but the worst per-dataset floor (0.57), while blinded verbalized confidence is statistically indistinguishable on average (0.67) with a better floor (0.59); the evaluation dataset explains more of a signal's apparent quality than the signal's identity, with all signals near chance on LegalBench (0.56–0.59) but strong on TriviaQA (up to 0.77 and above). Second, the blinded critique acts as an error filter, not a better probability estimate: it lowers confidence on wrong answers in 19 of 24 model–dataset settings (row-level p < 10⁻¹⁶) and improves AUROC in 16 of 24, yet improves expected calibration error (Guo et al., 2017; Naeini et al., 2015) in only 7 of 24 — a dissociation between discrimination and calibration that echoes findings that calibration degrades under distribution shift even when in-domain calibration looks good (Ovadia et al., 2019). Third, coarse orderings transfer even when magnitudes do not: graded confidence beats binary self-verdicts nearly everywhere, but per-task thresholds do not transfer across datasets.

Our results provide practical guidance for building LLM systems that must decide when to trust a model without additional fine-tuning: no signal should be selected a priori, signal quality must be validated on the target task, and a cheap blinded self-critique is an effective filter for wrong answers even though it does not produce calibrated probabilities. More broadly, this work establishes a reproducible standard of evidence for claims about LLM metacognition: per-task, per-model evaluation across signal families, rather than headline numbers from a single benchmark.

In summary, our contributions are:

- A unified, open evaluation framework measuring logit-based, verbalized, and critique-derived confidence under one protocol across six open-weight models (7B–31B, three of them reasoning models) and four task domains, totaling roughly 3,400 scored responses.
- A three-stage elicitation pipeline whose blinded-authorship manipulation isolates, for the first time, the causal effect of believed authorship on confidence.
- Evidence that no confidence signal reliably dominates, and that task identity explains more variance in signal quality than signal identity (mean AUROC 0.68 vs. 0.67; floors 0.57 vs. 0.59; all signals near chance on legal reasoning).
- Evidence that blinded critique improves discrimination (16 of 24 settings) and reduces wrong-answer confidence (19 of 24, p < 10⁻¹⁶) without improving calibration (7 of 24), identifying it as an error filter rather than a probability estimator.

# References

Cobbe, K., Kosaraju, V., Bavarian, M., Chen, M., Jun, H., Kaiser, L., Plappert, M., Tworek, J., Hilton, J., Nakano, R., Hesse, C., & Schulman, J. (2021). Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168.

Dahl, M., Magesh, V., Suzgun, M., & Ho, D. E. (2024). Large legal fictions: Profiling legal hallucinations in large language models. Journal of Legal Analysis, 16(1), 64–93.

Farquhar, S., Kossen, J., Kuhn, L., & Gal, Y. (2024). Detecting hallucinations in large language models using semantic entropy. Nature, 630, 625–630.

Gemma Team. (2024). Gemma 2: Improving open language models at a practical size. arXiv preprint arXiv:2408.00118.

Gemma Team. (2025). Gemma 3 technical report. arXiv preprint arXiv:2503.19786.

Geng, J., Cai, F., Wang, Y., Koeppl, H., Nakov, P., & Gurevych, I. (2024). A survey of confidence estimation and calibration in large language models. In Proceedings of NAACL 2024.

Geva, M., Khashabi, D., Segal, E., Khot, T., Roth, D., & Berant, J. (2021). Did Aristotle use a laptop? A question answering benchmark with implicit reasoning strategies. Transactions of the Association for Computational Linguistics, 9, 346–361.

Grattafiori, A., Dubey, A., Jauhri, A., et al. (2024). The Llama 3 herd of models. arXiv preprint arXiv:2407.21783.

Guha, N., Nyarko, J., Ho, D. E., et al. (2023). LegalBench: A collaboratively built benchmark for measuring legal reasoning in large language models. In Advances in Neural Information Processing Systems 36, Datasets and Benchmarks Track.

Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). On calibration of modern neural networks. In Proceedings of the 34th International Conference on Machine Learning.

Guo, D., Yang, D., Zhang, H., et al. (2025). DeepSeek-R1: Incentivizing reasoning capability in LLMs via reinforcement learning. arXiv preprint arXiv:2501.12948.

Hendrycks, D., & Gimpel, K. (2017). A baseline for detecting misclassified and out-of-distribution examples in neural networks. In International Conference on Learning Representations.

Huang, J., Chen, X., Mishra, S., Zheng, H. S., Yu, A. W., Song, X., & Zhou, D. (2024). Large language models cannot self-correct reasoning yet. In International Conference on Learning Representations.

Ji, Z., Lee, N., Frieske, R., Yu, T., Su, D., Xu, Y., Ishii, E., Bang, Y., Madotto, A., & Fung, P. (2023). Survey of hallucination in natural language generation. ACM Computing Surveys, 55(12), 1–38.

Jiang, Z., Araki, J., Ding, H., & Neubig, G. (2021). How can we know when language models know? On the calibration of language models for question answering. Transactions of the Association for Computational Linguistics, 9, 962–977.

Joshi, M., Choi, E., Weld, D. S., & Zettlemoyer, L. (2017). TriviaQA: A large scale distantly supervised challenge dataset for reading comprehension. In Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics.

Kadavath, S., Conerly, T., Askell, A., et al. (2022). Language models (mostly) know what they know. arXiv preprint arXiv:2207.05221.

Kuhn, L., Gal, Y., & Farquhar, S. (2023). Semantic uncertainty: Linguistic invariances for uncertainty estimation in natural language generation. In International Conference on Learning Representations.

Lin, S., Hilton, J., & Evans, O. (2022). Teaching models to express their uncertainty in words. Transactions on Machine Learning Research.

Naeini, M. P., Cooper, G. F., & Hauskrecht, M. (2015). Obtaining well calibrated probabilities using Bayesian binning. In Proceedings of the AAAI Conference on Artificial Intelligence.

OpenAI. (2023). GPT-4 technical report. arXiv preprint arXiv:2303.08774.

OpenAI. (2024). OpenAI o1 system card. arXiv preprint arXiv:2412.16720.

OpenAI. (2025). gpt-oss-120b & gpt-oss-20b model card. arXiv preprint arXiv:2508.10925.

Ovadia, Y., Fertig, E., Ren, J., Nado, Z., Sculley, D., Nowozin, S., Dillon, J. V., Lakshminarayanan, B., & Snoek, J. (2019). Can you trust your model's uncertainty? Evaluating predictive uncertainty under dataset shift. In Advances in Neural Information Processing Systems 32.

Panickssery, A., Bowman, S. R., & Feng, S. (2024). LLM evaluators recognize and favor their own generations. In Advances in Neural Information Processing Systems 37.

Qwen Team. (2024). Qwen2.5 technical report. arXiv preprint arXiv:2412.15115.

Qwen Team. (2025). Qwen3 technical report. arXiv preprint arXiv:2505.09388.

Tian, K., Mitchell, E., Zhou, A., Sharma, A., Rafailov, R., Yao, H., Finn, C., & Manning, C. D. (2023). Just ask for calibration: Strategies for eliciting calibrated confidence scores from language models fine-tuned with human feedback. In Proceedings of EMNLP 2023.

Xiong, M., Hu, Z., Lu, X., Li, Y., Fu, J., He, J., & Hooi, B. (2024). Can LLMs express their uncertainty? An empirical evaluation of confidence elicitation in LLMs. In International Conference on Learning Representations.

Yang, et al. (2025). CritiCal: Can critique help LLM uncertainty or confidence calibration? arXiv preprint arXiv:2510.24505. [VERIFY AUTHOR NAMES ON ARXIV BEFORE SUBMISSION]

Yoon, D., et al. (2025). Reasoning models better express their confidence. arXiv preprint arXiv:2505.14489. [VERIFY FULL AUTHOR LIST ON ARXIV BEFORE SUBMISSION]

Zheng, L., Chiang, W.-L., Sheng, Y., et al. (2023). Judging LLM-as-a-judge with MT-Bench and Chatbot Arena. In Advances in Neural Information Processing Systems 36.
