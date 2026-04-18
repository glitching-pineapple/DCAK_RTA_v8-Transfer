# semantic_entropy.py - Semantic Entropy implementation based on Kuhn et al. (2023)
# Paper: "Semantic Uncertainty: Linguistic Invariances for Uncertainty Estimation in NLG"

import torch
import numpy as np
from typing import List, Tuple, Dict, Optional
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from collections import defaultdict
import warnings


class SemanticEntropyCalculator:
    """
    Implements semantic entropy for uncertainty estimation.
    
    Key idea: Different text sequences can have the same meaning (semantic equivalence).
    We cluster semantically equivalent answers and compute entropy over meaning-clusters
    rather than individual sequences.
    """
    
    def __init__(
        self,
        nli_model_name: str = "microsoft/deberta-large-mnli",
        device: str = None,
        entailment_threshold: float = 0.5,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.entailment_threshold = entailment_threshold
        
        print(f"Loading NLI model: {nli_model_name} → {self.device}")
        self.nli_tokenizer = AutoTokenizer.from_pretrained(nli_model_name)
        self.nli_model = AutoModelForSequenceClassification.from_pretrained(
            nli_model_name
        ).to(self.device)
        self.nli_model.eval()
        
        # DeBERTa-MNLI label mapping: 0=contradiction, 1=neutral, 2=entailment
        self.entailment_id = 2
        print("NLI model loaded successfully")
    
    @staticmethod
    def truncate_for_nli(text: str, max_chars: int = 1800) -> str:
        """
        Trim a CoT response to fit within DeBERTa's 512-token limit.

        Strips the boilerplate answer/confidence/correct footer first (those
        are already captured elsewhere), then keeps the last `max_chars`
        characters of the remaining reasoning — the conclusion of the chain
        is most discriminative for semantic equivalence.
        """
        # Remove trailing answer declaration lines
        import re
        text = re.sub(r'\n*(Answer|Confidence|Correct)\s*:.*', '', text, flags=re.IGNORECASE).strip()
        # If still too long, keep the tail (conclusion of reasoning)
        if len(text) > max_chars:
            text = text[-max_chars:]
        return text

    def check_entailment_batch(
        self,
        premise_hypothesis_pairs: List[Tuple[str, str]],
        batch_size: int = 32,
    ) -> List[bool]:
        """
        Check entailment for multiple pairs in batched forward passes.

        Much faster than calling check_entailment one pair at a time.
        For 10 answers: up to 90 pairs → 3 batches of 32 instead of 90 calls.
        """
        if not premise_hypothesis_pairs:
            return []

        all_results = []

        for batch_start in range(0, len(premise_hypothesis_pairs), batch_size):
            batch = premise_hypothesis_pairs[batch_start:batch_start + batch_size]
            premises = [p for p, h in batch]
            hypotheses = [h for p, h in batch]

            inputs = self.nli_tokenizer(
                premises, hypotheses,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True,
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.nli_model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1)
                entailment_probs = probs[:, self.entailment_id]
                results = (entailment_probs > self.entailment_threshold).cpu().tolist()
            
            all_results.extend(results)
        
        return all_results
    
    def check_entailment(self, premise: str, hypothesis: str) -> bool:
        """Single-pair entailment check (kept for compatibility)."""
        return self.check_entailment_batch([(premise, hypothesis)])[0]
    
    def are_semantically_equivalent(
        self, 
        context: str, 
        answer1: str, 
        answer2: str
    ) -> bool:
        """
        Check if two answers are semantically equivalent using bidirectional entailment.
        """
        text1 = f"Question: {context} Answer: {answer1}"
        text2 = f"Question: {context} Answer: {answer2}"
        
        # Batch both directions in a single call
        results = self.check_entailment_batch([
            (text1, text2),
            (text2, text1),
        ])
        
        return results[0] and results[1]
    
    def cluster_answers(
        self, 
        context: str, 
        answers: List[str]
    ) -> List[List[int]]:
        """
        Cluster answers by semantic equivalence using batched NLI.
        
        Collects all needed comparison pairs upfront, runs them in one
        batched forward pass, then builds clusters from the results.
        
        For 10 answers with ~3 clusters, this typically requires ~18 pairs
        (9 answers × 2 directions each vs cluster representative) processed
        in a single batch instead of 18 separate forward passes.
        """
        if not answers:
            return []
        
        if len(answers) == 1:
            return [[0]]
        
        # Build comparison texts, truncating long CoT responses so they fit
        # within DeBERTa's 512-token limit without losing the reasoning conclusion.
        answer_texts = [
            f"Question: {context} Answer: {self.truncate_for_nli(a)}"
            for a in answers
        ]
        
        # We need to compare each answer (starting from index 1) against
        # existing cluster representatives. Since clusters form dynamically,
        # we process in rounds — but batch all comparisons within each round.
        
        clusters: List[List[int]] = [[0]]
        
        # Process in small groups to balance batching vs dynamic clustering
        # Group size of ~5 gives good batching while keeping clustering accurate
        group_size = min(5, len(answers) - 1)
        
        i = 1
        while i < len(answers):
            group_end = min(i + group_size, len(answers))
            group_indices = list(range(i, group_end))
            
            # Get current cluster representatives
            rep_indices = [c[0] for c in clusters]
            
            # Build all forward+backward pairs for this group
            pairs = []
            pair_map = []  # Track which (answer_idx, rep_idx) each pair corresponds to
            
            for ans_idx in group_indices:
                for rep_idx in rep_indices:
                    # Forward: answer → representative
                    pairs.append((answer_texts[ans_idx], answer_texts[rep_idx]))
                    # Backward: representative → answer
                    pairs.append((answer_texts[rep_idx], answer_texts[ans_idx]))
                    pair_map.append((ans_idx, rep_idx))
            
            # Run all pairs in one batch
            if pairs:
                results = self.check_entailment_batch(pairs)
            else:
                results = []
            
            # Process results: every 2 consecutive results are forward+backward
            result_idx = 0
            for ans_idx, rep_idx in pair_map:
                forward = results[result_idx]
                backward = results[result_idx + 1]
                result_idx += 2
                
                # If bidirectional entailment, add to that cluster
                if forward and backward:
                    # Find which cluster has this representative
                    for cluster in clusters:
                        if cluster[0] == rep_idx:
                            if ans_idx not in cluster:
                                cluster.append(ans_idx)
                            break
            
            # Any answers not assigned to a cluster get their own
            assigned = set()
            for cluster in clusters:
                assigned.update(cluster)
            
            for ans_idx in group_indices:
                if ans_idx not in assigned:
                    clusters.append([ans_idx])
            
            i = group_end
        
        return clusters
    
    def compute_semantic_entropy(
        self,
        context: str,
        answers: List[str],
        log_probs: List[float],
        length_normalize: bool = False,
        answer_lengths: List[int] = None,
        clustering_answers: List[str] = None,
    ) -> Dict[str, float]:
        """
        Compute semantic entropy over meaning-clusters.
        
        SE(x) = -sum_c p(c|x) * log(p(c|x))
        
        where p(c|x) = sum_{s in c} p(s|x) is the probability mass of a semantic cluster.
        """
        if not answers or not log_probs:
            return {
                "semantic_entropy": float('inf'),
                "num_clusters": 0,
                "cluster_sizes": [],
                "predictive_entropy": float('inf'),
                "predictive_entropy_normalized": float('inf'),
            }
        
        # Count-based predictive entropy over discrete answer choices.
        # Raw sequence log_probs produce entropy ≈ ln(N) for all long CoT
        # responses because similar-length sequences have nearly uniform
        # softmax weights. Count-based entropy directly measures answer
        # diversity: 0 when all samples agree, ln(N) when all differ.
        from collections import Counter
        answer_counts = Counter(answers)
        n = len(answers)
        count_probs = np.array([c / n for c in answer_counts.values()], dtype=np.float64)
        predictive_entropy = float(max(0.0, -np.sum(count_probs * np.log(count_probs + 1e-10))))
        predictive_entropy_normalized = predictive_entropy / np.log(n) if n > 1 else 0.0

        # Length-normalized sequence probs for SE cluster weighting
        log_probs_arr = np.array(log_probs, dtype=np.float64)
        if length_normalize and answer_lengths:
            norm_log_probs_arr = np.array(
                [lp / max(1, length) for lp, length in zip(log_probs, answer_lengths)],
                dtype=np.float64,
            )
        else:
            norm_log_probs_arr = log_probs_arr
        max_norm = np.max(norm_log_probs_arr)
        probs = np.exp(norm_log_probs_arr - max_norm)
        probs = probs / np.sum(probs)

        # --- SE over CoT reasoning chains (primary) ---
        # Cluster on full CoT if provided, otherwise fall back to answer letters.
        texts_to_cluster = clustering_answers if clustering_answers is not None else answers
        clusters = self.cluster_answers(context, texts_to_cluster)
        num_clusters = len(clusters)
        cluster_sizes = [len(c) for c in clusters]

        cluster_probs = np.array([sum(probs[i] for i in c) for c in clusters])
        semantic_entropy = float(max(0.0, -np.sum(cluster_probs * np.log(cluster_probs + 1e-10))))

        # --- SE over answer strings (secondary, useful for open-ended datasets) ---
        # Only computed when clustering_answers were provided (i.e. CoT was used
        # above). Groups answers that are semantically equivalent as strings —
        # e.g. "36" vs "thirty-six" for GSM8K, or paraphrase answers for TriviaQA.
        if clustering_answers is not None:
            ans_clusters = self.cluster_answers(context, answers)
            ans_cluster_probs = np.array([sum(probs[i] for i in c) for c in ans_clusters])
            semantic_entropy_answers = float(max(0.0, -np.sum(ans_cluster_probs * np.log(ans_cluster_probs + 1e-10))))
            num_answer_clusters = len(ans_clusters)
        else:
            semantic_entropy_answers = semantic_entropy
            num_answer_clusters = num_clusters

        return {
            "semantic_entropy": semantic_entropy,
            "semantic_entropy_answers": semantic_entropy_answers,
            "num_clusters": num_clusters,
            "num_answer_clusters": num_answer_clusters,
            "cluster_sizes": cluster_sizes,
            "predictive_entropy": predictive_entropy,
            "predictive_entropy_normalized": predictive_entropy_normalized,
        }


def sample_answers_with_probs(
    model,
    tokenizer,
    prompt: str,
    num_samples: int = 10,
    max_new_tokens: int = 256,
    temperature: float = 0.5,
) -> Tuple[List[str], List[float], List[int]]:
    """
    Sample multiple answers from the model with their log-probabilities.
    
    Following the paper, we use multinomial sampling with temperature=0.5
    to balance diversity and accuracy.
    
    Uses batched generation: one forward pass for all samples instead of
    num_samples sequential calls. This gives identical results but runs
    significantly faster on GPU by parallelizing the KV-cache computation.
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_length = inputs.input_ids.shape[1]
    
    # Expand input to batch of num_samples (all identical prompts)
    batch_input_ids = inputs.input_ids.expand(num_samples, -1)
    batch_attention = inputs.attention_mask.expand(num_samples, -1)
    
    with torch.no_grad():
        outputs = model.generate(
            input_ids=batch_input_ids,
            attention_mask=batch_attention,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            return_dict_in_generate=True,
            output_scores=True,
            pad_token_id=tokenizer.pad_token_id,
        )
    
    answers = []
    log_probs = []
    lengths = []
    
    for sample_idx in range(num_samples):
        # Extract generated tokens for this sample
        generated_ids = outputs.sequences[sample_idx, input_length:]
        
        # Remove padding tokens from the end
        if tokenizer.pad_token_id is not None:
            non_pad = (generated_ids != tokenizer.pad_token_id).nonzero(as_tuple=True)[0]
            if len(non_pad) > 0:
                generated_ids = generated_ids[:non_pad[-1] + 1]
            else:
                generated_ids = generated_ids[:0]
        
        answer = tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        # Compute log-probability of the sequence
        # outputs.scores[t] has shape (batch_size, vocab_size)
        total_log_prob = 0.0
        seq_len = len(generated_ids)
        for i in range(seq_len):
            if i < len(outputs.scores):
                probs = torch.softmax(outputs.scores[i][sample_idx], dim=-1)
                token_id = generated_ids[i].item()
                token_prob = probs[token_id].item()
                total_log_prob += np.log(token_prob + 1e-10)
        
        answers.append(answer)
        log_probs.append(total_log_prob)
        lengths.append(seq_len)
    
    return answers, log_probs, lengths


def compute_semantic_entropy_for_sample(
    model,
    tokenizer,
    semantic_calculator: SemanticEntropyCalculator,
    question: str,
    prompt: str,
    num_samples: int = 10,
    temperature: float = 0.5,
    max_new_tokens: int = 256,
    answer_extractor=None,
) -> Dict:
    """
    Convenience function to compute semantic entropy for a single question.
    """
    # Sample multiple answers
    answers, log_probs, lengths = sample_answers_with_probs(
        model, tokenizer, prompt,
        num_samples=num_samples,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )
    
    # Extract just the answer portion if extractor provided
    if answer_extractor:
        extracted_answers = [answer_extractor(a) or a for a in answers]
    else:
        extracted_answers = answers
    
    # Compute semantic entropy
    se_results = semantic_calculator.compute_semantic_entropy(
        context=question,
        answers=extracted_answers,
        log_probs=log_probs,
        length_normalize=True,
        answer_lengths=lengths,
    )
    
    return {
        **se_results,
        "sampled_answers": answers,
        "extracted_answers": extracted_answers,
        "log_probs": log_probs,
        "answer_lengths": lengths,
    }