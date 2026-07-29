#!/usr/bin/env python3
"""PRM (Process Reward Model) step-scoring over saved CoT responses.

Splits each row's full_response into steps, scores every step with
Qwen2.5-Math-PRM-7B, and writes min/mean step rewards per row plus summary
stats (mean, std, AUROC vs is_correct).

Usage:
    python3 prm_scoring.py <input.csv> [-o output.csv]
    python3 prm_scoring.py Steps_Qwen_Instruct_StratQA.csv -o PRM_QwenInstruct_StratQA.csv

Previously this was a run-on-import script with hardcoded filenames and no
main() guard — importing it loaded a 7B model as a side effect.
"""
import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

PRM_MODEL_NAME = "Qwen/Qwen2.5-Math-PRM-7B"
DEFAULT_SYSTEM_PROMPT = ("Please reason step by step, and put your final answer "
                         "within \\boxed{}.")


# ============ PRM Functions ============

def make_step_rewards(logits, token_masks):
    probabilities = F.softmax(logits, dim=-1)
    probabilities = probabilities * token_masks.unsqueeze(-1)

    all_scores_res = []
    for i in range(probabilities.size(0)):
        sample = probabilities[i]
        positive_probs = sample[sample != 0].view(-1, 2)[:, 1]
        non_zero_elements_list = positive_probs.cpu().tolist()
        all_scores_res.append(non_zero_elements_list)
    return all_scores_res


def split_into_steps(full_response):
    """Split response by double newlines."""
    steps = full_response.strip().split("\n\n")
    steps = [s.strip() for s in steps if s.strip()]
    return steps


def get_step_rewards(model, tokenizer, question, steps,
                     system_prompt=DEFAULT_SYSTEM_PROMPT):
    """Get PRM scores for each step."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
        {"role": "assistant", "content": "<extra_0>".join(steps) + "<extra_0>"},
    ]

    conversation_str = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    input_ids = tokenizer.encode(
        conversation_str,
        return_tensors="pt",
    ).to(model.device)

    # convert_tokens_to_ids is the robust lookup — the previous
    # tokenizer.encode("<extra_0>")[0] silently breaks if the tokenizer ever
    # prepends a BOS token or splits the marker.
    step_sep_id = tokenizer.convert_tokens_to_ids("<extra_0>")
    token_masks = (input_ids == step_sep_id)

    with torch.no_grad():
        outputs = model(input_ids=input_ids)

    step_rewards = make_step_rewards(outputs[0], token_masks)
    return step_rewards[0]  # list of per-step scores


def load_prm_model():
    from transformers import AutoModel, AutoTokenizer
    from config import PRM_MODEL_REVISION

    print(f"Loading PRM model {PRM_MODEL_NAME} (revision={PRM_MODEL_REVISION})...")
    # The PRM head is custom code, so trust_remote_code is genuinely required;
    # the pinned revision ensures the executed code can't change upstream.
    tokenizer = AutoTokenizer.from_pretrained(
        PRM_MODEL_NAME, revision=PRM_MODEL_REVISION, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        PRM_MODEL_NAME,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        revision=PRM_MODEL_REVISION,
        trust_remote_code=True,
    ).eval()
    print("Model loaded!")
    return model, tokenizer


def score_dataframe(df: pd.DataFrame, model, tokenizer) -> pd.DataFrame:
    df = df.copy()
    df['steps'] = df['full_response'].apply(split_into_steps)

    print("Computing step rewards...")
    step_rewards_list, min_rewards, mean_rewards = [], [], []
    n_failed = 0
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        try:
            rewards = get_step_rewards(model, tokenizer, row['question'], row['steps'])
            step_rewards_list.append(rewards)
            min_rewards.append(min(rewards) if rewards else None)
            mean_rewards.append(sum(rewards) / len(rewards) if rewards else None)
        except Exception as e:
            n_failed += 1
            print(f"Error on row {idx}: {e}")
            step_rewards_list.append([])
            min_rewards.append(None)
            mean_rewards.append(None)

    df['step_rewards'] = step_rewards_list
    df['min_step_reward'] = min_rewards
    df['mean_step_reward'] = mean_rewards
    if n_failed:
        print(f"WARNING: {n_failed}/{len(df)} rows failed PRM scoring "
              f"(step_rewards empty; excluded from summary stats).")
    return df


def _auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Rank-based AUROC (no sklearn dependency)."""
    pos, neg = scores[labels], scores[~labels]
    if len(pos) == 0 or len(neg) == 0:
        return float('nan')
    greater = (pos[:, None] > neg[None, :]).sum()
    equal = (pos[:, None] == neg[None, :]).sum()
    return float((greater + 0.5 * equal) / (len(pos) * len(neg)))


def print_summary(df: pd.DataFrame):
    print("\n=== Summary ===")
    for col in ("min_step_reward", "mean_step_reward"):
        valid = df[df[col].notna()]
        print(f"\n{col}: mean={valid[col].mean():.4f}  std={valid[col].std():.4f}  "
              f"n={len(valid)}")
        if 'is_correct' in valid.columns and valid['is_correct'].nunique() == 2:
            correct = valid[valid['is_correct'].astype(bool)]
            wrong = valid[~valid['is_correct'].astype(bool)]
            print(f"  correct rows: {correct[col].mean():.4f} (n={len(correct)})")
            print(f"  wrong rows:   {wrong[col].mean():.4f} (n={len(wrong)})")
            auroc = _auroc(valid['is_correct'].astype(bool).values,
                           valid[col].astype(float).values)
            print(f"  AUROC (predicts correctness): {auroc:.4f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input_csv", help="CSV with question / full_response / is_correct columns")
    ap.add_argument("-o", "--output", default=None,
                    help="output CSV (default: PRM_<input name>)")
    args = ap.parse_args()

    out_path = args.output or f"PRM_{args.input_csv}"

    df = pd.read_csv(args.input_csv)
    for col in ("question", "full_response"):
        if col not in df.columns:
            raise SystemExit(f"Input CSV missing required column: {col}")

    model, tokenizer = load_prm_model()
    df = score_dataframe(df, model, tokenizer)

    df.to_csv(out_path, index=False)
    print(f"Saved to {out_path}")

    print_summary(df)


if __name__ == "__main__":
    main()
