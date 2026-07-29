#!/usr/bin/env python3
"""GPU smoke test: validate generate_with_logits_batched against the serial path.

Run this ONCE per model family (on the GPU box) before trusting GEN1_BATCH_SIZE > 1:

    python3 smoke_test_batched.py

It loads the model from the current config, decodes a handful of short prompts
both serially and batched, and compares:
  - generated text (must be identical under greedy decoding)
  - per-token probabilities (allclose; batched kernels can differ in the last
    few ulps — report max abs diff)
  - finish_reason / was_truncated meta

Batched greedy decoding is *expected* to be equivalent, but attention-kernel
batching (and left-padding interactions in some architectures) can break that
assumption — this script is the check that it holds for YOUR model + stack.

Interpreting output:
  - text mismatch            -> do NOT use batching for this model. Report it.
  - max prob diff < 1e-4     -> fine (kernel noise).
  - max prob diff in 1e-4..1e-2 -> borderline; inspect which tokens differ.
  - finish_reason mismatch   -> trimming bug or model-specific EOS handling;
                                do not use batching until understood.
"""
import numpy as np
import torch

from config import MAX_NEW_TOKENS, get_model_label
from model_utils import load_model_and_tokenizer
from confidence import generate_with_logits, generate_with_logits_batched

# Short, answer-shaped prompts with deliberately different lengths so the
# batch exercises left-padding. Formatted through the chat template below.
QUESTIONS = [
    "What is 17 + 25? Answer with just the number.",
    "Name the capital city of France. Answer with just the city name.",
    "A patient has a fever of 39C and a productive cough for three days. "
    "Name the single most likely common diagnosis. Answer briefly.",
    "Is the Pacific larger than the Atlantic? Answer Yes or No.",
]
MAX_NEW = 64          # short: this is an equivalence check, not a benchmark
BATCH_SIZE = len(QUESTIONS)


def main():
    print(f"Model: {get_model_label()}  |  max_new_tokens={MAX_NEW}")
    model, tokenizer = load_model_and_tokenizer()

    prompts = []
    for q in QUESTIONS:
        try:
            p = tokenizer.apply_chat_template(
                [{"role": "user", "content": q}],
                tokenize=False, add_generation_prompt=True,
            )
        except Exception:
            p = q + "\n\nAnswer:"
        prompts.append(p)

    print("\nSerial pass...")
    serial = [generate_with_logits(model, tokenizer, p, max_new_tokens=MAX_NEW)
              for p in prompts]

    print("Batched pass...")
    batched = generate_with_logits_batched(
        model, tokenizer, prompts, max_new_tokens=MAX_NEW, batch_size=BATCH_SIZE,
    )

    all_ok = True
    for i, (s, b) in enumerate(zip(serial, batched)):
        s_text, s_probs, s_toks, _, s_meta = s
        b_text, b_probs, b_toks, _, b_meta = b
        text_ok = s_text == b_text
        len_ok = len(s_probs) == len(b_probs)
        if len_ok and s_probs:
            max_diff = float(np.max(np.abs(np.array(s_probs) - np.array(b_probs))))
        else:
            max_diff = float("nan")
        meta_ok = (s_meta["finish_reason"] == b_meta["finish_reason"]
                   and s_meta["was_truncated"] == b_meta["was_truncated"])
        row_ok = text_ok and len_ok and meta_ok and (max_diff < 1e-3)
        all_ok &= row_ok
        print(f"\n[{i}] {'OK' if row_ok else 'MISMATCH'}")
        print(f"    text identical:  {text_ok}")
        print(f"    n_tokens:        serial={len(s_probs)} batched={len(b_probs)}")
        print(f"    max |prob diff|: {max_diff:.2e}")
        print(f"    finish_reason:   serial={s_meta['finish_reason']} "
              f"batched={b_meta['finish_reason']} (match={meta_ok})")
        if not text_ok:
            print(f"    serial : {s_text[:160]!r}")
            print(f"    batched: {b_text[:160]!r}")

    print("\n" + "=" * 50)
    if all_ok:
        print(f"PASS — batching validated for {get_model_label()}. "
              f"GEN1_BATCH_SIZE > 1 is safe for this model.")
    else:
        print("FAIL — do NOT enable GEN1_BATCH_SIZE > 1 for this model "
              "until the mismatch is understood.")
    print("=" * 50)
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
