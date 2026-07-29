#!/usr/bin/env python3
"""Golden-snapshot guard for prompt construction.

Prompts are part of the experimental setup: ANY byte change (wording,
whitespace) changes model behavior and breaks comparability with previously
committed results. This script renders every prompt the pipeline can build
(dataset × variant × include_confidence, plus the SE sampling prompt and the
Gen-2 / two-pass builder prompts) and compares them against
prompt_golden.json.

Usage:
    python3 check_prompt_golden.py            # verify (CI-style; exit 1 on drift)
    python3 check_prompt_golden.py --update   # re-capture after an INTENTIONAL change

If this fails after a refactor that was supposed to be behavior-preserving,
the refactor changed prompt bytes — fix the refactor, don't update the golden.
"""
import argparse
import json
import os
import sys
import types

# Stub heavy deps (same approach as verify_rubric.py) so this runs anywhere.
for name, attrs in [
    ("torch", {"no_grad": lambda *a, **k: (lambda f: f), "softmax": lambda *a, **k: None}),
    ("datasets", {"load_dataset": lambda *a, **k: None}),
]:
    if name not in sys.modules:
        mod = types.ModuleType(name)
        for k, v in attrs.items():
            setattr(mod, k, v)
        sys.modules[name] = mod
if "numpy" not in sys.modules:
    np_stub = types.ModuleType("numpy")
    for k in ("array", "exp", "mean", "log", "min", "sum"):
        setattr(np_stub, k, lambda *a, **kw: 0)
    np_stub.float64 = float
    sys.modules["numpy"] = np_stub
if "transformers" not in sys.modules:
    tf_stub = types.ModuleType("transformers")
    tf_stub.AutoModelForCausalLM = type("AutoModelForCausalLM", (), {})
    tf_stub.AutoTokenizer = type("AutoTokenizer", (), {})
    sys.modules["transformers"] = tf_stub

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config  # noqa: E402

GOLDEN_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompt_golden.json")

FIXTURES = {
    "gsm8k": ("If Tom has 3 apples and gives 1 away, how many remain?", None),
    "mmlupro": ("What is 2+2?", ["1", "2", "3", "4"]),
    "strategyqa": ("Is the sky blue?", None),
    "medqa": ("A patient presents with chest pain. Diagnosis?",
              ["Angina", "MI", "PE", "Pericarditis", "Aortic dissection"]),
    "triviaqa": ("What is the capital of France?", None),
    "legalbench": ("Is hearsay admissible in this scenario?", None),
}


class StubTokenizer:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        # Wrap so template application is visible in the snapshot
        return "<CHAT>" + messages[0]["content"] + "</CHAT>"


def render_all() -> dict:
    from confidence import create_prompt, create_simple_prompt
    out = {}
    tok = StubTokenizer()
    for variant in ("instruct", "base"):
        config.MODEL_VARIANT = variant
        for ds, (q, choices) in FIXTURES.items():
            config.DATASET = ds
            for inc in (True, False):
                key = f"create_prompt/{ds}/{variant}/conf={inc}"
                out[key] = create_prompt(tok, q, choices, include_confidence=inc)
            out[f"create_simple_prompt/{ds}/{variant}"] = create_simple_prompt(tok, q, choices)

    # Gen-2 / two-pass builder prompts (pure builders when available; else
    # skipped — pre-refactor goldens only cover create_* which is fine, the
    # builders are added by the dedup refactor itself).
    try:
        from confidence import build_gen2_prompt, build_two_pass_prompt
        for ds, (q, choices) in FIXTURES.items():
            config.DATASET = ds
            out[f"gen2_prompt/{ds}"] = build_gen2_prompt(
                q, "some earlier reasoning", "AnswerX", choices)
            out[f"two_pass_prompt/{ds}"] = build_two_pass_prompt(
                q, "AnswerX", "some earlier reasoning", choices,
                gen2_confidence=7.0, gen2_correct=True)
            out[f"two_pass_prompt_nogen2/{ds}"] = build_two_pass_prompt(
                q, "AnswerX", "some earlier reasoning", choices)
    except ImportError:
        pass
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--update", action="store_true", help="re-capture goldens")
    args = ap.parse_args()

    rendered = render_all()

    if args.update or not os.path.exists(GOLDEN_PATH):
        with open(GOLDEN_PATH, "w", encoding="utf-8") as f:
            json.dump(rendered, f, indent=1, ensure_ascii=False, sort_keys=True)
        print(f"Wrote {len(rendered)} golden prompts to {GOLDEN_PATH}")
        return 0

    with open(GOLDEN_PATH, encoding="utf-8") as f:
        golden = json.load(f)

    drift = []
    for key, text in rendered.items():
        if key not in golden:
            drift.append((key, "NEW KEY (not in golden — run --update if intentional)"))
        elif golden[key] != text:
            # locate first differing char for a useful message
            g = golden[key]
            pos = next((i for i, (a, b) in enumerate(zip(g, text)) if a != b),
                       min(len(g), len(text)))
            drift.append((key, f"BYTE DRIFT at char {pos}: "
                               f"golden={g[pos:pos+40]!r} vs now={text[pos:pos+40]!r}"))
    missing = [k for k in golden if k not in rendered]

    for key, msg in drift:
        print(f"DRIFT  {key}: {msg}")
    for key in missing:
        print(f"MISSING  {key} (rendered set no longer produces it)")

    if drift or missing:
        print(f"\nFAIL — {len(drift)} drifted, {len(missing)} missing "
              f"(of {len(golden)} golden prompts)")
        return 1
    print(f"PASS — all {len(golden)} prompts byte-identical to golden")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
