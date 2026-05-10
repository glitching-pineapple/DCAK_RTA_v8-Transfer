"""Verify that all 5 datasets present the same 10-class rubric on a 1-10 scale,
and that extract_verbalized_confidence parses the expected output format.

Stubs torch and numpy so the script can run on machines without those deps —
we only need the prompt-construction code paths and the pure-Python regex
extractors, not actual model inference.
"""

import sys
import os
import types

# Stub heavy deps before importing confidence.py
if "torch" not in sys.modules:
    torch_stub = types.ModuleType("torch")
    torch_stub.no_grad = lambda *a, **kw: (lambda f: f)
    torch_stub.softmax = lambda *a, **kw: None
    sys.modules["torch"] = torch_stub
if "numpy" not in sys.modules:
    numpy_stub = types.ModuleType("numpy")
    numpy_stub.array = lambda *a, **kw: None
    numpy_stub.exp = lambda *a, **kw: 0
    numpy_stub.mean = lambda *a, **kw: 0
    numpy_stub.log = lambda *a, **kw: 0
    numpy_stub.min = lambda *a, **kw: 0
    numpy_stub.sum = lambda *a, **kw: 0
    numpy_stub.float64 = float
    sys.modules["numpy"] = numpy_stub
if "transformers" not in sys.modules:
    transformers_stub = types.ModuleType("transformers")
    transformers_stub.AutoModelForCausalLM = type("AutoModelForCausalLM", (), {})
    transformers_stub.AutoTokenizer = type("AutoTokenizer", (), {})
    sys.modules["transformers"] = transformers_stub
if "datasets" not in sys.modules:
    datasets_stub = types.ModuleType("datasets")
    datasets_stub.load_dataset = lambda *a, **kw: None
    sys.modules["datasets"] = datasets_stub

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from confidence import (
    create_prompt,
    extract_verbalized_confidence,
    extract_more_likely_than_not,
)


class StubTokenizer:
    """Pretend tokenizer for prompt-construction tests; bypasses chat template."""

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return messages[0]["content"]


CLASSES = [
    "Almost no chance",
    "Highly unlikely",
    "Chances are slight",
    "Unlikely",
    "Less than even",
    "Better than even",
    "Likely",
    "Very good chance",
    "Highly likely",
    "Almost certain",
]

PERCENTAGES = [
    "0-10% likely correct",
    "10-20% likely correct",
    "20-30% likely correct",
    "30-40% likely correct",
    "40-50% likely correct",
    "50-60% likely correct",
    "60-70% likely correct",
    "70-80% likely correct",
    "80-90% likely correct",
    "90-100% likely correct",
]

DATASET_FIXTURES = {
    "gsm8k": ("If Tom has 3 apples and gives 1 away, how many remain?", None),
    "mmlupro": ("What is 2+2?", ["1", "2", "3", "4"]),
    "strategyqa": ("Is the sky blue?", None),
    "medqa": ("A patient presents with chest pain. Diagnosis?",
              ["Angina", "MI", "PE", "Pericarditis", "Aortic dissection"]),
    "triviaqa": ("What is the capital of France?", None),
}


def check_prompt(dataset: str, question: str, choices) -> None:
    config.DATASET = dataset
    config.MODEL_VARIANT = "base"  # bypass chat template
    prompt = create_prompt(StubTokenizer(), question, choices, include_confidence=True)

    # All 10 verbal classes must appear (in quoted form)
    for cls in CLASSES:
        assert f'"{cls}"' in prompt, f"[{dataset}] missing class label: {cls!r}"

    # All 10 percentage descriptors must appear
    for pct in PERCENTAGES:
        assert pct in prompt, f"[{dataset}] missing percentage: {pct!r}"

    # All 10 indices `- N =` must appear
    for i in range(1, 11):
        assert f"- {i} = " in prompt, f"[{dataset}] missing bulleted index '- {i} ='"

    # Output format must request 1-10
    assert "Confidence: <1-10>" in prompt, f"[{dataset}] missing 'Confidence: <1-10>' format"
    assert "Correct: Yes or No" in prompt, f"[{dataset}] missing 'Correct: Yes or No' format"

    # Mapping reminder must appear
    assert "Confidence number MUST match the class" in prompt, \
        f"[{dataset}] missing class-mapping reminder"

    print(f"  [{dataset}] OK ({len(prompt)} chars)")


def check_extractors() -> None:
    """Sanity-check that the regex still pulls Confidence and Correct correctly."""
    cases = [
        ("Answer: 4\nConfidence: 7\nCorrect: Yes", 7, True),
        ("Answer: A\n**Confidence:** 9\n**Correct:** No", 9, False),
        ("Confidence: 8/10\nCorrect: Yes", 8, True),
        ("Confidence : 6\nCorrect: Yes", 6, True),
    ]
    for response, exp_conf, exp_correct in cases:
        conf = extract_verbalized_confidence(response, "gsm8k")
        correct = extract_more_likely_than_not(response)
        assert conf == exp_conf, f"extract_verbalized_confidence({response!r}) = {conf}, want {exp_conf}"
        assert correct == exp_correct, f"extract_more_likely_than_not({response!r}) = {correct}, want {exp_correct}"
    print("  extractors OK")


def check_two_pass_rubric() -> None:
    """Spot-check that get_two_pass_confidence's critique prompt embeds the same rubric."""
    # Read the source instead of executing the function (which would need a model)
    src_path = os.path.join(os.path.dirname(__file__), "confidence.py")
    with open(src_path) as f:
        src = f.read()

    # Locate critique_prompt definition and verify rubric labels + percentages appear
    for cls in CLASSES:
        assert f'"{cls}"' in src, f"[two-pass src] missing class label: {cls!r}"
    for pct in PERCENTAGES:
        assert pct in src, f"[two-pass src] missing percentage: {pct!r}"

    # Two distinct occurrences of the bulleted rubric: first-pass (_CONF_RUBRIC) and
    # both gen2 + two-pass critique. Each rubric line should appear in source.
    for i in range(1, 11):
        count = src.count(f'- {i} = "{CLASSES[i-1]}" ({PERCENTAGES[i-1]})')
        assert count >= 3, (
            f"line '- {i} = \"{CLASSES[i-1]}\" ({PERCENTAGES[i-1]})' "
            f"appears {count} times in source, expected at least 3 "
            f"(first-pass _CONF_RUBRIC + gen2 + two-pass critique)"
        )
    print("  rubric appears in first-pass + gen2 + two-pass in source")


def check_generate_with_logits_signature() -> None:
    """generate_with_logits must return 5 things now (text, probs, tokens, scores, info)."""
    import inspect
    from confidence import generate_with_logits

    src = inspect.getsource(generate_with_logits)
    assert "info = {" in src, "generate_with_logits should build an `info` dict"
    assert '"finish_reason"' in src, "info should carry 'finish_reason'"
    assert '"was_truncated"' in src, "info should carry 'was_truncated'"
    assert "return generated_text, token_probs, tokens, raw_scores, info" in src, \
        "generate_with_logits should return a 5-tuple ending with info"
    print("  generate_with_logits returns 5-tuple with truncation info OK")


def check_forced_answer_paths() -> None:
    """get_forced_answer must produce a dataset-specific forced-answer prompt
    that asks ONLY for the answer in the expected format. Mock generate_simple_response
    so we can probe what prompt the function would have sent."""
    import confidence
    from confidence import get_forced_answer

    captured = {}

    def fake_generate_simple_response(model, tokenizer, prompt, max_new_tokens=512, base_suffix=""):
        captured["prompt"] = prompt
        captured["max_new_tokens"] = max_new_tokens
        # Return a canned valid Answer line so extract_model_answer succeeds
        ds = captured["dataset"]
        return {
            "gsm8k": "Answer: 42",
            "mmlupro": "Answer: B",
            "strategyqa": "Answer: Yes",
            "medqa": "Answer: C",
            "triviaqa": "Answer: Paris",
        }[ds]

    # Monkey-patch the model_utils.generate_simple_response import inside
    # get_forced_answer (it imports lazily, so patch the module).
    import model_utils  # may not exist if torch was stubbed badly; guard
    real_simple = getattr(model_utils, "generate_simple_response", None)
    model_utils.generate_simple_response = fake_generate_simple_response

    try:
        cases = {
            "gsm8k": ("If x=2, what is 7x+8?", None, "42"),
            "mmlupro": ("Pick the right physics answer.",
                        ["A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "A9", "A10"], "B"),
            "strategyqa": ("Is water wet?", None, "Yes"),
            "medqa": ("Diagnosis?", ["A.foo", "B.bar", "C.baz", "D.qux", "E.zap"], "C"),
            "triviaqa": ("Capital of France?", None, "Paris"),
        }
        for ds, (q, choices, expected) in cases.items():
            captured["dataset"] = ds
            forced_answer, forced_resp = get_forced_answer(
                None, None, q, "[some truncated reasoning]", ds, choices
            )
            prompt = captured["prompt"]
            assert "ran out of thinking time" in prompt, f"[{ds}] forced prompt missing truncation framing"
            assert "Output ONLY" in prompt, f"[{ds}] forced prompt missing 'Output ONLY' instruction"
            assert "Answer:" in prompt, f"[{ds}] forced prompt missing 'Answer:' format"
            assert q in prompt, f"[{ds}] forced prompt missing original question"
            assert forced_answer == expected, \
                f"[{ds}] forced extraction got {forced_answer!r}, want {expected!r}"
            print(f"  [{ds}] forced-answer path OK (prompt={len(prompt)} chars, forced={forced_answer!r})")
    finally:
        if real_simple is not None:
            model_utils.generate_simple_response = real_simple


def check_evaluate_sample_columns() -> None:
    """evaluate_sample's result dict should now include truncation/forced columns."""
    src_path = os.path.join(os.path.dirname(__file__), "evaluation.py")
    with open(src_path) as f:
        src = f.read()
    for col in ("main_pass_finish_reason", "main_pass_was_truncated",
                "was_forced", "forced_answer_response"):
        assert f'"{col}"' in src, f"evaluation.py result dict missing column: {col}"
    # Both branches must consume the new 5-tuple from generate_with_logits
    assert src.count("response, token_probs, tokens, raw_scores, gen_info") >= 2, \
        "Both qwen3 and non-qwen3 branches should unpack the 5-tuple"
    # Both branches must call get_forced_answer when truncated
    assert src.count("get_forced_answer(") >= 2, \
        "Both branches should call get_forced_answer on truncation"
    print("  evaluate_sample exposes truncation + forced-answer columns OK")


def main():
    print("Verifying first-pass prompts...")
    for ds, (q, c) in DATASET_FIXTURES.items():
        check_prompt(ds, q, c)

    print("\nVerifying extractor regex...")
    check_extractors()

    print("\nVerifying gen2 + two-pass critique embed the same rubric...")
    check_two_pass_rubric()

    print("\nVerifying generate_with_logits returns truncation info...")
    check_generate_with_logits_signature()

    print("\nVerifying get_forced_answer prompt + extraction per dataset...")
    check_forced_answer_paths()

    print("\nVerifying evaluate_sample exposes new columns...")
    check_evaluate_sample_columns()

    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    main()
