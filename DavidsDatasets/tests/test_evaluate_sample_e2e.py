"""End-to-end evaluate_sample test with mocked generation.

Exercises the full result-dict assembly no other test covers:
  - gen1_precomputed path
  - answer extraction -> correctness via answers_match
  - Gen2/Gen3 merge + source tracking (two_pass primary, single_pass fallback)
  - hard-failure NaN policy (answer_extraction_failed)
  - logit metric columns (mean vs sum) and last-token variants
"""
import sys, types, math
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.dirname(__import__("os").path.abspath(__file__))))
ds_stub = types.ModuleType('datasets'); ds_stub.load_dataset = lambda *a, **k: None
sys.modules['datasets'] = ds_stub
tf = types.ModuleType('transformers')
tf.AutoModelForCausalLM = type("A", (), {}); tf.AutoTokenizer = type("B", (), {})
sys.modules['transformers'] = tf

import numpy as np
import config
config.DATASET = "medqa"
config.MODEL_VARIANT = "instruct"

import evaluation

DATASET_ROWS = [
    {"question": "Q0?", "options": {"A": "opt a", "B": "opt b", "C": "opt c",
                                    "D": "opt d", "E": "opt e"}, "answer_idx": "B"},
]

def fake_gen1(text, probs):
    tokens = []          # empty -> answer-token-entropy returns its null/nan form
    raw_scores = []
    meta = {"finish_reason": "eos", "was_truncated": False, "decoding_guards_active": False}
    return (text, probs, tokens, raw_scores, meta)

ok = True
def check(name, cond, detail=""):
    global ok
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  ({detail})" if detail and not cond else ""))
    ok &= cond

# ---- Case 1: clean answer, both Gen2 and Gen3 succeed ----
evaluation.get_gen2_confidence = lambda *a, **k: {
    "gen2_confidence": 7.0, "gen2_correct": True, "gen2_response": "x"}
evaluation.get_two_pass_confidence = lambda *a, **k: {
    "two_pass_confidence": 6.0, "two_pass_correct": True, "two_pass_critique": "c",
    "two_pass_finish_reason": "eos", "two_pass_was_truncated": False}
evaluation.get_forced_answer = lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not force"))

r = evaluation.evaluate_sample(None, None, DATASET_ROWS, 0,
    gen1_precomputed=fake_gen1("Reasoning here.\nAnswer: B\nConfidence: 8\nCorrect: Yes",
                               [0.9, 0.8, 0.5]))
check("model_answer == B", r["model_answer"] == "B", r["model_answer"])
check("is_correct", r["is_correct"] is True or r["is_correct"] == True)
check("verbalized = two_pass value", r["verbalized_confidence"] == 6.0)
check("verbalized source = two_pass", r["verbalized_conf_source"] == "two_pass")
check("single_pass stored separately", r["single_pass_confidence"] == 7.0)
check("two_pass stored separately", r["two_pass_confidence"] == 6.0)
exp_mean = float(np.mean(np.log(np.array([0.9, 0.8, 0.5]) + 1e-10)))
exp_sum = float(np.sum(np.log(np.array([0.9, 0.8, 0.5]) + 1e-10)))
check("seq_confidence_mean is the MEAN", abs(r["seq_confidence_mean"] - exp_mean) < 1e-12)
check("seq_log_prob_sum is the SUM", abs(r["seq_log_prob_sum"] - exp_sum) < 1e-12)
check("last-token mean == log(last prob)",
      abs(r["seq_confidence_mean_last_token"] - math.log(0.5 + 1e-10)) < 1e-12)
check("guards flag surfaced", r["decoding_guards_active"] is False)
check("not forced", r["was_forced"] is False)

# ---- Case 2: two-pass fails -> falls back to single_pass, source recorded ----
evaluation.get_two_pass_confidence = lambda *a, **k: {
    "two_pass_confidence": None, "two_pass_correct": None, "two_pass_critique": "",
    "two_pass_finish_reason": "length", "two_pass_was_truncated": True}
r = evaluation.evaluate_sample(None, None, DATASET_ROWS, 0,
    gen1_precomputed=fake_gen1("Reasoning.\nAnswer: C\nConfidence: 4\nCorrect: No", [0.7]))
check("fallback value = single_pass", r["verbalized_confidence"] == 7.0)
check("fallback source recorded", r["verbalized_conf_source"] == "single_pass")
check("wrong answer marked incorrect", r["is_correct"] == False)

# ---- Case 3: no parseable answer, forcing also fails -> NaN policy ----
evaluation.get_forced_answer = lambda *a, **k: (None, "no luck")
r = evaluation.evaluate_sample(None, None, DATASET_ROWS, 0,
    gen1_precomputed=fake_gen1("I rambled with no committed answer at all.", [0.6]))
check("extraction failed flagged", r["answer_extraction_failed"] is True)
check("was_forced flagged", r["was_forced"] is True)
check("is_correct False", r["is_correct"] == False)
check("verbalized NaN", isinstance(r["verbalized_confidence"], float)
      and math.isnan(r["verbalized_confidence"]))
check("source cleared", r["verbalized_conf_source"] is None)
check("more_likely None", r["more_likely_than_not"] is None)

print("\nALL PASS" if ok else "\nFAILURES PRESENT")
sys.exit(0 if ok else 1)
