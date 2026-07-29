"""Test _clean_generated_logits: all three kwarg branches + guarded-path e2e."""
import sys, types
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.dirname(__import__("os").path.abspath(__file__))))
ds = types.ModuleType('datasets'); ds.load_dataset = lambda *a, **k: None
sys.modules['datasets'] = ds

import torch
from confidence import _clean_generated_logits, generate_with_logits

VOCAB = 50
SEQ = torch.arange(12).unsqueeze(0)  # (1, 12)
INPUT_LEN, NUM_GEN = 7, 5

def full_logits():
    # deterministic: logits[p, v] = p*1000 + v  → row identity is checkable
    g = torch.arange(12).unsqueeze(1) * 1000.0 + torch.arange(VOCAB).unsqueeze(0)
    return g

class HonorsKwarg:
    def __call__(self, seq, num_logits_to_keep=None, **kw):
        fl = full_logits()
        if num_logits_to_keep is not None:
            fl = fl[-num_logits_to_keep:]
        return types.SimpleNamespace(logits=fl.unsqueeze(0))

class RaisesKwarg:
    def __call__(self, seq, **kw):
        if kw:
            raise TypeError("unexpected kwarg")
        return types.SimpleNamespace(logits=full_logits().unsqueeze(0))

class IgnoresKwarg:
    def __call__(self, seq, **kw):  # silently ignores num_logits_to_keep
        return types.SimpleNamespace(logits=full_logits().unsqueeze(0))

expected = full_logits()[INPUT_LEN - 1: INPUT_LEN - 1 + NUM_GEN]
ok = True
for name, m in [("honors", HonorsKwarg()), ("raises", RaisesKwarg()), ("ignores", IgnoresKwarg())]:
    got = _clean_generated_logits(m, SEQ, INPUT_LEN, NUM_GEN)
    match = got.shape == (NUM_GEN, VOCAB) and torch.equal(got, expected)
    cpu = got.device.type == "cpu"
    print(f"  [{name}] shape={tuple(got.shape)} values_match={match} on_cpu={cpu}")
    ok &= match and cpu

# --- end-to-end: guarded generate_with_logits (guards forced via ngram kwarg)
# must produce identical token_probs/raw_scores to the unguarded path when the
# mock's generation is loop-free (the guard is inert).
class MockBatch(dict):
    __getattr__ = dict.__getitem__
    def to(self, device): return self

class MockTok:
    pad_token_id, eos_token_id = 0, 1
    def __call__(self, text, return_tensors="pt", padding=False):
        ids = [5, 6, 7]
        return MockBatch(input_ids=torch.tensor([ids]),
                         attention_mask=torch.ones(1, 3, dtype=torch.long))
    def decode(self, ids, skip_special_tokens=False):
        if hasattr(ids, "tolist"): ids = ids.tolist()
        return "".join(f"<{t}>" for t in ids
                       if not (skip_special_tokens and t in (0, 1)))

class MockGenModel:
    device = "cpu"
    PLAN = [10, 11, 12, 1]  # 3 tokens + EOS
    def generate(self, input_ids=None, max_new_tokens=None, **kw):
        gen = torch.tensor([self.PLAN])
        scores = []
        for t, tok in enumerate(self.PLAN):
            l = torch.full((1, VOCAB), -20.0); l[0, tok] = 20.0
            scores.append(l)
        return types.SimpleNamespace(
            sequences=torch.cat([input_ids, gen], dim=1), scores=tuple(scores))
    def __call__(self, seq, **kw):
        if any(k in kw for k in ("num_logits_to_keep", "logits_to_keep")):
            raise TypeError
        # teacher-forced logits consistent with PLAN: position p predicts seq[p+1]
        L = seq.shape[1]
        fl = torch.full((L, VOCAB), -20.0)
        for p in range(L - 1):
            fl[p, seq[0, p + 1]] = 20.0
        return types.SimpleNamespace(logits=fl.unsqueeze(0))

tok, m = MockTok(), MockGenModel()
un = generate_with_logits(m, tok, "x", max_new_tokens=16)                       # unguarded
gu = generate_with_logits(m, tok, "x", max_new_tokens=16, no_repeat_ngram_size=3)  # guarded
same_text = un[0] == gu[0]
same_probs = torch.allclose(torch.tensor(un[1]), torch.tensor(gu[1]))
same_scores = all(torch.allclose(a.float(), b.float()) for a, b in zip(un[3], gu[3]))
guard_flag = (un[4]["decoding_guards_active"], gu[4]["decoding_guards_active"]) == (False, True)
print(f"  [e2e] text={same_text} probs={same_probs} scores={same_scores} guard_flags={guard_flag}")
ok &= same_text and same_probs and same_scores and guard_flag

print("ALL PASS" if ok else "FAILURES PRESENT")
sys.exit(0 if ok else 1)
