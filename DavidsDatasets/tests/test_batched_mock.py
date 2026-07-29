"""Mock-model equivalence test: generate_with_logits_batched vs serial.

Validates the batching bookkeeping that could silently corrupt token_probs:
  - left-padding alignment (scores[t][i] indexing)
  - right-side pad trimming, including the pad==eos single-EOS retention rule
  - finish_reason / was_truncated classification per sample
No transformers needed: mock tokenizer + mock deterministic greedy model.
"""
import sys, types
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.dirname(__import__("os").path.abspath(__file__))))

# stub the `datasets` module so confidence->config imports cleanly
ds = types.ModuleType('datasets'); ds.load_dataset = lambda *a, **k: None
sys.modules['datasets'] = ds

import torch
from confidence import generate_with_logits, generate_with_logits_batched


class MockBatch(dict):
    __getattr__ = dict.__getitem__
    def to(self, device):
        return self


class MockTokenizer:
    def __init__(self, pad_id, eos_id):
        self.pad_token_id = pad_id
        self.eos_token_id = eos_id
        self.padding_side = "right"
        self._vocab = {}  # prompt -> ids

    def register(self, prompt, ids):
        self._vocab[prompt] = ids

    def __call__(self, text, return_tensors="pt", padding=False):
        if isinstance(text, str):
            ids = self._vocab[text]
            return MockBatch(
                input_ids=torch.tensor([ids]),
                attention_mask=torch.ones(1, len(ids), dtype=torch.long),
            )
        seqs = [self._vocab[t] for t in text]
        max_len = max(len(s) for s in seqs)
        input_ids, attn = [], []
        for s in seqs:
            npad = max_len - len(s)
            if self.padding_side == "left":
                input_ids.append([self.pad_token_id] * npad + s)
                attn.append([0] * npad + [1] * len(s))
            else:
                input_ids.append(s + [self.pad_token_id] * npad)
                attn.append([1] * len(s) + [0] * npad)
        return MockBatch(
            input_ids=torch.tensor(input_ids),
            attention_mask=torch.tensor(attn),
        )

    def decode(self, ids, skip_special_tokens=False):
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        out = []
        for t in ids:
            if skip_special_tokens and t in (self.pad_token_id, self.eos_token_id):
                continue
            out.append(f"<{t}>")
        return "".join(out)


class MockModel:
    """Deterministic greedy generator. Each row's continuation depends ONLY on
    its last real (non-pad) input token, so serial and batched runs of the
    same prompt must produce identical tokens."""
    device = "cpu"
    VOCAB = 100

    def __init__(self, eos_id):
        self.eos_id = eos_id

    def _plan(self, seed, max_new):
        # (seed % 3) + 2 content tokens, then EOS — truncated at max_new
        n = (seed % 3) + 2
        toks = [10 + ((seed + k) % 80) for k in range(n)] + [self.eos_id]
        return toks[:max_new]

    def generate(self, input_ids=None, attention_mask=None, max_new_tokens=None,
                 do_sample=False, return_dict_in_generate=True, output_scores=True,
                 pad_token_id=None, eos_token_id=None, **kw):
        B = input_ids.shape[0]
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        plans = []
        for i in range(B):
            real = input_ids[i][attention_mask[i].bool()]
            plans.append(self._plan(int(real[-1]), max_new_tokens))
        steps = max(len(p) for p in plans)
        gen = torch.full((B, steps), pad_token_id if pad_token_id is not None else 0,
                         dtype=input_ids.dtype)
        scores = []
        for t in range(steps):
            logits = torch.full((B, self.VOCAB), -20.0)
            for i in range(B):
                chosen = plans[i][t] if t < len(plans[i]) else pad_token_id
                logits[i, chosen] = 20.0
                gen[i, t] = chosen
            scores.append(logits)
        return types.SimpleNamespace(
            sequences=torch.cat([input_ids, gen], dim=1),
            scores=tuple(scores),
        )


def compare(name, tok, model, prompts, max_new):
    batched = generate_with_logits_batched(model, tok, prompts,
                                           max_new_tokens=max_new, batch_size=len(prompts))
    ok = True
    for i, p in enumerate(prompts):
        serial = generate_with_logits(model, tok, p, max_new_tokens=max_new)
        s_text, s_probs, s_toks, s_scores, s_meta = serial
        b_text, b_probs, b_toks, b_scores, b_meta = batched[i]
        checks = [
            ("text", s_text == b_text, (s_text, b_text)),
            ("tokens", s_toks == b_toks, (s_toks, b_toks)),
            ("n_scores", len(s_scores) == len(b_scores), (len(s_scores), len(b_scores))),
            ("probs", torch.allclose(torch.tensor(s_probs), torch.tensor(b_probs)), (s_probs, b_probs)),
            ("scores", all(torch.allclose(a, b) for a, b in zip(s_scores, b_scores)), None),
            ("meta", s_meta == b_meta, (s_meta, b_meta)),
        ]
        for cname, passed, detail in checks:
            if not passed:
                print(f"  [{name}] prompt {i} MISMATCH {cname}: {detail}")
                ok = False
    print(f"  [{name}] {'PASS' if ok else 'FAIL'} ({len(prompts)} prompts)")
    return ok


all_ok = True

# --- Case 1: pad != eos, varying prompt lengths (exercises left-pad alignment)
tok = MockTokenizer(pad_id=0, eos_id=1)
model = MockModel(eos_id=1)
tok.register("pA", [5, 6, 7])          # seed 7 -> 3 content tokens
tok.register("pB", [5, 6, 7, 8, 9])    # seed 9 -> 2 content tokens (finishes first)
tok.register("pC", [5, 11])            # seed 11 -> 4 content tokens (longest)
all_ok &= compare("pad!=eos", tok, model, ["pA", "pB", "pC"], max_new=16)

# --- Case 2: pad == eos (the single-EOS retention rule)
tok2 = MockTokenizer(pad_id=1, eos_id=1)
model2 = MockModel(eos_id=1)
tok2.register("qA", [5, 6, 7])
tok2.register("qB", [5, 6, 7, 8, 9])
tok2.register("qC", [5, 11])
all_ok &= compare("pad==eos", tok2, model2, ["qA", "qB", "qC"], max_new=16)

# --- Case 3: truncation (max_new_tokens smaller than plan; no EOS emitted)
all_ok &= compare("truncated", tok, model, ["pA", "pC"], max_new=3)

# --- Case 4: mixed — one row truncates, one finishes with EOS
all_ok &= compare("mixed", tok2, model2, ["qB", "qC"], max_new=4)

print("\nALL PASS" if all_ok else "\nFAILURES PRESENT")
sys.exit(0 if all_ok else 1)
