# Test & verification suite

Everything here runs WITHOUT a GPU or the full ML stack (heavy deps are
stubbed; `torch` is needed only for the two tensor-logic tests).

## Run before any commit that touches the pipeline

```bash
cd DavidsDatasets

# Prompt construction is byte-pinned (prompts are part of the experimental
# setup — any byte change alters model behavior):
python3 check_prompt_golden.py

# Extractor behavior is pinned over 1,410 real captured model responses:
python3 check_extraction_golden.py

# Prompt/rubric/extraction invariants + forced-answer paths:
python3 verify_rubric.py

# Mock-model unit tests (needs torch, not transformers):
python3 tests/test_batched_mock.py        # batched == serial generation bookkeeping
python3 tests/test_clean_logits.py        # guarded-path clean re-scoring, 3 fallback branches
python3 tests/test_evaluate_sample_e2e.py # full result-dict assembly, merge + NaN policy
```

After an INTENTIONAL prompt or extractor change, inspect the diff, then
re-pin with `--update`:

```bash
python3 check_prompt_golden.py --update
python3 check_extraction_golden.py --update
```

## Run once per model family on the GPU box

```bash
# Validates batched decoding is equivalent to serial for YOUR model + stack
# before enabling DCAK_GEN1_BATCH_SIZE > 1:
python3 smoke_test_batched.py
```
