#!/usr/bin/env python3
"""Golden-snapshot guard for the answer/confidence extractors.

The regex extraction stack (extract_model_answer, extract_model_answer_strict,
extract_verbalized_confidence, extract_more_likely_than_not,
is_refusal_response) has grown model-specific special cases; historically,
each fix risked silently changing behavior on other models' outputs. This
suite pins extractor outputs over REAL captured responses sampled from the
result CSVs in this repo, so any extractor change shows exactly which rows it
touches.

Usage:
    python3 check_extraction_golden.py            # verify (exit 1 on drift)
    python3 check_extraction_golden.py --update   # re-capture after an INTENTIONAL change

A failure after an "innocent" refactor means the refactor changed extraction
behavior on real data — inspect the diff before updating the golden.
"""
import argparse
import glob
import hashlib
import json
import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Only the pure-regex extractors are exercised — stub the heavy deps so this
# runs on machines without the ML stack (same approach as verify_rubric.py).
if "datasets" not in sys.modules:
    _ds = types.ModuleType("datasets")
    _ds.load_dataset = lambda *a, **k: None
    sys.modules["datasets"] = _ds
if "torch" not in sys.modules:
    _t = types.ModuleType("torch")
    _t.no_grad = lambda *a, **k: (lambda f: f)
    _t.softmax = lambda *a, **k: None
    sys.modules["torch"] = _t
if "numpy" not in sys.modules:
    try:
        import numpy  # noqa: F401
    except ImportError:
        _np = types.ModuleType("numpy")
        for k in ("array", "exp", "mean", "log", "min", "sum"):
            setattr(_np, k, lambda *a, **kw: 0)
        _np.float64 = float
        sys.modules["numpy"] = _np

import pandas as pd

from data_utils import (
    extract_model_answer,
    extract_model_answer_strict,
    is_refusal_response,
)
from confidence import extract_verbalized_confidence, extract_more_likely_than_not

HERE = os.path.dirname(os.path.abspath(__file__))
GOLDEN_PATH = os.path.join(HERE, "extraction_golden.json")

# Result files to sample real responses from (root + per-dataset dirs + Trash:
# older files exercise older models' failure modes, which is exactly the value).
CSV_GLOBS = [
    os.path.join(HERE, "*_confidence_*.csv"),
    os.path.join(HERE, "GSM8k", "**", "*.csv"),
    os.path.join(HERE, "MMLUPro", "**", "*.csv"),
    os.path.join(HERE, "StrategyQa", "**", "*.csv"),
    os.path.join(HERE, "Trash", "*.csv"),
]
ROWS_PER_FILE = 25
_DATASETS = ("gsm8k", "mmlupro", "strategyqa", "medqa", "triviaqa", "legalbench")


def infer_dataset(path: str):
    name = os.path.basename(path).lower()
    for d in _DATASETS:
        if d in name:
            return d
    # per-dataset directories carry the dataset in the path instead
    lower = path.lower()
    for d, hint in [("gsm8k", "gsm8k"), ("mmlupro", "mmlupro"),
                    ("strategyqa", "strategyqa"), ("strategyqa", "stratqa")]:
        if hint in lower:
            return d
    return None


def collect_cases():
    cases = {}
    files = sorted({f for g in CSV_GLOBS for f in glob.glob(g, recursive=True)})
    for path in files:
        ds = infer_dataset(path)
        if ds is None:
            continue
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        if "full_response" not in df.columns:
            continue
        sub = df[df["full_response"].notna()].head(ROWS_PER_FILE)
        rel = os.path.relpath(path, HERE)
        for i, resp in zip(sub.index, sub["full_response"]):
            resp = str(resp)
            # stable key: file + row + content hash (rows can shift if files change)
            key = f"{rel}#{i}#{hashlib.sha1(resp.encode('utf-8')).hexdigest()[:10]}"
            answer = extract_model_answer(resp, ds)
            cases[key] = {
                "dataset": ds,
                "answer": answer,
                "answer_strict": extract_model_answer_strict(resp, ds),
                "verbalized": extract_verbalized_confidence(resp, ds),
                "more_likely": extract_more_likely_than_not(resp),
                "is_refusal": is_refusal_response(resp, answer),
            }
    return cases


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--update", action="store_true")
    args = ap.parse_args()

    cases = collect_cases()
    if not cases:
        print("No result CSVs with full_response found — nothing to pin.")
        return 0

    if args.update or not os.path.exists(GOLDEN_PATH):
        with open(GOLDEN_PATH, "w", encoding="utf-8") as f:
            json.dump(cases, f, indent=1, ensure_ascii=False, sort_keys=True)
        print(f"Wrote {len(cases)} extraction cases to {GOLDEN_PATH}")
        return 0

    with open(GOLDEN_PATH, encoding="utf-8") as f:
        golden = json.load(f)

    drift = 0
    for key, now in cases.items():
        if key not in golden:
            continue  # content changed or new file — hash key won't collide
        for field, val in now.items():
            if golden[key].get(field) != val:
                print(f"DRIFT  {key} [{field}]: golden={golden[key].get(field)!r} now={val!r}")
                drift += 1
    matched = sum(1 for k in cases if k in golden)
    print(f"\nChecked {matched} pinned cases ({len(cases)} rendered): "
          f"{'FAIL — ' + str(drift) + ' drifted' if drift else 'PASS — no drift'}")
    return 1 if drift else 0


if __name__ == "__main__":
    raise SystemExit(main())
