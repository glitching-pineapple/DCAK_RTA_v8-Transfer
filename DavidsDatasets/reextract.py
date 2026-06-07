#!/usr/bin/env python3
"""Re-extraction utility — fix stale answer parses in existing result CSVs
without re-running the model (the "re-extraction vs re-inference" path).

Why this exists
---------------
The answer extractor in ``data_utils.py`` was fixed so that an empty ``Answer:``
line can no longer capture the following ``Confidence: N`` line (the post-colon
whitespace used to cross the newline). CSVs produced before that fix have the
bug baked in — e.g. Gemma2-9B-instruct TriviaQA rows where ``model_answer`` is
literally ``"Confidence: 3"``. This script re-parses ``full_response`` with the
current extractor and rewrites the affected fields in place.

It is deliberately CONSERVATIVE:
  * ``model_answer`` is recomputed for every row from ``full_response``.
  * ``is_correct`` is recomputed ONLY for rows whose parsed answer actually
    changed. Unchanged rows keep their original label, because the original
    TriviaQA correctness used alias-aware matching (the full alias set is NOT
    in the CSV — only ``ground_truth`` — so re-judging an unchanged row here
    could wrongly flip an alias match to wrong).
  * ``is_refusal`` (new column, handoff §19.3) is computed for all rows.
  * When a changed row now has no parseable answer, the verbalized/single-pass
    confidence fields are NaN-ed to match the live pipeline policy
    (evaluation.py), so abstentions don't pollute calibration.

Usage
-----
    python3 reextract.py "<glob-or-path>" [more paths ...]      # dry run (default)
    python3 reextract.py "<glob>" --write                       # apply + .bak backups
    python3 reextract.py "<glob>" --dataset triviaqa            # force dataset
"""
import argparse
import glob
import math
import os
import re
import sys

import pandas as pd

from data_utils import (
    extract_model_answer,
    answers_match,
    check_triviaqa_correct,
    is_refusal_response,
)

_DATASETS = ("gsm8k", "mmlupro", "strategyqa", "medqa", "triviaqa", "legalbench")

# The mis-parse signature: a stored model_answer that is really a leaked rubric
# line, e.g. "Confidence: 3" or "**Correct:** No". Only these rows are rewritten.
_BUG_SIGNATURE = re.compile(r'^\*{0,2}\s*(?:Confidence|Correct)\b', re.IGNORECASE)


def infer_dataset(path: str) -> str:
    name = os.path.basename(path).lower()
    for d in _DATASETS:
        if d in name:
            return d
    raise ValueError(
        f"Could not infer dataset from filename {name!r}; pass --dataset."
    )


def _norm(v) -> str:
    """Normalize a stored cell to a comparable string ('' for missing)."""
    if v is None:
        return ""
    if isinstance(v, float) and math.isnan(v):
        return ""
    s = str(v).strip()
    return "" if s.lower() in ("nan", "none") else s


def _recompute_is_correct(new_answer, ground_truth, dataset):
    if dataset == "triviaqa":
        # CSV lacks the alias set; build a minimal sample from ground_truth.
        sample = {"answer": {"value": str(ground_truth), "aliases": [str(ground_truth)]}}
        return check_triviaqa_correct(new_answer, sample)
    return answers_match(new_answer, ground_truth, dataset, None)


def process_file(path: str, dataset: str, write: bool):
    df = pd.read_csv(path)
    if "full_response" not in df.columns:
        print(f"  [skip] {os.path.basename(path)} — no full_response column")
        return None

    changed_rows = []
    new_model_answer = df["model_answer"].tolist()
    new_is_correct = df["is_correct"].tolist() if "is_correct" in df else [None] * len(df)
    new_aef = (
        df["answer_extraction_failed"].tolist()
        if "answer_extraction_failed" in df
        else [False] * len(df)
    )
    is_refusal_col = []

    # confidence fields we blank on newly-failed rows (matches evaluation.py).
    conf_nan_fields = ["verbalized_confidence", "single_pass_confidence"]
    conf_none_fields = ["more_likely_than_not", "single_pass_correct"]
    # the *_correct fields are bool dtype; cast to object so blanking them
    # doesn't trip pandas' incompatible-dtype warning.
    for f in conf_none_fields:
        if f in df.columns:
            df[f] = df[f].astype(object)

    for i, row in df.iterrows():
        resp = row["full_response"]
        stored = row.get("model_answer")
        stored_norm = _norm(stored)

        # ONLY rewrite the unambiguous mis-parse signature: an answer that is
        # actually a leaked rubric line ("Confidence: 3" / "Correct: No"),
        # produced by an older extractor when the model left "Answer:" blank.
        # Anything else is left exactly as generated — crucially, we do NOT
        # re-parse forced rows (their answer lives in forced_answer_response,
        # not full_response) or reformat clean numeric answers.
        is_bug = bool(_BUG_SIGNATURE.match(stored_norm))

        if not is_bug:
            # New column only; effective answer is whatever was stored.
            is_refusal_col.append(
                bool(is_refusal_response(resp, stored if stored_norm else None))
            )
            continue

        # Re-parse with the current extractor. For a genuinely blank Answer
        # line this returns None (the per-branch rubric guards reject the
        # leaked "Confidence:" text); if a real answer was recoverable it is
        # returned instead.
        parsed = extract_model_answer(resp, dataset) if isinstance(resp, str) else None
        gt = row.get("ground_truth")
        aef = parsed is None or parsed == ""
        ic = False if aef else bool(_recompute_is_correct(parsed, gt, dataset))
        refusal = bool(is_refusal_response(resp, parsed))
        is_refusal_col.append(refusal)

        new_model_answer[i] = parsed if parsed is not None else ""
        new_is_correct[i] = ic
        new_aef[i] = aef
        if aef:
            for f in conf_nan_fields:
                if f in df.columns:
                    df.at[i, f] = math.nan
            for f in conf_none_fields:
                if f in df.columns:
                    df.at[i, f] = None

        changed_rows.append(
            {
                "idx": row.get("idx"),
                "ground_truth": gt,
                "old_answer": stored_norm,
                "new_answer": _norm(parsed) or "(none)",
                "old_correct": row.get("is_correct"),
                "new_correct": ic,
                "is_refusal": refusal,
            }
        )

    df["model_answer"] = new_model_answer
    df["is_correct"] = new_is_correct
    if "answer_extraction_failed" in df.columns:
        df["answer_extraction_failed"] = new_aef
    df["is_refusal"] = is_refusal_col

    n_ref = sum(is_refusal_col)
    print(
        f"  {os.path.basename(path):55} rows={len(df):4d} "
        f"changed={len(changed_rows):3d} refusals={n_ref:3d}"
    )
    for c in changed_rows:
        print(
            f"      idx {str(c['idx']):>6}: {c['old_answer']!r} -> {c['new_answer']!r}"
            f"  (correct {c['old_correct']}->{c['new_correct']}, refusal={c['is_refusal']})"
            f"  | GT={c['ground_truth']}"
        )

    if write and (changed_rows or "is_refusal" not in pd.read_csv(path).columns):
        bak = path + ".bak"
        if not os.path.exists(bak):
            os.rename(path, bak)  # preserve the original exactly once
        df.to_csv(path, index=False)
        print(f"      WROTE {os.path.basename(path)} (backup: {os.path.basename(bak)})")

    return changed_rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("paths", nargs="+", help="CSV file(s) or glob(s)")
    ap.add_argument("--dataset", choices=_DATASETS, help="force dataset (else inferred)")
    ap.add_argument("--write", action="store_true", help="apply changes (.bak backups)")
    args = ap.parse_args()

    files = []
    for p in args.paths:
        files.extend(sorted(glob.glob(p)))
    if not files:
        print("No files matched.", file=sys.stderr)
        sys.exit(1)

    mode = "WRITE" if args.write else "DRY-RUN"
    print(f"Re-extraction [{mode}] over {len(files)} file(s)\n")
    total_changed = 0
    for f in files:
        ds = args.dataset or infer_dataset(f)
        changed = process_file(f, ds, args.write)
        if changed is not None:
            total_changed += len(changed)
    print(f"\nTotal changed rows: {total_changed}")
    if not args.write:
        print("(dry run — re-run with --write to apply)")


if __name__ == "__main__":
    main()
