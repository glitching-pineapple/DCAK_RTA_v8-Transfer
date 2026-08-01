#!/usr/bin/env python3
# ============================================================================
# compute_metrics_colab.py — add calibration/quality metrics to result CSVs
#
# Works standalone in Google Colab or locally. Only needs pandas + numpy.
#
# WHAT IT DOES, per input CSV:
#   1. Adds per-row columns (written to <name>_with_metrics.csv, originals
#      untouched):
#        conf_p        - verbalized rating mapped to probability via the
#                        rubric midpoint: rating N -> (N - 0.5) / 10
#                        (the prompt defines N as "(N-1)*10%..N*10% likely")
#        brier_row     - (conf_p - is_correct)^2
#        logloss_row   - -[y*ln(p) + (1-y)*ln(1-p)]
#        (same three with _sp suffix for single_pass_confidence when present)
#   2. Computes aggregate metrics -> one row in metrics_summary.csv per file,
#      AND per seed within a file when a `seed` column exists:
#        n, n_scored, accuracy,
#        ECE (rating-level bins), mean_conf_p, overconfidence gap,
#        brier, logloss,
#        AUROC for every known confidence/uncertainty column present,
#        metacognition (more_likely_than_not vs is_correct):
#           mln_accuracy, mln_precision, mln_recall, mln_f1
#
# ROW POLICY:
#   - exact duplicate rows are dropped for aggregates (reported)
#   - rows with answer_extraction_failed=True or missing verbalized
#     confidence are excluded from calibration metrics (ECE/brier/logloss)
#     but INCLUDED in task accuracy as incorrect (matches pipeline policy)
#
# USAGE (Colab):
#   1. Upload your CSVs (or mount Drive) into a folder, e.g. "results/"
#   2. Set INPUT_GLOBS below (or leave "*.csv" and put this file next to them)
#   3. Run:  python compute_metrics_colab.py     (or run the cell)
#   Outputs land in OUTPUT_DIR.
#
# POOLING (optional): to get one combined metrics row for a model-benchmark
# pair whose 300 samples are split across two files (e.g. 150 with top-20
# logits + 150 without), list them together in POOL below. Metrics that need
# columns missing from one file are computed over the rows that have them.
# ============================================================================

import glob
import math
import os

import numpy as np
import pandas as pd

# ----------------------------- CONFIG ---------------------------------------
INPUT_GLOBS = [
    "*.csv",                     # adjust to where your result files live
    # "results/**/*.csv",
]
OUTPUT_DIR = "with_metrics"

# Optional pooled groups: {"pool-name": [list of globs]}
POOL = {
    # "gemma2instruct-gsm8k-300": [
    #     "Gemma2_GSM8k_150combined_reruntop20.csv",
    #     "150gsm8kGemma2-9B-instruct - 1.csv",
    # ],
}

# Confidence/uncertainty columns to compute AUROC for, with direction.
# higher_is_better=True  -> higher value should mean MORE likely correct
# higher_is_better=False -> higher value means more UNCERTAIN (flipped)
# Missing columns are skipped silently per file.
AUROC_COLUMNS = [
    ("verbalized_confidence",               True),
    ("single_pass_confidence",              True),
    ("two_pass_confidence",                 True),
    ("logit_confidence_geom",               True),
    ("logit_confidence_mean_prob",          True),
    ("logit_confidence_min",                True),
    # NOTE: in files from before 2026-07 this column holds the log-prob SUM
    # (length-confounded). Kept for completeness; prefer logit_confidence_geom.
    ("seq_confidence_mean",                 True),
    ("logit_confidence_geom_last_token",    True),
    ("logit_confidence_mean_prob_last_token", True),
    ("geom_content",                        True),
    ("prob_margin_mean",                    True),
    ("top20_entropy_mean",                  False),
    ("top20_entropy_last_token",            False),
    ("entropy_mean_content",                False),
    ("entropy_max",                         False),
]

# ----------------------------- HELPERS --------------------------------------

def to_bool(x):
    """Robust bool parsing: True/False, 'TRUE'/'FALSE', 'true', 1/0, NaN->None."""
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (int, float, np.integer, np.floating)):
        if isinstance(x, float) and math.isnan(x):
            return None
        return bool(x)
    if isinstance(x, str):
        s = x.strip().lower()
        if s in ("true", "yes", "1"):
            return True
        if s in ("false", "no", "0"):
            return False
    return None


def rank_auroc(labels, scores):
    """AUROC via rank comparison (no sklearn). labels: bool array, scores: float."""
    labels = np.asarray(labels, dtype=bool)
    scores = np.asarray(scores, dtype=float)
    ok = ~np.isnan(scores)
    labels, scores = labels[ok], scores[ok]
    pos, neg = scores[labels], scores[~labels]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    greater = (pos[:, None] > neg[None, :]).sum()
    equal = (pos[:, None] == neg[None, :]).sum()
    return float((greater + 0.5 * equal) / (len(pos) * len(neg)))


def rating_to_p(rating):
    """Rubric-midpoint mapping: rating N (1-10) -> probability (N-0.5)/10."""
    p = (np.asarray(rating, dtype=float) - 0.5) / 10.0
    return np.clip(p, 0.001, 0.999)  # clip only guards malformed ratings


def ece_by_rating(ratings, correct):
    """ECE with one bin per discrete rating level (adaptive; exact for 1-10
    integer ratings). Compares each level's empirical accuracy to its stated
    midpoint probability, weighted by bin size."""
    df = pd.DataFrame({"r": ratings, "y": correct.astype(float)}).dropna()
    if len(df) == 0:
        return np.nan
    total = len(df)
    ece = 0.0
    for r, grp in df.groupby("r"):
        stated = float(np.mean(rating_to_p(grp["r"])))
        acc = float(grp["y"].mean())
        ece += (len(grp) / total) * abs(acc - stated)
    return float(ece)


def prf1(pred, truth):
    """Precision/recall/F1 treating pred=True as 'predicts correct'."""
    pred = np.asarray(pred, dtype=bool)
    truth = np.asarray(truth, dtype=bool)
    tp = int(np.sum(pred & truth))
    fp = int(np.sum(pred & ~truth))
    fn = int(np.sum(~pred & truth))
    prec = tp / (tp + fp) if (tp + fp) else np.nan
    rec = tp / (tp + fn) if (tp + fn) else np.nan
    f1 = (2 * prec * rec / (prec + rec)
          if prec == prec and rec == rec and (prec + rec) else np.nan)
    return prec, rec, f1


# ----------------------------- CORE -----------------------------------------

def add_row_metrics(df):
    """Add per-row conf_p / brier / logloss columns (verbalized + single-pass)."""
    y = df["_is_correct"].astype(float)

    for src_col, suffix in [("verbalized_confidence", ""),
                            ("single_pass_confidence", "_sp")]:
        if src_col not in df.columns:
            continue
        conf = pd.to_numeric(df[src_col], errors="coerce")
        p = pd.Series(rating_to_p(conf), index=df.index)
        p[conf.isna()] = np.nan
        df[f"conf_p{suffix}"] = p.round(4)
        df[f"brier_row{suffix}"] = ((p - y) ** 2).round(6)
        df[f"logloss_row{suffix}"] = (
            -(y * np.log(p) + (1 - y) * np.log(1 - p))).round(6)
    return df


def aggregate_metrics(df, label):
    """One summary dict for a (file or file+seed or pool) slice."""
    out = {"slice": label, "n": len(df)}

    y_all = df["_is_correct"]
    scored = df[y_all.notna()]
    out["n_scored"] = len(scored)
    if len(scored) == 0:
        return out
    y = scored["_is_correct"].astype(bool).values
    out["accuracy"] = round(float(np.mean(y)), 4)

    # --- calibration on verbalized confidence (valid rows only) ---
    if "verbalized_confidence" in scored.columns:
        conf = pd.to_numeric(scored["verbalized_confidence"], errors="coerce")
        aef = scored.get("_aef", pd.Series(False, index=scored.index))
        valid = conf.notna() & ~aef.fillna(False).astype(bool)
        vc, vy = conf[valid], scored["_is_correct"][valid].astype(bool)
        out["n_calibration"] = int(valid.sum())
        if valid.sum() >= 2:
            p = rating_to_p(vc)
            yv = vy.values.astype(float)
            out["ece"] = round(ece_by_rating(vc, vy), 4)
            out["mean_conf_p"] = round(float(np.mean(p)), 4)
            out["overconfidence"] = round(float(np.mean(p) - np.mean(yv)), 4)
            out["brier"] = round(float(np.mean((p - yv) ** 2)), 4)
            out["logloss"] = round(float(np.mean(
                -(yv * np.log(p) + (1 - yv) * np.log(1 - p)))), 4)

    # --- AUROC for every known confidence column present ---
    for col, hib in AUROC_COLUMNS:
        if col in scored.columns:
            s = pd.to_numeric(scored[col], errors="coerce").values
            if not hib:
                s = -s
            auc = rank_auroc(y, s)
            if auc == auc:
                out[f"auroc_{col}"] = round(auc, 4)

    # --- metacognition: does the model KNOW when it's right? ---
    if "more_likely_than_not" in scored.columns:
        mln = scored["more_likely_than_not"].map(to_bool)
        m = mln.notna()
        if m.sum() >= 2:
            pred = mln[m].astype(bool).values
            truth = scored["_is_correct"][m].astype(bool).values
            out["mln_n"] = int(m.sum())
            out["mln_accuracy"] = round(float(np.mean(pred == truth)), 4)
            prec, rec, f1 = prf1(pred, truth)
            out["mln_precision"] = round(prec, 4) if prec == prec else np.nan
            out["mln_recall"] = round(rec, 4) if rec == rec else np.nan
            out["mln_f1"] = round(f1, 4) if f1 == f1 else np.nan

    return out


def load_and_prepare(path):
    df = pd.read_csv(path)
    n0 = len(df)
    df = df.drop_duplicates()
    if len(df) < n0:
        print(f"    note: dropped {n0 - len(df)} exact duplicate rows")

    if "is_correct" not in df.columns:
        print("    SKIP: no is_correct column")
        return None
    df["_is_correct"] = df["is_correct"].map(to_bool)
    if "answer_extraction_failed" in df.columns:
        df["_aef"] = df["answer_extraction_failed"].map(to_bool)
    return df


def process_file(path):
    print(f"\n=== {os.path.basename(path)} ===")
    df = load_and_prepare(path)
    if df is None:
        return []

    df = add_row_metrics(df)

    # write augmented per-row file (originals untouched)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    stem = os.path.splitext(os.path.basename(path))[0]
    out_path = os.path.join(OUTPUT_DIR, f"{stem}_with_metrics.csv")
    df.drop(columns=[c for c in ("_is_correct", "_aef") if c in df.columns]) \
      .to_csv(out_path, index=False)
    print(f"    wrote {out_path}")

    # aggregate: whole file + per seed if present
    rows = [aggregate_metrics(df, stem)]
    if "seed" in df.columns and df["seed"].nunique() > 1:
        for seed, grp in df.groupby("seed"):
            rows.append(aggregate_metrics(grp, f"{stem} [seed={seed}]"))
    for r in rows:
        acc = r.get("accuracy", "n/a")
        ece = r.get("ece", "n/a")
        print(f"    {r['slice']}: n={r['n']} acc={acc} ECE={ece} "
              f"brier={r.get('brier', 'n/a')} logloss={r.get('logloss', 'n/a')} "
              f"mln_f1={r.get('mln_f1', 'n/a')}")
    return rows


def main():
    files = sorted({f for g in INPUT_GLOBS for f in glob.glob(g, recursive=True)})
    files = [f for f in files if "_with_metrics" not in f
             and os.path.basename(f) != "metrics_summary.csv"]
    if not files:
        print("No CSVs matched INPUT_GLOBS — edit the config at the top.")
        return

    all_rows = []
    for f in files:
        try:
            all_rows.extend(process_file(f))
        except Exception as e:
            print(f"    ERROR on {f}: {type(e).__name__}: {e}")

    # pooled groups
    for name, globs in POOL.items():
        pool_files = sorted({f for g in globs for f in glob.glob(g)})
        if not pool_files:
            continue
        print(f"\n=== POOL: {name} ({len(pool_files)} files) ===")
        parts = [load_and_prepare(p) for p in pool_files]
        parts = [p for p in parts if p is not None]
        if parts:
            pooled = pd.concat(parts, ignore_index=True, sort=False)
            r = aggregate_metrics(pooled, f"POOL:{name}")
            print(f"    n={r['n']} acc={r.get('accuracy')} ECE={r.get('ece')}")
            all_rows.append(r)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    summary = pd.DataFrame(all_rows)
    summary_path = os.path.join(OUTPUT_DIR, "metrics_summary.csv")
    summary.to_csv(summary_path, index=False)
    print(f"\nSummary: {len(summary)} rows -> {summary_path}")


if __name__ == "__main__":
    main()
