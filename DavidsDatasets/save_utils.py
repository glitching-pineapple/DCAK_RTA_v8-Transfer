# save_utils.py - Saving results to files

import json
import numpy as np
import pandas as pd
from config import get_model_label, DATASET

class NumpyEncoder(json.JSONEncoder):
    """Custom encoder to handle numpy types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

class IncrementalJSONLWriter:
    """Crash-safe per-row results log.

    save_results() only runs after the WHOLE loop finishes, so a crash at row
    149/150 used to lose every completed evaluation. This writer appends each
    result as one JSON line and flushes immediately — after a crash, the
    completed rows are on disk and loadable with pd.read_json(path, lines=True).

    JSONL also round-trips raw model text exactly (no CSV mojibake — the
    'â¯'-style artifacts that reextract.py exists to repair), making it the
    preferred source of truth over the CSV for any re-analysis.
    """

    def __init__(self, path: str = None):
        self.path = path or f'{DATASET}_confidence_rows_{get_model_label()}.jsonl'
        self._fh = open(self.path, 'w', encoding='utf-8')
        print(f"Incremental results log: {self.path}")

    def write_row(self, row: dict):
        self._fh.write(json.dumps(row, cls=NumpyEncoder, ensure_ascii=False) + "\n")
        self._fh.flush()

    def close(self):
        if not self._fh.closed:
            self._fh.close()


def save_results(results: list, df: pd.DataFrame, errors: list = None):
    """Save results to JSON and CSV files.

    errors: optional list of {"idx", "error", "traceback"} dicts for samples
    that raised during evaluation. Written to a sidecar *_errors_* JSON so
    failed rows stay auditable without polluting the analysis CSV.
    """
    label = get_model_label()

    json_filename = f'{DATASET}_confidence_detailed_{label}.json'
    with open(json_filename, 'w') as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)
    print(f"Saved JSON to {json_filename}")

    csv_filename = f'{DATASET}_confidence_{label}.csv'
    df.to_csv(csv_filename, index=False)
    print(f"Saved CSV to {csv_filename}")

    if errors:
        err_filename = f'{DATASET}_errors_{label}.json'
        with open(err_filename, 'w') as f:
            json.dump(errors, f, indent=2, cls=NumpyEncoder)
        print(f"Saved {len(errors)} failed-sample records to {err_filename}")
