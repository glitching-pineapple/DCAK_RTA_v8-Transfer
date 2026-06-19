# save_utils.py - Saving results to files

import json
import torch
import numpy as np
import pandas as pd
from config import get_model_label, DATASET  # Update import

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

def save_results(results: list, df: pd.DataFrame):
    """Save results to JSON, CSV, and top-20 logits .pt file."""
    label = get_model_label()

    # Extract top-20 logit tensors before JSON/CSV serialization.
    # Each entry: {"idx": int, "top20_values": float16 (N,20), "top20_token_ids": int32 (N,20)}
    top20_entries = []
    has_top20 = False
    for r in results:
        entry = {"idx": r["idx"]}
        v = r.pop("top20_values", None)
        i = r.pop("top20_token_ids", None)
        if v is not None:
            entry["top20_values"] = v
            entry["top20_token_ids"] = i
            has_top20 = True
        top20_entries.append(entry)

    if has_top20:
        pt_filename = f'{DATASET}_top20_logits_{label}.pt'
        torch.save(top20_entries, pt_filename)
        print(f"Saved top-20 logits to {pt_filename}")

    # Drop tensor columns from df (already created in main.py before this call)
    df = df.drop(columns=["top20_values", "top20_token_ids"], errors="ignore")

    json_filename = f'{DATASET}_confidence_detailed_{label}.json'
    with open(json_filename, 'w') as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)
    print(f"Saved JSON to {json_filename}")

    csv_filename = f'{DATASET}_confidence_{label}.csv'
    df.to_csv(csv_filename, index=False)
    print(f"Saved CSV to {csv_filename}")
