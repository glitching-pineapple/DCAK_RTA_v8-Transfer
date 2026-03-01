# save_utils.py - Saving results to files

import json
import numpy as np
import pandas as pd
from config import MODEL_VARIANT, DATASET  


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
    """Save results to JSON and CSV files."""
    # Save JSON with full data
    json_filename = f'{DATASET}_confidence_detailed_{MODEL_VARIANT}.json'  # Add DATASET
    with open(json_filename, 'w') as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)
    print(f"Saved JSON to {json_filename}")
    
    # Save CSV summary
    csv_filename = f'{DATASET}_confidence_{MODEL_VARIANT}.csv'  # Add DATASET
    df.to_csv(csv_filename, index=False)
    print(f"Saved CSV to {csv_filename}")
