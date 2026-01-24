# save_utils.py - Saving results to files

import json
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
    """Save results to JSON and CSV files."""
    label = get_model_label()
    
    json_filename = f'{DATASET}_confidence_detailed_{label}.json'
    with open(json_filename, 'w') as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)
    print(f"Saved JSON to {json_filename}")
    
    csv_filename = f'{DATASET}_confidence_{label}.csv'
    df.to_csv(csv_filename, index=False)
    print(f"Saved CSV to {csv_filename}")
