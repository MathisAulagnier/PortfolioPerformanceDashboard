# utils.py
import pandas as pd
import numpy as np

def convert_for_json(obj):
    """Convertit récursivement les objets pandas/numpy en types Python natifs pour la sérialisation JSON."""
    if isinstance(obj, dict):
        return {key: convert_for_json(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_for_json(item) for item in obj]
    elif isinstance(obj, (pd.Series, pd.DataFrame)):
        return convert_for_json(obj.to_dict())
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return convert_for_json(obj.tolist())
    elif pd.isna(obj):
        return None
    else:
        return obj