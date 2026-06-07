# tests/test_preprocessing.py
import pytest
import pandas as pd
import numpy as np
from src.data_preprocessing import engineer_features

def test_engineer_features_calculation():
    """Verify time difference metric computations are accurate and type-safe."""
    sample_data = pd.DataFrame({
        "user_id": [1, 1],
        "device_id": ["D1", "D1"],
        "signup_time": pd.to_datetime(["2026-06-01 10:00:00", "2026-06-01 10:00:00"]),
        "purchase_time": pd.to_datetime(["2026-06-01 10:30:00", "2026-06-01 11:00:00"])
    })
    
    processed = engineer_features(sample_data)
    
    # 30 minutes delta calculation check
    assert processed.loc[0, "time_since_signup"] == 30.0
    # User count grouping validation
    assert processed.loc[0, "user_transaction_count"] == 2