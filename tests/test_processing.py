# tests/test_preprocessing.py
import pandas as pd
from src.data_processing import clean_data

def test_duplicates():
    df = pd.DataFrame({
        "A": [1, 1, 2]
    })
    result = clean_data(df)
    assert len(result) == 2