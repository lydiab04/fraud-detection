# src/data_preprocessing.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def load_and_clean_fraud(fraud_path, ip_path):
    """Loads, cleans, and merges fraud dataset with geolocation data safely."""
    try:
        fraud = pd.read_csv(fraud_path, parse_dates=["signup_time", "purchase_time"])
        ip = pd.read_csv(ip_path)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Error loading source files: {e}")

    # Deduplicate
    fraud = fraud.drop_duplicates()
    
    # Cast IPs to float for range assessment evaluation
    fraud["ip_address"] = fraud["ip_address"].astype(float)
    ip["lower_bound_ip_address"] = ip["lower_bound_ip_address"].astype(float)
    ip["upper_bound_ip_address"] = ip["upper_bound_ip_address"].astype(float)
    
    # Sort for merge_asof execution context
    fraud = fraud.sort_values("ip_address")
    ip = ip.sort_values("lower_bound_ip_address")
    
    # Merge geolocation records
    fraud_geo = pd.merge_asof(
        fraud, ip, 
        left_on="ip_address", 
        right_on="lower_bound_ip_address", 
        direction="backward"
    )
    
    # Flag invalid locations instead of dropping observations completely
    invalid_idx = fraud_geo["ip_address"] > fraud_geo["upper_bound_ip_address"]
    fraud_geo.loc[invalid_idx, "country"] = "Unknown"
    fraud_geo["country"] = fraud_geo["country"].fillna("Unknown")
    
    # Drop intermediate indexing columns safely
    fraud_geo.drop(columns=["lower_bound_ip_address", "upper_bound_ip_address"], errors="ignore", inplace=True)
    return fraud_geo

def engineer_features(df):
    """Generates temporal, velocity, and frequency indicators."""
    df = df.copy()
    
    # Core temporal components
    df["time_since_signup"] = (df["purchase_time"] - df["signup_time"]).dt.total_seconds() / 60.0
    df["hour_of_day"] = df["purchase_time"].dt.hour
    df["day_of_week"] = df["purchase_time"].dt.dayofweek
    
    # Frequency network parameters
    df["user_transaction_count"] = df.groupby("user_id")["user_id"].transform("count")
    df["device_count"] = df.groupby("device_id")["device_id"].transform("count")
    
    return df