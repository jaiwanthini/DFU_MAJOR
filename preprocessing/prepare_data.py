"""
==========================================================
Smart Insole DFU Risk Prediction
prepare_data.py

Reads:
    - dfu_raw_dataset_50000.csv
    - patient_profile.xlsx

Creates:
    - dfu_processed_dataset.csv
==========================================================
"""

import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ---------------------------------------------------------
# Allow importing from project root
# ---------------------------------------------------------

ROOT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

sys.path.append(ROOT_DIR)

# ---------------------------------------------------------
# Import Config
# ---------------------------------------------------------

from config import (
    RAW_DATA_PATH,
    PATIENT_PROFILE_PATH,
    PROCESSED_DATA_PATH,
    SEQUENCE_WINDOW,
    ROLLING_WINDOW,
    MODEL_FEATURE_COLUMNS
)

# Resolve all paths relative to project root
RAW_DATA_PATH      = os.path.join(ROOT_DIR, RAW_DATA_PATH)
PATIENT_PROFILE_PATH = os.path.join(ROOT_DIR, PATIENT_PROFILE_PATH)
PROCESSED_DATA_PATH  = os.path.join(ROOT_DIR, PROCESSED_DATA_PATH)

# ---------------------------------------------------------
# Import Utilities
# ---------------------------------------------------------

from utils import (
    banner,
    finished,
    load_csv,
    load_excel,
    merge_patient_data,
    dataset_info,
    validate_columns,
    save_csv
)

# =========================================================
# Required Columns
# =========================================================

REQUIRED_COLUMNS = [

    "patient_id",

    "timestamp",

    "fsr_1",
    "fsr_2",
    "fsr_3",
    "fsr_4",

    "temperature",

    "spo2",

    "heart_rate"

]

# =========================================================
# Load Dataset
# =========================================================

def load_dataset():

    banner()

    raw_df = load_csv(RAW_DATA_PATH)

    patient_df = load_excel(
        PATIENT_PROFILE_PATH
    )

    return raw_df, patient_df

# =========================================================
# Merge Patient Profile
# =========================================================

def merge_dataset(raw_df, patient_df):

    print("\nMerging patient profile...")

    df = merge_patient_data(
        raw_df,
        patient_df
    )

    print("Merge completed.")

    return df

# =========================================================
# Clean Dataset
# =========================================================

def clean_dataset(df):

    print("\nCleaning dataset...")

    # Remove duplicate rows

    before = len(df)

    df = df.drop_duplicates()

    after = len(df)

    print(
        f"Removed {before-after} duplicate rows."
    )

    # Timestamp

    df["timestamp"] = pd.to_datetime(
        df["timestamp"]
    )

    # Sort

    df = df.sort_values(
        by=[
            "patient_id",
            "timestamp"
        ]
    )

    df.reset_index(
        drop=True,
        inplace=True
    )

    # Session tracking to prevent data leakage across sessions (separated by >10 seconds)
    df["time_diff"] = df.groupby("patient_id")["timestamp"].diff().dt.total_seconds().fillna(0)
    df["new_session"] = (df["time_diff"] > 10).astype(int)
    df["session_id"] = df.groupby("patient_id")["new_session"].cumsum()

    # Numeric Columns
    numeric_cols = df.select_dtypes(include=np.number).columns

    for col in numeric_cols:
        # Strict causal forward fill ONLY within the same session
        df[col] = df.groupby(["patient_id", "session_id"])[col].ffill()
        
        # Initial safe baselines for the first sample of a session (no bfill)
        if col == "temperature":
            df[col] = df[col].fillna(31.0)
        elif col == "heart_rate":
            df[col] = df[col].fillna(60.0)
        elif col == "spo2":
            df[col] = df[col].fillna(98.0)
        else:
            df[col] = df[col].fillna(0.0)

    # Object Columns

    object_cols = df.select_dtypes(
        include="object"
    ).columns

    for col in object_cols:

        df[col] = df[col].fillna(
            "Unknown"
        )

    print("Cleaning completed.")

    return df

# =========================================================
# Validate Dataset
# =========================================================

def validate_dataset(df):

    print("\nValidating dataset...")

    validate_columns(
        df,
        REQUIRED_COLUMNS
    )

    dataset_info(
        df,
        "Merged Dataset"
    )

    return df

    # =========================================================
# Pressure Features
# =========================================================

def calculate_pressure_features(df):

    print("\nCalculating pressure features...")

    fsr_cols = [
        "fsr_1",
        "fsr_2",
        "fsr_3",
        "fsr_4"
    ]

    df["avg_pressure"] = df[fsr_cols].mean(axis=1)
    df["max_pressure"] = df[fsr_cols].max(axis=1)
    df["min_pressure"] = df[fsr_cols].min(axis=1)
    
    df["pressure_var"] = df[fsr_cols].var(axis=1).fillna(0)
    df["pressure_std"] = df[fsr_cols].std(axis=1).fillna(0)
    
    df["pressure_change_rate"] = (
        df.groupby(["patient_id", "session_id"])["avg_pressure"]
        .diff()
        .fillna(0)
    )
    
    # Pressure Gradient is the second discrete difference of average pressure
    df["pressure_gradient"] = (
        df.groupby(["patient_id", "session_id"])["pressure_change_rate"]
        .diff()
        .fillna(0)
    )
    
    df["pressure_stability"] = df["pressure_std"] / (df["avg_pressure"] + 1e-6)
    
    # COP Approximation (Front vs Back)
    # This is an approximate pressure-distribution / center-of-pressure proxy, not a clinically calibrated physical CoP measurement.
    df["cop_approx"] = ((df["fsr_1"] + df["fsr_2"]) - (df["fsr_3"] + df["fsr_4"])) / (df[fsr_cols].sum(axis=1) + 1e-6)
    
    # Symmetry (Medial vs Lateral, e.g. fsr_2 vs fsr_3)
    df["pressure_symmetry"] = np.abs(df["fsr_2"] - df["fsr_3"]) / (df["fsr_2"] + df["fsr_3"] + 1e-6)
    
    df["loading_rate"] = np.where(df["pressure_change_rate"] > 0, df["pressure_change_rate"], 0)
    df["unloading_rate"] = np.where(df["pressure_change_rate"] < 0, np.abs(df["pressure_change_rate"]), 0)
    
    # Pressure Duration (30-second elevated-pressure sample count)
    threshold = df.groupby(["patient_id", "session_id"])["avg_pressure"].transform(lambda x: x.expanding(min_periods=1).median())
    is_high = (df["avg_pressure"] > threshold).astype(int)
    df["pressure_duration"] = is_high.groupby([df["patient_id"], df["session_id"]]).transform(
        lambda x: x.rolling(ROLLING_WINDOW, min_periods=1).sum()
    )

    return df



# =========================================================
# Estimated Pressure Time Integral
# =========================================================

def calculate_pressure_integral(df):

    print("Calculating Pressure Integral (EPTI)...")

    # EPTI proxy: 30-second pressure-time integral proxy. 
    # PTI_proxy = sum(avg_pressure[t-29:t]). At 1 Hz this is equivalent to multiplying by dt=1 second.
    df["pressure_integral"] = (
        df.groupby(["patient_id", "session_id"])["avg_pressure"]
        .transform(lambda x: x.rolling(ROLLING_WINDOW, min_periods=1).sum())
    )

    return df


# =========================================================
# Temperature Trend
# =========================================================

def temperature_features(df):

    print("Temperature trend...")

    df["temp_diff"] = (
        df.groupby(["patient_id", "session_id"])["temperature"]
        .diff()
        .fillna(0)
    )

    return df


# =========================================================
# Heart Rate Trend
# =========================================================

def heart_rate_features(df):

    print("Heart-rate trend...")

    df["hr_diff"] = (
        df.groupby(["patient_id", "session_id"])["heart_rate"]
        .diff()
        .fillna(0)
    )

    return df


# =========================================================
# SpO2 Trend
# =========================================================

def spo2_features(df):

    print("SpO2 trend...")

    df["spo2_diff"] = (
        df.groupby(["patient_id", "session_id"])["spo2"]
        .diff()
        .fillna(0)
    )

    return df


# =========================================================
# Interaction Features
# =========================================================

def interaction_features(df):

    print("Calculating interaction features...")
    
    df["pressure_temp_interaction"] = df["avg_pressure"] * df["temperature"]
    df["pressure_hr_interaction"] = df["avg_pressure"] * df["heart_rate"]
    df["temp_hr_interaction"] = df["temperature"] * df["heart_rate"]
    
    return df


# =========================================================
# Rolling Window Features
# =========================================================

def rolling_features(df):

    print("Calculating rolling features...")

    rolling_cols = [
        "avg_pressure",
        "temperature",
        "heart_rate",
        "spo2"
    ]

    for col in rolling_cols:

        df[f"{col}_rolling_mean"] = (
            df.groupby(["patient_id", "session_id"])[col]
            .transform(
                lambda x:
                x.rolling(
                    ROLLING_WINDOW,
                    min_periods=1
                ).mean()
            )
        )

    return df


# =========================================================
# Recovery Factor
# =========================================================

def recovery_factor(df):

    print("Calculating recovery factor...")

    # Strictly causal expanding median within the session
    baseline = (
        df.groupby(["patient_id", "session_id"])["avg_pressure"]
        .transform(lambda x: x.expanding(min_periods=1).median())
    )

    df["recovery_factor"] = np.clip(
        (baseline - df["avg_pressure"]) / (baseline + 1e-6),
        0,
        1
    )

    return df


# =========================================================
# Final Feature Engineering
# =========================================================

def feature_engineering(df):

    df = calculate_pressure_features(df)
    df = calculate_pressure_integral(df)
    df = temperature_features(df)
    df = heart_rate_features(df)
    df = spo2_features(df)
    df = interaction_features(df)
    df = rolling_features(df)
    df = recovery_factor(df)

    # Note: df has many features now. 
    # We will pick the final set in sequence_generator.py

    return df

    # =========================================================
# Save Processed Dataset
# =========================================================

def save_processed_dataset(df):

    print("\nSaving processed dataset...")

    save_csv(
        df,
        PROCESSED_DATA_PATH
    )

    print("\nProcessed dataset saved successfully!")

    print(f"\nLocation:\n{PROCESSED_DATA_PATH}")


# =========================================================
# Main Pipeline
# =========================================================

def main():

    # Step 1: Load datasets
    raw_df, patient_df = load_dataset()

    # Step 2: Merge patient profile
    df = merge_dataset(
        raw_df,
        patient_df
    )

    # Step 3: Clean data
    df = clean_dataset(df)

    # Step 4: Validate
    df = validate_dataset(df)

    # Step 5: Feature Engineering
    df = feature_engineering(df)

    # Step 6: Final dataset info
    dataset_info(
        df,
        "Processed Dataset"
    )

    # Step 7: Save
    save_processed_dataset(df)

    finished()


# =========================================================
# Run
# =========================================================

if __name__ == "__main__":
    main()

