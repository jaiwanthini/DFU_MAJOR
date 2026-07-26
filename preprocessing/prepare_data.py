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
    WINDOW_SIZE
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

    # Numeric Columns

    numeric_cols = df.select_dtypes(
        include=np.number
    ).columns

    for col in numeric_cols:

        df[col] = df[col].fillna(
            df[col].median()
        )

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

    df["pressure_std"] = df[fsr_cols].std(axis=1)

    df["pressure_range"] = (
        df["max_pressure"] -
        df["min_pressure"]
    )

    return df


# =========================================================
# Pressure Distribution
# =========================================================

def calculate_pressure_distribution(df):

    print("Calculating pressure distribution...")

    total = (
        df["fsr_1"] +
        df["fsr_2"] +
        df["fsr_3"] +
        df["fsr_4"]
    )

    total = total.replace(0, 1)

    df["heel_ratio"] = df["fsr_1"] / total

    df["mid_ratio"] = df["fsr_2"] / total

    df["forefoot_ratio"] = df["fsr_3"] / total

    df["toe_ratio"] = df["fsr_4"] / total

    return df


# =========================================================
# Estimated Pressure Time Integral
# =========================================================

def calculate_epti(df):

    print("Calculating EPTI...")

    df["epti"] = (
        df["avg_pressure"] *
        1
    )

    return df


# =========================================================
# Temperature Trend
# =========================================================

def temperature_features(df):

    print("Temperature trend...")

    df["temp_diff"] = (
        df.groupby("patient_id")["temperature"]
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
        df.groupby("patient_id")["heart_rate"]
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
        df.groupby("patient_id")["spo2"]
        .diff()
        .fillna(0)
    )

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

            df.groupby("patient_id")[col]

            .transform(

                lambda x:

                x.rolling(
                    WINDOW_SIZE,
                    min_periods=1
                ).mean()

            )

        )

        df[f"{col}_rolling_std"] = (

            df.groupby("patient_id")[col]

            .transform(

                lambda x:

                x.rolling(
                    WINDOW_SIZE,
                    min_periods=1
                ).std()

            )

        ).fillna(0)

    return df


# =========================================================
# Recovery Factor
# =========================================================

def recovery_factor(df):

    print("Calculating recovery factor...")

    baseline = (

        df.groupby("patient_id")["avg_pressure"]

        .transform("median")

    )

    df["recovery_factor"] = np.clip(
    (baseline - df["avg_pressure"]) / baseline,
    0,
    1
)

    return df


# =========================================================
# Final Feature Engineering
# =========================================================

def feature_engineering(df):

    df = calculate_pressure_features(df)

    df = calculate_pressure_distribution(df)

    df = calculate_epti(df)

    df = temperature_features(df)

    df = heart_rate_features(df)

    df = spo2_features(df)

    df = rolling_features(df)

    df = recovery_factor(df)

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

