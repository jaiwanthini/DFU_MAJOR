"""
==========================================================
Smart Insole DFU Risk Prediction
Utility Functions
==========================================================
"""

import os
import pandas as pd


# ==========================================================
# Create directory if not exists
# ==========================================================

def create_directory(path: str):
    """
    Create folder if it doesn't exist.
    """
    os.makedirs(path, exist_ok=True)


# ==========================================================
# Load CSV
# ==========================================================

def load_csv(path: str) -> pd.DataFrame:
    """
    Load CSV file.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV not found:\n{path}")

    print(f"\nLoading CSV:\n{path}")

    return pd.read_csv(path)


# ==========================================================
# Load Excel
# ==========================================================

def load_excel(path: str) -> pd.DataFrame:
    """
    Load Excel file.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Excel file not found:\n{path}")

    print(f"\nLoading Excel:\n{path}")

    return pd.read_excel(path)


# ==========================================================
# Save CSV
# ==========================================================

def save_csv(df: pd.DataFrame, path: str):
    """
    Save dataframe as CSV.
    """

    folder = os.path.dirname(path)

    create_directory(folder)

    df.to_csv(path, index=False)

    print(f"\nDataset saved successfully.")

    print(path)


# ==========================================================
# Dataset Information
# ==========================================================

def dataset_info(df: pd.DataFrame, title="Dataset"):

    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)

    print(f"Rows      : {len(df)}")
    print(f"Columns   : {len(df.columns)}")

    print("\nColumns")

    for col in df.columns:
        print(f" - {col}")

    print("\nMissing Values")

    print(df.isnull().sum())

    print("\nData Types")

    print(df.dtypes)

    print("=" * 60)


# ==========================================================
# Validate Required Columns
# ==========================================================

def validate_columns(df: pd.DataFrame, required_columns):

    missing = []

    for col in required_columns:

        if col not in df.columns:
            missing.append(col)

    if len(missing):

        raise Exception(
            f"\nMissing Columns\n{missing}"
        )

    print("\nAll required columns found.")


# ==========================================================
# Remove Duplicate Columns
# ==========================================================

def remove_duplicate_columns(raw_df, patient_df):

    duplicate = []

    for col in patient_df.columns:

        if col in raw_df.columns and col != "patient_id":
            duplicate.append(col)

    if duplicate:

        patient_df = patient_df.drop(columns=duplicate)

    return patient_df


# ==========================================================
# Merge Patient Data
# ==========================================================

def merge_patient_data(raw_df, patient_df):

    patient_df = remove_duplicate_columns(
        raw_df,
        patient_df
    )

    merged = raw_df.merge(
        patient_df,
        on="patient_id",
        how="left"
    )

    return merged


# ==========================================================
# Print Banner
# ==========================================================

def banner():

    print("\n")
    print("=" * 60)
    print(" Smart Insole DFU Risk Prediction ")
    print("=" * 60)


# ==========================================================
# Finish Banner
# ==========================================================

def finished():

    print("\n")
    print("=" * 60)
    print(" Preprocessing Completed Successfully ")
    print("=" * 60)