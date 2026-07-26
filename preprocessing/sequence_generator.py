"""
==========================================================
Smart Insole DFU Risk Prediction
Sequence Generator

Reads:
    data/processed/dfu_processed_dataset.csv

Creates:
    data/sequences/X_train.npy
    data/sequences/X_test.npy
    data/sequences/y_train.npy
    data/sequences/y_test.npy
    models/scaler.pkl
    models/class_weights.pkl
==========================================================
"""

import os
import sys
import warnings
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

warnings.filterwarnings("ignore")

# ----------------------------------------------------------
# Project Root  (works regardless of CWD)
# ----------------------------------------------------------

ROOT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)
sys.path.append(ROOT_DIR)

# ----------------------------------------------------------
# Config
# ----------------------------------------------------------

from config import (
    PROCESSED_DATA_PATH,
    WINDOW_SIZE,
    TRAIN_TEST_SPLIT,
    RANDOM_STATE,
    SEQUENCE_DATA_PATH,
    SCALER_PATH,
    CLASS_WEIGHTS_PATH,
    # Risk score weights
    PRESSURE_WEIGHT,
    ROLLING_PRESSURE_WEIGHT,
    TEMPERATURE_WEIGHT,
    TEMPERATURE_TREND_WEIGHT,
    RECOVERY_WEIGHT,
    HEART_RATE_WEIGHT,
    SPO2_WEIGHT,
    # Label thresholds
    LOW_RISK_THRESHOLD,
    MEDIUM_RISK_THRESHOLD,
    IMBALANCE_THRESHOLD,
)

# Resolve all paths to absolute so CWD never matters
PROCESSED_DATA_PATH = os.path.join(ROOT_DIR, PROCESSED_DATA_PATH)
SEQUENCE_DATA_PATH  = os.path.join(ROOT_DIR, SEQUENCE_DATA_PATH)
SCALER_PATH         = os.path.join(ROOT_DIR, SCALER_PATH)
CLASS_WEIGHTS_PATH  = os.path.join(ROOT_DIR, CLASS_WEIGHTS_PATH)

# ----------------------------------------------------------
# Utils
# ----------------------------------------------------------

from utils import (
    banner,
    finished,
    load_csv,
    dataset_info,
)

# ==========================================================
# LSTM Feature Columns
# ==========================================================

FEATURE_COLUMNS: List[str] = [
    "avg_pressure",
    "max_pressure",
    "pressure_std",
    "heel_ratio",
    "mid_ratio",
    "forefoot_ratio",
    "toe_ratio",
    "temperature",
    "spo2",
    "heart_rate",
    "temp_diff",
    "hr_diff",
    "spo2_diff",
    "avg_pressure_rolling_mean",
    "temperature_rolling_mean",
    "heart_rate_rolling_mean",
    "spo2_rolling_mean",
    "recovery_factor",
]

# Label names for display
LABEL_NAMES: Dict[int, str] = {0: "Low", 1: "Medium", 2: "High"}


# ==========================================================
# 1. Load Dataset
# ==========================================================

def load_dataset() -> pd.DataFrame:
    """Load the processed CSV dataset and print basic info."""
    banner()
    df = load_csv(PROCESSED_DATA_PATH)
    dataset_info(df, "Processed Dataset")
    return df


# ==========================================================
# 2. Clinical Risk Score — Individual Feature Scorers
# ==========================================================

def pressure_score(value: float) -> float:
    """
    Convert instantaneous avg_pressure to a 0-100 risk score.

    Clinically: sustained plantar pressure > 700 kPa is associated
    with tissue ischaemia leading to DFU (Bus et al., 2016).
    The FSR ADC values are mapped to a 5-step ordinal scale.
    """
    if value < 300:
        return 0
    elif value < 500:
        return 25
    elif value < 700:
        return 50
    elif value < 900:
        return 75
    return 100


def rolling_pressure_score(value: float) -> float:
    """
    Score based on 30-second rolling mean of avg_pressure.

    Clinically: cumulative sustained load (pressure-time integral)
    is a stronger predictor of DFU than peak pressure alone.
    """
    if value < 250:
        return 0
    elif value < 450:
        return 25
    elif value < 650:
        return 50
    elif value < 850:
        return 75
    return 100


def temperature_score(value: float) -> float:
    """
    Convert skin temperature (°C) to a 0-100 risk score.

    Clinically: a skin temperature > 35.5°C on the plantar surface
    indicates early inflammatory activity; > 36.5°C is a validated
    DFU warning threshold (IWGDF 2019 guideline).
    """
    if value < 34.5:
        return 0
    elif value < 35.5:
        return 20
    elif value < 36.5:
        return 45
    elif value < 37.2:
        return 70
    elif value < 38.0:
        return 85
    return 100


def temperature_trend_score(value: float) -> float:
    """
    Score based on temp_diff (per-second temperature change).

    Clinically: a rapid temperature rise (>0.5°C over seconds)
    reflects acute inflammation even when absolute value is low.
    Negative trend is protective.
    """
    if value <= 0:
        return 0
    elif value < 0.1:
        return 15
    elif value < 0.3:
        return 40
    elif value < 0.5:
        return 65
    elif value < 0.8:
        return 85
    return 100


def recovery_score(value: float) -> float:
    """
    Score based on recovery_factor (0-1, clipped).

    Clinically: recovery_factor ≈ (baseline_pressure - current) / baseline.
    A value near 1 means the foot is fully off-loaded (safe).
    A value near 0 means no off-loading — continuous tissue stress.
    """
    # High recovery -> low risk; invert the score
    inverted = 1.0 - float(np.clip(value, 0, 1))
    return round(inverted * 100, 2)


def heart_rate_score(value: float) -> float:
    """
    Convert heart rate (bpm) to a 0-100 risk score.

    Clinically: tachycardia can indicate pain, infection, or
    autonomic neuropathy — all DFU risk amplifiers.
    """
    if value <= 80:
        return 0
    elif value <= 90:
        return 20
    elif value <= 100:
        return 45
    elif value <= 110:
        return 70
    elif value <= 120:
        return 85
    return 100


def spo2_score(value: float) -> float:
    """
    Convert SpO2 (%) to a 0-100 risk score.

    Clinically: peripheral oxygen saturation < 94% impairs tissue
    oxygenation and wound healing — a key DFU progression factor.
    """
    if value >= 98:
        return 0
    elif value >= 96:
        return 20
    elif value >= 94:
        return 50
    elif value >= 92:
        return 75
    return 100


# ==========================================================
# 3. Composite Clinical Risk Score (Temporal + Instantaneous)
# ==========================================================

def calculate_risk_score(row: pd.Series) -> float:
    """
    Compute a weighted composite DFU risk score (0–100) for one row.

    Incorporates both instantaneous sensor readings AND temporal
    engineered features so the score reflects dynamic foot behaviour,
    not just a single-point snapshot.

    Weights are fully driven by config.py — no magic numbers here.

    Returns
    -------
    float
        Risk score in [0, 100].
    """
    p_inst   = pressure_score(row["avg_pressure"])
    p_roll   = rolling_pressure_score(row["avg_pressure_rolling_mean"])
    t_abs    = temperature_score(row["temperature"])
    t_trend  = temperature_trend_score(row["temp_diff"])
    recovery = recovery_score(row["recovery_factor"])
    hr       = heart_rate_score(row["heart_rate"])
    spo2     = spo2_score(row["spo2"])

    score = (
        p_inst   * PRESSURE_WEIGHT         +
        p_roll   * ROLLING_PRESSURE_WEIGHT  +
        t_abs    * TEMPERATURE_WEIGHT       +
        t_trend  * TEMPERATURE_TREND_WEIGHT +
        recovery * RECOVERY_WEIGHT          +
        hr       * HEART_RATE_WEIGHT        +
        spo2     * SPO2_WEIGHT
    )

    return round(score, 4)


# ==========================================================
# 4. Label Distribution Analysis  (Enhancement #1)
# ==========================================================

def _print_distribution(labels: pd.Series) -> None:
    """Print risk label counts and percentages in a readable format."""
    total = len(labels)
    counts = labels.value_counts().sort_index()

    print("\nRisk Label Distribution")
    print("-" * 35)
    for idx, name in LABEL_NAMES.items():
        cnt = counts.get(idx, 0)
        print(f"  {name:<10} ({idx})  :  {cnt:>6,}")

    print("\nPercentage")
    print("-" * 35)
    dominant = False
    for idx, name in LABEL_NAMES.items():
        cnt = counts.get(idx, 0)
        pct = cnt / total * 100
        print(f"  {name:<10}       :  {pct:>6.2f}%")
        if pct / 100 > IMBALANCE_THRESHOLD:
            dominant = True

    if dominant:
        print(
            "\n  [!] Warning: Dataset appears to be imbalanced.\n"
            "      Class weights will be computed automatically."
        )


# ==========================================================
# 5. Label Validation  (Enhancement #5)
# ==========================================================

def validate_labels(df: pd.DataFrame) -> None:
    """
    Validate that risk_label column is complete and contains only {0, 1, 2}.

    Raises
    ------
    ValueError
        If NaN values or out-of-range labels are found.
    """
    labels = df["risk_label"]
    total  = len(labels)
    counts = labels.value_counts().sort_index()

    print("\n" + "=" * 50)
    print("Label Validation")
    print("=" * 50)
    print(f"  Total Samples   : {total:,}")
    for idx, name in LABEL_NAMES.items():
        cnt = counts.get(idx, 0)
        pct = cnt / total * 100
        print(f"  {name:<10} ({idx}) : {cnt:>6,}  ({pct:.2f}%)")

    # NaN check
    nan_count = labels.isna().sum()
    if nan_count > 0:
        raise ValueError(
            f"Label validation failed: {nan_count} NaN values found in 'risk_label'."
        )

    # Valid label check
    invalid = labels[~labels.isin([0, 1, 2])]
    if not invalid.empty:
        raise ValueError(
            f"Label validation failed: {len(invalid)} invalid label(s) found.\n"
            f"Unexpected values: {sorted(invalid.unique())}\n"
            f"Expected only: {{0, 1, 2}}"
        )

    print("\n  [OK] All labels are valid (no NaN, no out-of-range values).")
    print("=" * 50)


# ==========================================================
# 6. Class Weight Computation  (Enhancement #2)
# ==========================================================

def compute_class_weights(y: np.ndarray) -> Dict[int, float]:
    """
    Compute balanced class weights for use in model.fit(class_weight=...).

    Uses sklearn's 'balanced' strategy:
        weight[c] = n_samples / (n_classes * count[c])

    This counteracts label imbalance without over-sampling.

    Parameters
    ----------
    y : np.ndarray
        Integer label array.

    Returns
    -------
    dict
        {0: float, 1: float, 2: float}
    """
    classes = np.array([0, 1, 2])
    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y,
    )
    class_weight_dict = {int(c): round(float(w), 4) for c, w in zip(classes, weights)}

    print("\nClass Weights (for model.fit):")
    print("-" * 35)
    for cls, w in class_weight_dict.items():
        print(f"  Class {cls} ({LABEL_NAMES[cls]:<6}) : {w:.4f}")
    print()

    return class_weight_dict


# ==========================================================
# 7. Generate Labels  (Enhanced: dist + validation)
# ==========================================================

def generate_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the clinical risk score for every row, then bin into
    three risk labels: 0=Low, 1=Medium, 2=High.

    Also prints label distribution and raises on invalid labels.

    Parameters
    ----------
    df : pd.DataFrame
        Processed dataset with all engineered features.

    Returns
    -------
    pd.DataFrame
        Input dataframe with two new columns:
            - 'risk_score'  (float, 0-100)
            - 'risk_label'  (int, 0|1|2)
    """
    print("\nGenerating Clinical Risk Score...")

    df = df.copy()

    df["risk_score"] = df.apply(calculate_risk_score, axis=1)

    df["risk_label"] = pd.cut(
        df["risk_score"],
        bins=[-1, LOW_RISK_THRESHOLD, MEDIUM_RISK_THRESHOLD, 100],
        labels=[0, 1, 2],
    ).astype(int)

    print("Labels Generated.\n")

    # Distribution analysis
    _print_distribution(df["risk_label"])

    # Strict validation
    validate_labels(df)

    return df


# ==========================================================
# 8. Generate Sequences  (unchanged logic, added type hints + docstring)
# ==========================================================

def generate_sequences(
    df: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create sliding-window LSTM sequences per patient.

    Each sequence spans WINDOW_SIZE timesteps; the label is the
    risk class at the NEXT timestep (predictive labelling).

    Parameters
    ----------
    df : pd.DataFrame
        Labelled processed dataset.

    Returns
    -------
    X : np.ndarray, shape (n_sequences, WINDOW_SIZE, n_features)
    y : np.ndarray, shape (n_sequences,)
    patient_ids : np.ndarray, shape (n_sequences,)
    """
    print("\nGenerating LSTM Sequences...")

    X: List[np.ndarray] = []
    y: List[int] = []
    patient_ids: List[str] = []

    patients = df["patient_id"].unique()

    for patient in patients:
        patient_df = (
            df[df["patient_id"] == patient]
            .reset_index(drop=True)
        )

        features = patient_df[FEATURE_COLUMNS].values
        labels   = patient_df["risk_label"].values

        for i in range(len(patient_df) - WINDOW_SIZE):
            sequence = features[i : i + WINDOW_SIZE]
            target   = labels[i + WINDOW_SIZE]      # predict NEXT timestamp

            X.append(sequence)
            y.append(target)
            patient_ids.append(patient)

    X_arr  = np.array(X,           dtype=np.float32)
    y_arr  = np.array(y,           dtype=np.int32)
    pid_arr= np.array(patient_ids)

    print(f"\n  Sequences Created : {len(X_arr):,}")
    print(f"  Sequence Shape    : {X_arr.shape}")

    return X_arr, y_arr, pid_arr


# ==========================================================
# 9. Patient-wise Train/Test Split
# ==========================================================

def patient_split(
    X: np.ndarray,
    y: np.ndarray,
    patient_ids: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split sequences by unique patient identity to prevent data leakage.

    Parameters
    ----------
    X, y, patient_ids : arrays from generate_sequences()

    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    print("\nPerforming Patient-wise Split...")

    unique_patients = np.unique(patient_ids)

    train_patients, test_patients = train_test_split(
        unique_patients,
        test_size=TRAIN_TEST_SPLIT,
        random_state=RANDOM_STATE,
        shuffle=True,
    )

    train_mask = np.isin(patient_ids, train_patients)
    test_mask  = np.isin(patient_ids, test_patients)

    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]

    print(f"  Train Patients   : {len(train_patients)}")
    print(f"  Test Patients    : {len(test_patients)}")
    print(f"  Training Samples : {len(X_train):,}")
    print(f"  Testing Samples  : {len(X_test):,}")

    return X_train, X_test, y_train, y_test


# ==========================================================
# 10. Feature Scaling
# ==========================================================

def scale_features(
    X_train: np.ndarray,
    X_test: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    """
    Fit a StandardScaler on training data and transform both splits.

    The scaler is fit on the flattened 2-D view of X_train to
    compute per-feature statistics, then applied identically to
    X_test to prevent leakage.

    Returns
    -------
    X_train_scaled, X_test_scaled, scaler
    """
    print("\nScaling Features...")

    n_features = X_train.shape[2]
    scaler = StandardScaler()

    X_train_2d = X_train.reshape(-1, n_features)
    X_test_2d  = X_test.reshape(-1,  n_features)

    scaler.fit(X_train_2d)

    X_train_scaled = scaler.transform(X_train_2d).reshape(X_train.shape)
    X_test_scaled  = scaler.transform(X_test_2d ).reshape(X_test.shape)

    print("  Features scaled successfully.")

    return X_train_scaled, X_test_scaled, scaler


# ==========================================================
# 11. Save All Artifacts
# ==========================================================

def save_datasets(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    scaler: StandardScaler,
    class_weights: Dict[int, float],
) -> None:
    """
    Persist all sequence arrays, the scaler, and class weights to disk.

    Outputs
    -------
    data/sequences/X_train.npy
    data/sequences/X_test.npy
    data/sequences/y_train.npy
    data/sequences/y_test.npy
    models/scaler.pkl
    models/class_weights.pkl
    """
    print("\nSaving datasets...")

    os.makedirs(SEQUENCE_DATA_PATH, exist_ok=True)
    os.makedirs(os.path.dirname(SCALER_PATH), exist_ok=True)

    np.save(os.path.join(SEQUENCE_DATA_PATH, "X_train.npy"), X_train)
    np.save(os.path.join(SEQUENCE_DATA_PATH, "X_test.npy"),  X_test)
    np.save(os.path.join(SEQUENCE_DATA_PATH, "y_train.npy"), y_train)
    np.save(os.path.join(SEQUENCE_DATA_PATH, "y_test.npy"),  y_test)

    joblib.dump(scaler,        SCALER_PATH)
    joblib.dump(class_weights, CLASS_WEIGHTS_PATH)

    print("\n  Datasets Saved Successfully!")
    print(f"  X_train : {X_train.shape}")
    print(f"  X_test  : {X_test.shape}")
    print(f"  y_train : {y_train.shape}")
    print(f"  y_test  : {y_test.shape}")
    print(f"\n  Scaler        -> {SCALER_PATH}")
    print(f"  Class weights -> {CLASS_WEIGHTS_PATH}")


# ==========================================================
# Main Pipeline
# ==========================================================

def main() -> None:
    """End-to-end sequence generation pipeline."""

    # Step 1: Load processed dataset
    df = load_dataset()

    # Step 2: Generate risk score + labels (includes dist + validation)
    df = generate_labels(df)

    # Step 3: Generate sliding-window sequences
    X, y, patient_ids = generate_sequences(df)

    # Step 4: Patient-wise train/test split (no leakage)
    X_train, X_test, y_train, y_test = patient_split(X, y, patient_ids)

    # Step 5: Compute class weights from training labels only
    class_weights = compute_class_weights(y_train)

    # Step 6: Scale features (fit on train, transform both)
    X_train, X_test, scaler = scale_features(X_train, X_test)

    # Step 7: Save all artifacts
    save_datasets(X_train, X_test, y_train, y_test, scaler, class_weights)

    finished()


# ==========================================================
# Entry Point
# ==========================================================

if __name__ == "__main__":
    main()
