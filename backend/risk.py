"""
==========================================================
Smart Insole DFU Risk Prediction
Shared Risk Scoring Module

Contains rule-based clinical risk scoring logic.
Extracted from preprocessing to decouple backend from
training code.
==========================================================
"""

import numpy as np
import pandas as pd

from config import (
    PRESSURE_WEIGHT,
    ROLLING_PRESSURE_WEIGHT,
    TEMPERATURE_WEIGHT,
    TEMPERATURE_TREND_WEIGHT,
    RECOVERY_WEIGHT,
    HEART_RATE_WEIGHT,
    SPO2_WEIGHT
)

def pressure_score(value: float) -> float:
    """
    Evaluate instantaneous plantar pressure.
    Returns a risk score between 0 and 100 based on clinical thresholds.
    """
    if value < 300:
        return 0.0
    elif value < 500:
        return 25.0
    elif value < 700:
        return 50.0
    elif value < 900:
        return 75.0
    return 100.0

def rolling_pressure_score(value: float) -> float:
    """
    Evaluate cumulative sustained pressure (rolling mean).
    Returns a risk score between 0 and 100 based on load thresholds.
    """
    if value < 250:
        return 0.0
    elif value < 450:
        return 25.0
    elif value < 650:
        return 50.0
    elif value < 850:
        return 75.0
    return 100.0

def temperature_score(value: float) -> float:
    """
    Evaluate absolute skin temperature.
    Returns a risk score between 0 and 100 indicating inflammatory activity.
    """
    if value < 34.5:
        return 0.0
    elif value < 35.5:
        return 20.0
    elif value < 36.5:
        return 45.0
    elif value < 37.2:
        return 70.0
    elif value < 38.0:
        return 85.0
    return 100.0

def temperature_trend_score(value: float) -> float:
    """
    Evaluate rapid per-second temperature changes.
    Returns a risk score between 0 and 100 based on the heating trend.
    """
    if value <= 0:
        return 0.0
    elif value < 0.1:
        return 15.0
    elif value < 0.3:
        return 40.0
    elif value < 0.5:
        return 65.0
    elif value < 0.8:
        return 85.0
    return 100.0

def recovery_score(value: float) -> float:
    """
    Evaluate tissue recovery based on the off-loading factor.
    Returns a risk score between 0 and 100 (lower factor means higher risk).
    """
    inverted = 1.0 - float(np.clip(value, 0.0, 1.0))
    return round(inverted * 100.0, 2)

def heart_rate_score(value: float) -> float:
    """
    Evaluate systemic heart rate.
    Returns a risk score between 0 and 100 indicating autonomic stress.
    """
    if value <= 80:
        return 0.0
    elif value <= 90:
        return 20.0
    elif value <= 100:
        return 45.0
    elif value <= 110:
        return 70.0
    elif value <= 120:
        return 85.0
    return 100.0

def spo2_score(value: float) -> float:
    """
    Evaluate peripheral oxygen saturation (SpO2).
    Returns a risk score between 0 and 100 indicating hypoxia risk.
    """
    if value >= 98:
        return 0.0
    elif value >= 96:
        return 20.0
    elif value >= 94:
        return 50.0
    elif value >= 92:
        return 75.0
    return 100.0

def calculate_risk_score(row: pd.Series) -> float:
    """
    Compute a weighted composite DFU risk score (0-100) for one sensor reading.
    Validates that all required features are present before scoring.
    """
    required_features = [
        "avg_pressure",
        "avg_pressure_rolling_mean",
        "temperature",
        "temp_diff",
        "recovery_factor",
        "heart_rate",
        "spo2"
    ]
    
    missing_features = [feature for feature in required_features if feature not in row]
    if missing_features:
        raise ValueError(f"Cannot calculate risk score. Missing required features: {missing_features}")

    p_inst   = pressure_score(float(row["avg_pressure"]))
    p_roll   = rolling_pressure_score(float(row["avg_pressure_rolling_mean"]))
    t_abs    = temperature_score(float(row["temperature"]))
    t_trend  = temperature_trend_score(float(row["temp_diff"]))
    recovery = recovery_score(float(row["recovery_factor"]))
    hr       = heart_rate_score(float(row["heart_rate"]))
    spo2     = spo2_score(float(row["spo2"]))

    score = (
        p_inst   * PRESSURE_WEIGHT          +
        p_roll   * ROLLING_PRESSURE_WEIGHT  +
        t_abs    * TEMPERATURE_WEIGHT       +
        t_trend  * TEMPERATURE_TREND_WEIGHT +
        recovery * RECOVERY_WEIGHT          +
        hr       * HEART_RATE_WEIGHT        +
        spo2     * SPO2_WEIGHT
    )

    return round(score, 4)
