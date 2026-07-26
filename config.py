# ==========================================
# Smart Insole DFU Risk Prediction
# Global Configuration
# ==========================================

# -------------------------------
# File Paths
# -------------------------------

RAW_DATA_PATH        = "data/raw/dfu_raw_dataset_50000.csv"
PATIENT_PROFILE_PATH = "data/raw/patient_profile.xlsx"
PROCESSED_DATA_PATH  = "data/processed/dfu_processed_dataset.csv"
SEQUENCE_DATA_PATH   = "data/sequences/"

MODEL_PATH           = "models/dfu_lstm.h5"
SCALER_PATH          = "models/scaler.pkl"
LABEL_ENCODER_PATH   = "models/label_encoder.pkl"
CLASS_WEIGHTS_PATH   = "models/class_weights.pkl"

# -------------------------------
# Window Parameters
# -------------------------------

WINDOW_SIZE   = 30   # seconds per sequence
SAMPLING_RATE = 1    # readings per second

# -------------------------------
# Train / Test Split
# -------------------------------

TRAIN_TEST_SPLIT = 0.20
RANDOM_STATE     = 42

# -------------------------------
# Clinical Risk Score Weights
#
# Justification (DFU pathophysiology):
#   avg_pressure          -> Sustained plantar pressure is the
#                            primary mechanical driver of DFU.
#   rolling_pressure_mean -> Cumulative load over 30 s reveals
#                            prolonged off-loading failure.
#   temperature           -> Skin temperature rise (>2°C) is an
#                            early biomarker of pre-ulcerative
#                            inflammation (IWGDF guideline).
#   temperature_trend     -> A rising trend signals acute
#                            inflammatory onset even when absolute
#                            value is still borderline.
#   recovery_factor       -> Low recovery indicates the foot never
#                            fully off-loads, predicting cumulative
#                            tissue damage.
#   heart_rate            -> Tachycardia correlates with pain,
#                            infection, or autonomic neuropathy.
#   spo2                  -> Peripheral hypoxia (<94 %) accelerates
#                            tissue ischaemia and impairs healing.
# -------------------------------

PRESSURE_WEIGHT         = 0.35   # instantaneous plantar pressure
ROLLING_PRESSURE_WEIGHT = 0.20   # 30-second sustained load
TEMPERATURE_WEIGHT      = 0.15   # absolute skin temperature
TEMPERATURE_TREND_WEIGHT= 0.10   # rising temp trend (temp_diff)
RECOVERY_WEIGHT         = 0.10   # foot off-loading ability
HEART_RATE_WEIGHT       = 0.05   # autonomic / systemic stress
SPO2_WEIGHT             = 0.05   # peripheral oxygenation

# Guard: weights must sum to 1.0
_TOTAL_WEIGHT = (
    PRESSURE_WEIGHT +
    ROLLING_PRESSURE_WEIGHT +
    TEMPERATURE_WEIGHT +
    TEMPERATURE_TREND_WEIGHT +
    RECOVERY_WEIGHT +
    HEART_RATE_WEIGHT +
    SPO2_WEIGHT
)
assert abs(_TOTAL_WEIGHT - 1.0) < 1e-6, (
    f"Risk weights must sum to 1.0, got {_TOTAL_WEIGHT:.4f}"
)

# -------------------------------
# Risk Label Thresholds
# (applied to 0-100 risk score)
# -------------------------------

LOW_RISK_THRESHOLD    = 34   # score <= 34  -> Low  (0)
MEDIUM_RISK_THRESHOLD = 64   # score <= 64  -> Medium (1)
                              # score >  64  -> High  (2)

# Imbalance warning: if any class exceeds this share, warn user
IMBALANCE_THRESHOLD = 0.70   # 70 %

# -------------------------------
# LSTM Hyperparameters
# -------------------------------

BATCH_SIZE    = 64
EPOCHS        = 50
LEARNING_RATE = 0.001
# -------------------------------
# Feature Information
# -------------------------------
NUM_FEATURES = 18

FEATURE_DISPLAY_NAMES = [
    "Average Pressure",
    "Max Pressure",
    "Pressure Variability",
    "Heel Pressure Ratio",
    "Midfoot Pressure Ratio",
    "Forefoot Pressure Ratio",
    "Toe Pressure Ratio",
    "Skin Temperature",
    "SpO2 Level",
    "Heart Rate",
    "Temperature Trend",
    "Heart Rate Trend",
    "SpO2 Trend",
    "Sustained Pressure (Rolling)",
    "Sustained Temperature",
    "Sustained Heart Rate",
    "Sustained SpO2",
    "Tissue Recovery Factor",
]

CLASS_NAMES = ["Low", "Medium", "High"]
