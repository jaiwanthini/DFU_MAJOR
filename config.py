# ==========================================
# Smart Insole DFU Risk Prediction
# Global Configuration
# ==========================================

# -------------------------------
# File Paths
# -------------------------------

RAW_DATA_PATH        = "data/raw/dfu_raw_dataset_100000.csv"
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

SEQUENCE_WINDOW = 3 # seconds per sequence (for LSTM temporal dimension)
ROLLING_WINDOW  = 30 # seconds (for clinical rolling features like sustained pressure)
SAMPLING_RATE = 1    # readings per second

# -------------------------------
# Train / Test Split
# -------------------------------

TRAIN_SPLIT = 0.70
VAL_SPLIT   = 0.15
TEST_SPLIT  = 0.15
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
EPOCHS        = 100
LEARNING_RATE = 0.0005  # Task 3: Reduced from 0.001 to 0.0005
RECURRENT_DROPOUT = 0.1

# -------------------------------
# Feature Information
# -------------------------------
NUM_FEATURES = 28

MODEL_FEATURE_COLUMNS = [
    "avg_pressure",
    "max_pressure",
    "min_pressure",
    "pressure_var",
    "pressure_std",
    "pressure_gradient",
    "pressure_change_rate",
    "pressure_stability",
    "cop_approx",
    "pressure_symmetry",
    "temp_diff",
    "temperature_rolling_mean",
    "hr_diff",
    "heart_rate_rolling_mean",
    "spo2_diff",
    "spo2_rolling_mean",
    "pressure_temp_interaction",
    "pressure_hr_interaction",
    "temp_hr_interaction",
    "pressure_integral",
    "recovery_factor",
    "loading_rate",
    "unloading_rate",
    "pressure_duration",
    "avg_pressure_rolling_mean",
    "temperature",
    "spo2",
    "heart_rate"
]

FEATURE_DISPLAY_NAMES = [
    "Average Pressure",
    "Max Pressure",
    "Min Pressure",
    "Pressure Variance",
    "Pressure Standard Deviation",
    "Pressure Gradient",
    "Pressure Change Rate",
    "Pressure Stability",
    "Center of Pressure",
    "Pressure Symmetry",
    "Temperature Change Rate",
    "Rolling Temperature Mean",
    "Heart Rate Change Rate",
    "Rolling Heart Rate Mean",
    "SpO2 Change Rate",
    "Rolling SpO2 Mean",
    "Pressure x Temperature",
    "Pressure x Heart Rate",
    "Temperature x Heart Rate",
    "Pressure Integral",
    "Recovery Factor",
    "Loading Rate",
    "Unloading Rate",
    "Pressure Duration",
    "Sustained Pressure (Rolling)",
    "Skin Temperature",
    "SpO2 Level",
    "Heart Rate",
]

CLASS_NAMES = ["Low", "Medium", "High"]
