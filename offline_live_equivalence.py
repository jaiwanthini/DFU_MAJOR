import os
import sys
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

# ----------------------------------------------------------
# Setup paths
# ----------------------------------------------------------
ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
PREPROCESSING_DIR = os.path.join(ROOT_DIR, "preprocessing")
if PREPROCESSING_DIR not in sys.path:
    sys.path.insert(0, PREPROCESSING_DIR)

from config import MODEL_FEATURE_COLUMNS, SEQUENCE_WINDOW
from backend.preprocess_live import LivePreprocessor
from preprocessing.prepare_data import feature_engineering

def run_equivalence_test():
    print("=" * 60)
    print("Offline vs Live Equivalence Test")
    print("=" * 60)
    
    # 1. Create a dummy sequence of raw sensor data (simulate 1 session, 40 steps)
    print("\n1. Generating synthetic raw session (40 readings)...")
    np.random.seed(42)
    n_steps = 40
    
    raw_df = pd.DataFrame({
        "patient_id": ["TEST_001"] * n_steps,
        "session_id": [1] * n_steps,
        "timestamp": pd.date_range("2023-01-01 10:00:00", periods=n_steps, freq="1S"),
        "fsr_1": np.random.uniform(50, 200, n_steps),
        "fsr_2": np.random.uniform(100, 300, n_steps),
        "fsr_3": np.random.uniform(100, 300, n_steps),
        "fsr_4": np.random.uniform(20, 100, n_steps),
        "temperature": np.linspace(31.0, 32.5, n_steps) + np.random.normal(0, 0.1, n_steps),
        "spo2": np.linspace(98.0, 96.0, n_steps) + np.random.normal(0, 0.2, n_steps),
        "heart_rate": np.linspace(70.0, 85.0, n_steps) + np.random.normal(0, 1.0, n_steps),
    })

    # 2. Run Offline Feature Engineering
    print("2. Processing Offline (prepare_data.py)...")
    offline_df = feature_engineering(raw_df.copy())
    offline_features = offline_df[MODEL_FEATURE_COLUMNS].values

    # 3. Run Live Preprocessor
    print("3. Processing Live (preprocess_live.py)...")
    live_processor = LivePreprocessor()
    live_features_list = []
    
    for i in range(n_steps):
        row = raw_df.iloc[i]
        raw_dict = {
            "fsr1": row["fsr_1"],
            "fsr2": row["fsr_2"],
            "fsr3": row["fsr_3"],
            "fsr4": row["fsr_4"],
            "temperature": row["temperature"],
            "spo2": row["spo2"],
            "heart_rate": row["heart_rate"],
        }
        
        result = live_processor.process_reading(raw_dict)
        # The live buffer contains the raw features BEFORE scaling.
        # Let's extract the last row from the buffer to compare with offline features.
        live_features_list.append(live_processor.buffer.iloc[-1][MODEL_FEATURE_COLUMNS].values)

    live_features = np.array(live_features_list, dtype=np.float64)
    offline_features = np.array(offline_features, dtype=np.float64)

    # 4. Compare Timestep by Timestep
    print("\n4. Comparing Features (Max Absolute Difference)...")
    max_diffs = np.max(np.abs(offline_features - live_features), axis=0)
    
    all_match = True
    for idx, feature_name in enumerate(MODEL_FEATURE_COLUMNS):
        diff = max_diffs[idx]
        status = "PASS" if diff < 1e-5 else "FAIL"
        if status == "FAIL":
            all_match = False
        print(f"  {feature_name:<30} : {diff:.8f} [{status}]")

    print("\n" + "=" * 60)
    if all_match:
        print("SUCCESS: Offline and Live pipelines are MATHEMATICALLY IDENTICAL.")
    else:
        print("ERROR: Divergence detected between offline and live pipelines!")
    print("=" * 60)
    
    assert all_match, "Equivalence test failed."
    
if __name__ == "__main__":
    run_equivalence_test()
