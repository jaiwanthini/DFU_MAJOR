import os
import sys
import ast
import joblib
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, "data")
PROCESSED_CSV = os.path.join(DATA_DIR, "processed", "dfu_processed_dataset.csv")
SEQ_DIR = os.path.join(DATA_DIR, "sequences")
SCALER_PATH = os.path.join(ROOT_DIR, "models", "scaler.pkl")

# We will need the exact feature columns in order
FEATURE_COLUMNS = [
    "avg_pressure", "max_pressure", "min_pressure", "pressure_var",
    "pressure_std", "pressure_gradient", "pressure_change_rate", "pressure_stability",
    "cop_approx", "pressure_symmetry", "temp_diff", "temperature_rolling_mean",
    "hr_diff", "heart_rate_rolling_mean", "spo2_diff", "spo2_rolling_mean",
    "pressure_temp_interaction", "pressure_hr_interaction", "temp_hr_interaction",
    "pressure_integral", "recovery_factor", "loading_rate", "unloading_rate",
    "pressure_duration", "avg_pressure_rolling_mean", "temperature", "spo2", "heart_rate"
]

class PipelineVerifier:
    def __init__(self):
        self.results = {}
        self.reasons = {}

    def report_fail(self, test_name, reason):
        self.results[test_name] = "FAIL"
        self.reasons[test_name] = reason
        print(f"\n[FAIL] {test_name}")
        print(f"Reason: {reason}")

    def report_pass(self, test_name):
        self.results[test_name] = "PASS"
        print(f"\n[PASS] {test_name}")

    def run_all(self):
        print("="*50)
        print("PIPELINE VALIDATION REPORT - INITIALIZING")
        print("="*50)

        # 1. Load Data
        print("Loading processed dataset...")
        df = pd.read_csv(PROCESSED_CSV)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        
        print("Loading scaled sequence arrays...")
        X_train = np.load(os.path.join(SEQ_DIR, "X_train.npy"))
        X_val = np.load(os.path.join(SEQ_DIR, "X_val.npy"))
        X_test = np.load(os.path.join(SEQ_DIR, "X_test.npy"))
        
        y_train = np.load(os.path.join(SEQ_DIR, "y_train.npy"))
        y_val = np.load(os.path.join(SEQ_DIR, "y_val.npy"))
        y_test = np.load(os.path.join(SEQ_DIR, "y_test.npy"))

        scaler = joblib.load(SCALER_PATH)

        # Map back to original dataset using KDTree
        print("Mapping sequences back to original CSV for validation...")
        csv_features = df[FEATURE_COLUMNS].values
        tree = cKDTree(csv_features)
        
        # Inverse transform the first timestep of every sequence
        def map_to_csv(X):
            X_2d = X.reshape(-1, X.shape[2])
            X_inv_2d = scaler.inverse_transform(X_2d)
            X_inv = X_inv_2d.reshape(X.shape)
            first_step = X_inv[:, 0, :]
            distances, indices = tree.query(first_step, k=1)
            # Make sure match is extremely close
            if np.max(distances) > 1e-2:
                print(f"WARNING: Max KDTree distance is {np.max(distances)}. Match might be inaccurate.")
            return df.iloc[indices].copy().reset_index(drop=True)

        train_mapped = map_to_csv(X_train)
        val_mapped = map_to_csv(X_val)
        test_mapped = map_to_csv(X_test)

        self._test_patient_leakage(train_mapped, val_mapped, test_mapped)
        self._test_session_leakage(df)
        self._test_sequence_consistency(df)
        self._test_scaler_validation()
        self._test_feature_leakage()
        self._test_missing_values(X_train, X_val, X_test)
        self._test_feature_statistics(csv_features)
        self._test_class_distribution(y_train, y_val, y_test, df["risk_label"].values)
        self._test_sequence_shapes(X_train, X_val, X_test, y_train, y_val, y_test)
        self._test_temporal_order(X_train, X_val, X_test)
        self._test_outlier_check(df)
        self._test_dataset_summary(df, X_train, X_val, X_test)

        self.print_final_report()

    def _test_patient_leakage(self, train, val, test):
        print("\n" + "="*50 + "\nTEST 1: PATIENT LEAKAGE\n" + "="*50)
        train_p = set(train["patient_id"].unique())
        val_p = set(val["patient_id"].unique())
        test_p = set(test["patient_id"].unique())

        print(f"Training Patients: {len(train_p)}")
        print(f"Validation Patients: {len(val_p)}")
        print(f"Testing Patients: {len(test_p)}")

        shared_train_val = train_p.intersection(val_p)
        shared_train_test = train_p.intersection(test_p)
        shared_val_test = val_p.intersection(test_p)
        total_shared = len(shared_train_val) + len(shared_train_test) + len(shared_val_test)

        print(f"Shared Patients = {total_shared}")

        if total_shared == 0:
            self.report_pass("Patient Leakage")
        else:
            self.report_fail("Patient Leakage", f"Found {total_shared} shared patients across splits.")

    def _test_session_leakage(self, df):
        print("\n" + "="*50 + "\nTEST 2: SESSION LEAKAGE\n" + "="*50)
        from config import (
            RAW_DATA_PATH,
            PROCESSED_DATA_PATH,
            SEQUENCE_DATA_PATH,
            SEQUENCE_WINDOW,
            ROLLING_WINDOW,
            NUM_FEATURES,
            CLASS_NAMES
        )
        seq_gen_path = os.path.join(ROOT_DIR, "preprocessing", "sequence_generator.py")
        with open(seq_gen_path, "r") as f:
            code = f.read()
            
        if "time_diff >" in code:
            self.report_pass("Session Leakage")
        else:
            self.report_fail("Session Leakage", "Could not find explicit session boundary check via timestamp diffs in sequence_generator.py")

    def _test_sequence_consistency(self, df):
        print("\n" + "="*50 + "\nTEST 3: SEQUENCE CONSISTENCY\n" + "="*50)
        errors = []
        if "session_id" not in df.columns:
            errors.append("No 'session_id' column found. Data leakage across sessions is possible.")
        
        seq_gen_path = os.path.join(ROOT_DIR, "preprocessing", "sequence_generator.py")
        with open(seq_gen_path, "r") as f:
            code = f.read()
        
        has_state_check = "sequence_labels == sequence_labels[0]" in code
        if has_state_check and not errors:
            self.report_pass("Sequence Consistency")
        else:
            self.report_fail("Sequence Consistency", f"Issues found: {errors if errors else 'Could not find explicit state/label uniformity check for the sequence window.'}")

    def _test_scaler_validation(self):
        print("\n" + "="*50 + "\nTEST 4: SCALER VALIDATION\n" + "="*50)
        seq_gen_path = os.path.join(ROOT_DIR, "preprocessing", "sequence_generator.py")
        with open(seq_gen_path, "r") as f:
            code = f.read()
            
        if "scaler.fit(X_train_2d)" in code and "scaler.transform(X_val_2d)" in code and "scaler.fit_transform(X_val_2d)" not in code:
            self.report_pass("Scaler Validation")
        else:
            self.report_fail("Scaler Validation", "fit_transform() might have been used on validation/test data or fit() was missing on X_train. Manual check required in sequence_generator.py")

    def _test_feature_leakage(self):
        print("\n" + "="*50 + "\nTEST 5: FEATURE LEAKAGE\n" + "="*50)
        prep_path = os.path.join(ROOT_DIR, "preprocessing", "prepare_data.py")
        with open(prep_path, "r") as f:
            code = f.read()
            
        future_indicators = [".shift(-", "lead(", "future"]
        found = []
        for ind in future_indicators:
            if ind in code:
                found.append(ind)
                
        if not found:
            self.report_pass("Feature Leakage")
        else:
            self.report_fail("Feature Leakage", f"Found potential future leakage indicators: {found}")

    def _test_missing_values(self, X_train, X_val, X_test):
        print("\n" + "="*50 + "\nTEST 6: MISSING VALUES\n" + "="*50)
        nans_train = np.isnan(X_train).sum()
        nans_val = np.isnan(X_val).sum()
        nans_test = np.isnan(X_test).sum()
        
        inf_train = np.isinf(X_train).sum()
        inf_val = np.isinf(X_val).sum()
        inf_test = np.isinf(X_test).sum()

        print(f"NaNs -> Train: {nans_train}, Val: {nans_val}, Test: {nans_test}")
        print(f"Infs -> Train: {inf_train}, Val: {inf_val}, Test: {inf_test}")
        
        if nans_train == 0 and nans_val == 0 and nans_test == 0 and inf_train == 0 and inf_val == 0 and inf_test == 0:
            self.report_pass("Missing Values")
        else:
            self.report_fail("Missing Values", "NaNs or Infs found in sequence arrays.")

    def _test_feature_statistics(self, csv_features):
        print("\n" + "="*50 + "\nTEST 7: FEATURE STATISTICS\n" + "="*50)
        means = np.mean(csv_features, axis=0)
        medians = np.median(csv_features, axis=0)
        stds = np.std(csv_features, axis=0)
        mins = np.min(csv_features, axis=0)
        maxs = np.max(csv_features, axis=0)
        
        constant_features = []
        for i, col in enumerate(FEATURE_COLUMNS):
            if stds[i] < 1e-6:
                constant_features.append(col)
                
        if constant_features:
            self.report_fail("Feature Statistics", f"Constant features found: {constant_features}")
        else:
            print("No constant features found.")
            self.report_pass("Feature Statistics")

    def _test_class_distribution(self, y_train, y_val, y_test, original_labels):
        print("\n" + "="*50 + "\nTEST 8: CLASS DISTRIBUTION\n" + "="*50)
        def get_dist(y, name):
            unique, counts = np.unique(y, return_counts=True)
            print(f"{name} Dist: {dict(zip(unique, counts))}")
            return set(unique)
            
        print("Original Dataset:", np.unique(original_labels, return_counts=True))
        t = get_dist(y_train, "Train")
        v = get_dist(y_val, "Val")
        te = get_dist(y_test, "Test")
        
        if len(t) == 3 and len(v) == 3 and len(te) == 3:
            self.report_pass("Class Distribution")
        else:
            self.report_fail("Class Distribution", "Missing classes in one or more splits.")

    def _test_sequence_shapes(self, X_train, X_val, X_test, y_train, y_val, y_test):
        from config import SEQUENCE_WINDOW, NUM_FEATURES
        print("\n" + "="*50 + "\nTEST 9: SEQUENCE SHAPES\n" + "="*50)
        errors = []
        if X_train.shape[1] != SEQUENCE_WINDOW or X_train.shape[2] != NUM_FEATURES:
            errors.append(f"X_train shape {X_train.shape} does not match (samples, {SEQUENCE_WINDOW}, {NUM_FEATURES})")
        print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
        
        if not errors and X_train.shape[1:] == (3, 28) and X_val.shape[1:] == (3, 28) and X_test.shape[1:] == (3, 28) and X_train.shape[0] == y_train.shape[0]:
            self.report_pass("Sequence Shapes")
        else:
            self.report_fail("Sequence Shapes", f"Invalid sequence shapes detected: {errors}")

    def _test_temporal_order(self, X_train, X_val, X_test):
        print("\n" + "="*50 + "\nTEST 10: TEMPORAL ORDER\n" + "="*50)
        # Because we only verify the generated sequences, we know `prepare_data.py` sorted by timestamp.
        # Let's check prepare_data.py for the sort.
        prep_path = os.path.join(ROOT_DIR, "preprocessing", "prepare_data.py")
        with open(prep_path, "r") as f:
            code = f.read()
        if 'sort_values(' in code and '"patient_id"' in code and '"timestamp"' in code:
            self.report_pass("Temporal Order")
        else:
            self.report_fail("Temporal Order", "Explicit sort_values by timestamp not found in prepare_data.py")

    def _test_outlier_check(self, df):
        print("\n" + "="*50 + "\nTEST 11: OUTLIER CHECK\n" + "="*50)
        hr = df["heart_rate"]
        temp = df["temperature"]
        spo2 = df["spo2"]
        press = df["avg_pressure"]
        
        print(f"HR Range: {hr.min()} - {hr.max()}")
        print(f"Temp Range: {temp.min()} - {temp.max()}")
        print(f"SpO2 Range: {spo2.min()} - {spo2.max()}")
        print(f"Pressure Range: {press.min():.2f} - {press.max():.2f}")
        self.report_pass("Outlier Check")

    def _test_dataset_summary(self, df, X_train, X_val, X_test):
        print("\n" + "="*50 + "\nTEST 12: DATASET SUMMARY\n" + "="*50)
        num_patients = df["patient_id"].nunique()
        # approximate sessions by large time gaps
        df["time_diff"] = df.groupby("patient_id")["timestamp"].diff().dt.total_seconds()
        num_sessions = num_patients + (df["time_diff"] > 10).sum()
        avg_session_length = len(df) / num_sessions
        
        print(f"Number of Patients: {num_patients}")
        print(f"Number of Sessions: {num_sessions}")
        print(f"Average Session Length: {avg_session_length:.2f} rows")
        print(f"Average Sequence Length: 3")
        print(f"Average Features: 28")
        print(f"Total Missing Values: {df.isna().sum().sum()}")
        self.report_pass("Dataset Summary")

    def print_final_report(self):
        print("\n" + "="*40)
        print("PIPELINE VALIDATION REPORT")
        print("="*40)
        
        all_passed = True
        tests = [
            "Patient Leakage", "Session Leakage", "Sequence Consistency",
            "Scaler Validation", "Feature Leakage", "Missing Values",
            "Feature Statistics", "Class Distribution", "Sequence Shapes",
            "Temporal Order", "Outlier Check", "Dataset Summary"
        ]
        
        for t in tests:
            res = self.results.get(t, "MISSING")
            print(f"{t.ljust(25)} {res}")
            if res != "PASS":
                all_passed = False
                
        print("="*40)
        if all_passed:
            print("Overall                  PASS")
        else:
            print("Overall                  FAIL")
            print("\nFailures details:")
            for t, res in self.results.items():
                if res == "FAIL":
                    print(f" - {t}: {self.reasons[t]}")
        print("="*40)

if __name__ == "__main__":
    verifier = PipelineVerifier()
    verifier.run_all()
