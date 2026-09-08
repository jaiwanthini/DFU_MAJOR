"""
==========================================================
Smart Insole DFU Risk Prediction
Live Data Preprocessor

Handles real-time sensor streams, engineers features
identically to the training pipeline, and maintains
a 30-second rolling window deque.
==========================================================
"""

import os
import sys
import logging
from typing import Dict, Any, Tuple, Optional

import numpy as np
import pandas as pd
import joblib

# ----------------------------------------------------------
# Project Root
# ----------------------------------------------------------
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from config import SEQUENCE_WINDOW, ROLLING_WINDOW, SCALER_PATH, MODEL_FEATURE_COLUMNS
from backend.risk import calculate_risk_score


class LivePreprocessor:
    """
    Maintains a rolling window of live sensor data.
    Computes all engineered features on the fly.
    Returns (sequence, base_risk_score) when SEQUENCE_WINDOW is reached.
    Maintains a 30-second rolling buffer for clinical features.
    """

    def __init__(self, sequence_window: int = SEQUENCE_WINDOW, rolling_window: int = ROLLING_WINDOW):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.sequence_window = sequence_window
        self.rolling_window = rolling_window
        
        # Buffer needs to hold enough for rolling features, so we use rolling_window
        self.buffer = pd.DataFrame(columns=MODEL_FEATURE_COLUMNS)
        
        # Explicit stateful causality for the session
        self.session_pressures = []
        
        # Load StandardScaler identically to training pipeline
        scaler_full_path = os.path.join(ROOT_DIR, SCALER_PATH)
        self.logger.info(f"Loading Scaler from: {scaler_full_path}")
        if not os.path.exists(scaler_full_path):
            raise RuntimeError(f"Scaler missing at {scaler_full_path}")
            
        # Patch for loading NumPy 2.x pickles in NumPy 1.x environments
        import sys
        import numpy.core
        import numpy.core.multiarray
        if 'numpy._core' not in sys.modules:
            sys.modules['numpy._core'] = numpy.core
        if 'numpy._core.multiarray' not in sys.modules:
            sys.modules['numpy._core.multiarray'] = numpy.core.multiarray
            
        self.scaler = joblib.load(scaler_full_path)
        
        # Runtime Validations
        assert len(MODEL_FEATURE_COLUMNS) == 28, f"Expected 28 features, got {len(MODEL_FEATURE_COLUMNS)}"
        assert len(MODEL_FEATURE_COLUMNS) == self.scaler.n_features_in_, f"Scaler expects {self.scaler.n_features_in_} features, but {len(MODEL_FEATURE_COLUMNS)} are defined."

    def reset(self) -> None:
        """
        Clears the current live rolling windows.
        Useful when switching patients or resetting the simulation.
        """
        self.clear()
        
    def clear(self) -> None:
        """Resets the live window (e.g. for a new patient or session)."""
        self.buffer = pd.DataFrame(columns=MODEL_FEATURE_COLUMNS)
        self.session_pressures.clear()
        self.logger.info("Live preprocessor history cleared.")

    def get_buffer_size(self) -> int:
        """
        Returns the number of processed frames currently buffered in the window.
        """
        return len(self.buffer)

    def process_reading(self, raw_data: Dict[str, Any]) -> Optional[Tuple[np.ndarray, float]]:
        """
        Takes a raw sensor reading, engineers features, adds it to the rolling window.
        Returns a formatted numpy array and risk score if the window is full.

        Parameters
        ----------
        raw_data : Dict[str, Any]
            e.g. {"fsr1": 100, "fsr2": 200, "fsr3": 300, "fsr4": 50, 
                  "temperature": 36.5, "spo2": 98, "heart_rate": 80}

        Returns
        -------
        Tuple[np.ndarray, float] or None
            (Sequence of shape (1, seq, 28), base_risk_score) if window full, else None
        """
        try:
            # 1. Parse raw values safely
            fsr1 = float(raw_data.get("fsr1", 0.0))
            fsr2 = float(raw_data.get("fsr2", 0.0))
            fsr3 = float(raw_data.get("fsr3", 0.0))
            fsr4 = float(raw_data.get("fsr4", 0.0))
            
            temperature = float(raw_data.get("temperature", 31.0))
            spo2 = float(raw_data.get("spo2", 98.0))
            heart_rate = float(raw_data.get("heart_rate", 60.0))
            
            fsr = [fsr1, fsr2, fsr3, fsr4]
            total_pressure = sum(fsr)

            # 2. Compute Base Features
            avg_pressure = np.mean(fsr)
            self.session_pressures.append(avg_pressure)
            
            # Use ddof=1 for variance/std to match pandas exactly (sample variance)
            if len(fsr) > 1:
                pressure_var = np.var(fsr, ddof=1)
                pressure_std = np.std(fsr, ddof=1)
            else:
                pressure_var = 0.0
                pressure_std = 0.0

            pressure_stability = pressure_std / (avg_pressure + 1e-6)
            cop_approx = ((fsr1 + fsr2) - (fsr3 + fsr4)) / (total_pressure + 1e-6)
            pressure_symmetry = abs(fsr2 - fsr3) / (fsr2 + fsr3 + 1e-6)
            
            new_row = {
                "avg_pressure": avg_pressure,
                "max_pressure": np.max(fsr),
                "min_pressure": np.min(fsr),
                "pressure_var": pressure_var,
                "pressure_std": pressure_std,
                "pressure_stability": pressure_stability,
                "cop_approx": cop_approx,
                "pressure_symmetry": pressure_symmetry,
                "temperature": temperature,
                "spo2": spo2,
                "heart_rate": heart_rate,
                "pressure_temp_interaction": avg_pressure * temperature,
                "pressure_hr_interaction": avg_pressure * heart_rate,
                "temp_hr_interaction": temperature * heart_rate
            }

            # 3. Compute Temporal Trends (Diffs)
            if not self.buffer.empty:
                prev = self.buffer.iloc[-1]
                new_row["temp_diff"] = temperature - prev["temperature"]
                new_row["hr_diff"] = heart_rate - prev["heart_rate"]
                new_row["spo2_diff"] = spo2 - prev["spo2"]
                new_row["pressure_change_rate"] = avg_pressure - prev["avg_pressure"]
                new_row["pressure_gradient"] = new_row["pressure_change_rate"] - prev["pressure_change_rate"]
            else:
                new_row["temp_diff"] = 0.0
                new_row["hr_diff"] = 0.0
                new_row["spo2_diff"] = 0.0
                new_row["pressure_change_rate"] = 0.0
                new_row["pressure_gradient"] = 0.0
                
            new_row["loading_rate"] = new_row["pressure_change_rate"] if new_row["pressure_change_rate"] > 0 else 0.0
            new_row["unloading_rate"] = abs(new_row["pressure_change_rate"]) if new_row["pressure_change_rate"] < 0 else 0.0

            # Add to buffer
            self.buffer = pd.concat([self.buffer, pd.DataFrame([new_row])], ignore_index=True)

            # 4. Compute Rolling Means and Integrals
            self.buffer["avg_pressure_rolling_mean"] = self.buffer["avg_pressure"].rolling(self.rolling_window, min_periods=1).mean()
            self.buffer["temperature_rolling_mean"] = self.buffer["temperature"].rolling(self.rolling_window, min_periods=1).mean()
            self.buffer["heart_rate_rolling_mean"] = self.buffer["heart_rate"].rolling(self.rolling_window, min_periods=1).mean()
            self.buffer["spo2_rolling_mean"] = self.buffer["spo2"].rolling(self.rolling_window, min_periods=1).mean()
            self.buffer["pressure_integral"] = self.buffer["avg_pressure"].rolling(self.rolling_window, min_periods=1).sum()

            # 5. Compute explicitly stateful features (Recovery and Duration)
            baseline = np.median(self.session_pressures)
            
            # We recalculate these for the entire buffer because the baseline changes slightly each step,
            # but we only really need it for the current row to be mathematically identical to the expanding median.
            current_idx = len(self.buffer) - 1
            
            recovery_val = (baseline - avg_pressure) / (baseline + 1e-6)
            self.buffer.at[current_idx, "recovery_factor"] = max(0.0, min(recovery_val, 1.0))
            
            if "is_high" not in self.buffer.columns:
                self.buffer["is_high"] = 0
                
            self.buffer.at[current_idx, "is_high"] = 1 if (avg_pressure > baseline) else 0
            self.buffer["pressure_duration"] = self.buffer["is_high"].rolling(self.rolling_window, min_periods=1).sum()

            # Keep only what we need for rolling features
            if len(self.buffer) > self.rolling_window * 2:
                self.buffer = self.buffer.iloc[-self.rolling_window:].reset_index(drop=True)
            
            # 6. Check if Window is Full
            if len(self.buffer) >= self.sequence_window:
                seq_df = self.buffer.iloc[-self.sequence_window:].copy()
                
                # Reorder to MODEL_FEATURE_COLUMNS
                features_data = seq_df[MODEL_FEATURE_COLUMNS].values
                
                # Apply scaling
                scaled_features = self.scaler.transform(features_data)
                
                # Reshape to 3D for LSTM (1, seq_window, features)
                seq_arr = scaled_features.reshape(1, self.sequence_window, len(MODEL_FEATURE_COLUMNS))
                
                # 7. Compute Rule-based Risk Score
                try:
                    latest_row = seq_df.iloc[-1]
                    base_risk_score = float(calculate_risk_score(latest_row))
                except Exception as e:
                    self.logger.error(f"Error calculating base risk score: {e}")
                    base_risk_score = 0.0
                
                return seq_arr, base_risk_score

            # Window not yet full (gathering SEQUENCE_WINDOW of data)
            return None

        except Exception as e:
            self.logger.exception(f"Error processing live reading: {e}")
            raise RuntimeError(f"Live preprocessing failed: {e}")
