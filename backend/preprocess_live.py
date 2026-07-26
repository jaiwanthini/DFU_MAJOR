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
from collections import deque
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

from config import WINDOW_SIZE, SCALER_PATH
from backend.risk import calculate_risk_score


class LivePreprocessor:
    """
    Maintains a rolling window of live sensor data.
    Computes all engineered features on the fly.
    Returns (sequence, base_risk_score) when WINDOW_SIZE is reached.
    """

    # Must match EXACTLY the order used during training (sequence_generator.py)
    FEATURE_COLUMNS = [
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
        "recovery_factor"
    ]

    def __init__(self, window_size: int = WINDOW_SIZE):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.window_size = window_size
        
        # Deque for maintaining the rolling window of fully processed feature rows
        self.history = deque(maxlen=window_size)
        
        # Deque for raw values to compute trends (diffs)
        self.raw_history = deque(maxlen=window_size)

        # Load StandardScaler identically to training pipeline
        scaler_full_path = os.path.join(ROOT_DIR, SCALER_PATH)
        self.logger.info(f"Loading StandardScaler from: {scaler_full_path}")
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

    def reset(self) -> None:
        """
        Clears the current live rolling windows.
        Useful when switching patients or resetting the simulation.
        """
        self.history.clear()
        self.raw_history.clear()
        self.logger.info("LivePreprocessor buffers have been reset.")

    def get_buffer_size(self) -> int:
        """
        Returns the number of processed frames currently buffered in the window.
        """
        return len(self.history)

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
            (Sequence of shape (1, 30, 18), base_risk_score) if window full, else None
        """
        try:
            # 1. Parse raw values safely
            fsr1 = float(raw_data.get("fsr1", 0.0))
            fsr2 = float(raw_data.get("fsr2", 0.0))
            fsr3 = float(raw_data.get("fsr3", 0.0))
            fsr4 = float(raw_data.get("fsr4", 0.0))
            
            temperature = float(raw_data.get("temperature", 36.0))
            spo2 = float(raw_data.get("spo2", 98.0))
            heart_rate = float(raw_data.get("heart_rate", 80.0))
            
            # Save raw state for diff calculations
            self.raw_history.append({
                "temperature": temperature,
                "spo2": spo2,
                "heart_rate": heart_rate
            })

            fsr = [fsr1, fsr2, fsr3, fsr4]
            total_pressure = sum(fsr)
            # Prevent division by zero
            safe_total = total_pressure if total_pressure > 0 else 1.0

            # 2. Compute Base Features
            features = {
                "avg_pressure": np.mean(fsr),
                "max_pressure": np.max(fsr),
                "pressure_std": np.std(fsr),
                "heel_ratio": fsr1 / safe_total,
                "mid_ratio": fsr2 / safe_total,
                "forefoot_ratio": fsr3 / safe_total,
                "toe_ratio": fsr4 / safe_total,
                "temperature": temperature,
                "spo2": spo2,
                "heart_rate": heart_rate
            }

            # 3. Compute Temporal Trends (Diffs)
            if len(self.raw_history) >= 2:
                prev = self.raw_history[-2]
                features["temp_diff"] = temperature - prev["temperature"]
                features["hr_diff"] = heart_rate - prev["heart_rate"]
                features["spo2_diff"] = spo2 - prev["spo2"]
            else:
                features["temp_diff"] = 0.0
                features["hr_diff"] = 0.0
                features["spo2_diff"] = 0.0

            # 4. Compute Rolling Means
            # We use the current history + this new frame to calculate rolling metrics
            hist_list = list(self.history)
            hist_list.append(features)

            features["avg_pressure_rolling_mean"] = np.mean([x["avg_pressure"] for x in hist_list])
            features["temperature_rolling_mean"] = np.mean([x["temperature"] for x in hist_list])
            features["heart_rate_rolling_mean"] = np.mean([x["heart_rate"] for x in hist_list])
            features["spo2_rolling_mean"] = np.mean([x["spo2"] for x in hist_list])

            # 5. Compute Recovery Factor
            # In prepare_data.py, baseline is patient median. Here, we approximate 
            # the baseline using the median of the current rolling window.
            avg_pressures = [x["avg_pressure"] for x in hist_list]
            baseline = np.median(avg_pressures)
            if baseline <= 0:
                baseline = 1e-5
                
            rf = (baseline - features["avg_pressure"]) / baseline
            features["recovery_factor"] = np.clip(rf, 0.0, 1.0)

            # Append finalized features to history
            self.history.append(features)

            # 6. Check if Window is Full
            if len(self.history) == self.window_size:
                
                # Extract features in the EXACT order defined by FEATURE_COLUMNS
                sequence = []
                for item in self.history:
                    row = [float(item[col]) for col in self.FEATURE_COLUMNS]
                    sequence.append(row)
                    
                seq_arr = np.array(sequence, dtype=np.float32)
                
                # Reshape to 2D for scaling
                seq_2d = seq_arr.reshape(self.window_size, len(self.FEATURE_COLUMNS))
                
                # Apply scaling EXACTLY as done in training
                seq_scaled_2d = self.scaler.transform(seq_2d)
                
                # Reshape back to 3D for LSTM (1, 30, 18)
                seq_arr = seq_scaled_2d.reshape(1, self.window_size, len(self.FEATURE_COLUMNS))
                
                # 7. Compute Rule-based Risk Score 
                # Reuses the exact clinically justified function from sequence_generator.py
                try:
                    latest_row = pd.Series(features)
                    base_risk_score = float(calculate_risk_score(latest_row))
                except Exception as e:
                    self.logger.error(f"Error calculating base risk score: {e}")
                    base_risk_score = 0.0
                
                return seq_arr, base_risk_score

            # Window not yet full (gathering 30 seconds of data)
            return None

        except Exception as e:
            self.logger.exception(f"Error processing live reading: {e}")
            raise RuntimeError(f"Live preprocessing failed: {e}")

    def clear(self) -> None:
        """Resets the live window (e.g. for a new patient or session)."""
        self.history.clear()
        self.raw_history.clear()
        self.logger.info("Live preprocessor history cleared.")
