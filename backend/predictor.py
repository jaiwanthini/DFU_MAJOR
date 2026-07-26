"""
==========================================================
Smart Insole DFU Risk Prediction
ML Predictor Module

Loads the trained LSTM model and performs inference.

NOTE:
This module DOES NOT perform feature engineering or scaling.
Incoming sequences must already be preprocessed and scaled.

Input Shape:
(1, WINDOW_SIZE, NUM_FEATURES)

Output:
Risk label
Risk score
Confidence
Class probabilities
==========================================================
"""

import os
import sys
import logging
from typing import Dict, Any

import joblib
import numpy as np
import tensorflow as tf

# ----------------------------------------------------------
# Project Root
# ----------------------------------------------------------
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from config import (
    MODEL_PATH,
    SCALER_PATH,
    WINDOW_SIZE,
    NUM_FEATURES,
    CLASS_NAMES
)


class DfuPredictor:
    """
    Wrapper around the trained LSTM model.

    Responsibilities
    ----------------
    - Load trained model
    - Load scaler (for future compatibility)
    - Perform inference
    - Return prediction results

    NOTE:
    Scaling is intentionally NOT performed here.
    preprocess_live.py is responsible for feature engineering
    and scaling before calling this predictor.
    """

    def __init__(self):

        self.logger = logging.getLogger(self.__class__.__name__)

        self.model = None
        self.scaler = None

        self.is_ready = False

        # Used later by SHAP
        self.last_sequence = None

        self._load_artifacts()

    # ======================================================
    # Load Model
    # ======================================================

    def _load_artifacts(self):

        model_path = os.path.join(ROOT_DIR, MODEL_PATH)
        scaler_path = os.path.join(ROOT_DIR, SCALER_PATH)

        try:

            self.logger.info("Loading StandardScaler...")

            if not os.path.exists(scaler_path):
                raise FileNotFoundError(
                    f"Scaler not found:\n{scaler_path}"
                )

            # Patch for loading NumPy 2.x pickles in NumPy 1.x environments
            import sys
            import numpy.core
            import numpy.core.multiarray
            if 'numpy._core' not in sys.modules:
                sys.modules['numpy._core'] = numpy.core
            if 'numpy._core.multiarray' not in sys.modules:
                sys.modules['numpy._core.multiarray'] = numpy.core.multiarray

            self.scaler = joblib.load(scaler_path)
            self.logger.info("Loading LSTM model...")

            if not os.path.exists(model_path):
                raise FileNotFoundError(
                    f"Model not found:\n{model_path}"
                )

            self.model = tf.keras.models.load_model(model_path)

            self.is_ready = True

            self.logger.info("Predictor initialized successfully.")

        except Exception as e:

            self.logger.exception(e)

            self.is_ready = False

            raise RuntimeError(
                f"Unable to initialize predictor:\n{e}"
            )

    # ======================================================
    # Predict
    # ======================================================

    def predict(
        self,
        sequence: np.ndarray,
        base_risk_score: float = 0.0
    ) -> Dict[str, Any]:

        if not self.is_ready:
            raise RuntimeError("Predictor is not initialized.")

        expected_shape = (
            1,
            WINDOW_SIZE,
            NUM_FEATURES
        )

        if sequence.shape != expected_shape:

            raise ValueError(
                f"Expected shape {expected_shape}, "
                f"received {sequence.shape}"
            )

        try:

            # Save for SHAP later
            self.last_sequence = sequence.copy()

            # --------------------------------------------------
            # Model Prediction
            # --------------------------------------------------

            probabilities = self.model.predict(
                sequence,
                verbose=0
            )[0]

            predicted_index = int(
                np.argmax(probabilities)
            )

            confidence = float(
                probabilities[predicted_index] * 100
            )

            # --------------------------------------------------
            # Continuous Risk Score
            # --------------------------------------------------

            model_risk_score = (
                probabilities[1] * 50 +
                probabilities[2] * 100
            )

            # Blend rule-based + model score

            if base_risk_score > 0:

                final_risk_score = (
                    0.5 * base_risk_score +
                    0.5 * model_risk_score
                )

            else:

                final_risk_score = model_risk_score

            # --------------------------------------------------

            return {

                "risk_label":
                    CLASS_NAMES[predicted_index],

                "risk_score":
                    round(float(final_risk_score), 2),

                "confidence":
                    round(confidence, 2),

                "probabilities": {

                    "Low":
                        round(float(probabilities[0]), 4),

                    "Medium":
                        round(float(probabilities[1]), 4),

                    "High":
                        round(float(probabilities[2]), 4)

                }

            }

        except Exception as e:

            self.logger.exception(e)

            raise RuntimeError(
                f"Prediction failed:\n{e}"
            )
