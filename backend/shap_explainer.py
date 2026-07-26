"""
==========================================================
Smart Insole DFU Risk Prediction
SHAP Explainer Module

Generates feature importance and human-readable clinical
explanations for the LSTM model predictions.
==========================================================
"""

import logging
import numpy as np
import shap

from config import WINDOW_SIZE, NUM_FEATURES, FEATURE_DISPLAY_NAMES

class DfuShapExplainer:
    """
    Initializes SHAP GradientExplainer once to avoid overhead.
    Translates raw SHAP values into readable clinical explanations.
    """

    def __init__(self, model):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.model = model
        self.explainer = None
        
        self._initialize_explainer()

    def _initialize_explainer(self):
        """
        Creates the SHAP explainer using a neutral baseline.
        For scaled sequences (StandardScaler), a zero-array represents the mean 
        of the training data, which serves as an excellent neutral baseline for SHAP.
        """
        try:
            # Baseline: shape (1, WINDOW_SIZE, NUM_FEATURES) of zeros
            baseline = [np.zeros((1, WINDOW_SIZE, NUM_FEATURES), dtype=np.float32)]
            self.explainer = shap.GradientExplainer(self.model, baseline)
            self.logger.info("SHAP GradientExplainer initialized successfully.")
        except Exception as e:
            self.logger.exception(f"Failed to initialize SHAP explainer: {e}")
            self.explainer = None

    def explain(self, sequence: np.ndarray, predicted_class_idx: int, risk_label: str) -> dict:
        """
        Calculates SHAP values for the given sequence and constructs a readable explanation.
        
        Parameters
        ----------
        sequence : np.ndarray
            The scaled sequence used for prediction.
        predicted_class_idx : int
            The predicted class index (0=Low, 1=Medium, 2=High).
        risk_label : str
            The text label of the prediction.

        Returns
        -------
        dict
            Dictionary containing top features, positive/negative contributions, 
            a text summary, and a raw feature_importance dictionary.
        """
        if self.explainer is None:
            return self._fallback_explanation(risk_label)
            
        expected_shape = (1, WINDOW_SIZE, NUM_FEATURES)
        if sequence.shape != expected_shape:
            raise ValueError(
                f"Invalid sequence shape for SHAP explanation. "
                f"Expected {expected_shape}, received {sequence.shape}"
            )

        try:
            # shap_values format for Keras multi-class: list of arrays, one per class.
            # Each array is shape matching the input sequence.
            shap_values = self.explainer.shap_values(sequence)
            
            if isinstance(shap_values, list):
                class_shap = shap_values[predicted_class_idx]
            else:
                class_shap = shap_values
                
            shap_array = np.asarray(class_shap)
            self.logger.info(f"Raw SHAP shape: {shap_array.shape}")
            
            shap_array = np.squeeze(shap_array)
            
            if len(shap_array.shape) == 1 and shap_array.shape[0] == NUM_FEATURES:
                temporal_mean_shap = shap_array
            elif len(shap_array.shape) == 2 and shap_array.shape[0] == WINDOW_SIZE and shap_array.shape[1] == NUM_FEATURES:
                temporal_mean_shap = np.mean(shap_array, axis=0)
            elif len(shap_array.shape) == 3 and shap_array.shape[0] == WINDOW_SIZE and shap_array.shape[1] == NUM_FEATURES:
                # E.g., (30, 18, 3) where 3 is the class dimension
                try:
                    class_specific = shap_array[:, :, predicted_class_idx]
                    temporal_mean_shap = np.mean(class_specific, axis=0)
                except IndexError:
                    # Fallback to mean if class index is out of bounds for some reason
                    temporal_mean_shap = np.mean(shap_array, axis=(0, 2))
            else:
                raise ValueError(f"Unexpected SHAP shape after squeeze: {shap_array.shape}")
            
            # Map SHAP values to human readable feature names
            feature_contributions = []
            for i in range(NUM_FEATURES):
                val_scalar = float(np.ravel(temporal_mean_shap[i])[0])
                feature_contributions.append({
                    "feature": FEATURE_DISPLAY_NAMES[i], 
                    "value": round(val_scalar, 4)
                })
            
            # Create a comprehensive dictionary of all raw feature importances for charting
            feature_importance_dict = {f["feature"]: f["value"] for f in feature_contributions}
            
            # Sort by absolute impact (highest magnitude first)
            feature_contributions.sort(key=lambda x: abs(x["value"]), reverse=True)
            
            # Separate positive (pushing towards the predicted risk) and negative (pushing away)
            positive_contribs = [f for f in feature_contributions if f["value"] > 0]
            negative_contribs = [f for f in feature_contributions if f["value"] < 0]
            
            # Get top 3 features overall
            top_features = [f["feature"] for f in feature_contributions[:3]]
            
            # Build human readable clinical summary
            summary = self._build_summary(risk_label, positive_contribs, negative_contribs)

            return {
                "top_features": top_features,
                "positive_contributions": positive_contribs[:5],
                "negative_contributions": negative_contribs[:5],
                "feature_importance": feature_importance_dict,
                "summary": summary
            }

        except Exception as e:
            self.logger.exception(f"SHAP explanation failed during calculation: {e}")
            return self._fallback_explanation(risk_label)

    def _build_summary(self, risk_label: str, positive: list, negative: list) -> str:
        """Constructs a natural language summary of the SHAP findings."""
        if not positive:
            return f"The model predicted {risk_label} Risk based on a complex combination of temporal features."
            
        top_pos = positive[0]["feature"]
        second_pos = positive[1]["feature"] if len(positive) > 1 else None
        
        # Find the strongest negative contributor if any exist
        top_neg = negative[0]["feature"] if negative else None
        
        if risk_label == "High" or risk_label == "Medium":
            summary = f"{top_pos} was the strongest contributor to the predicted {risk_label} Risk."
            if second_pos:
                summary += f" Increased {second_pos} also increased the model's confidence"
                if top_neg:
                    summary += f", while {top_neg} slightly reduced the predicted risk."
                else:
                    summary += "."
        else:
            summary = f"Normal {top_pos} strongly indicated a Low Risk state."
            if second_pos:
                summary += f" Stable {second_pos} also contributed to this safe classification."
            
        return summary

    def _fallback_explanation(self, risk_label: str) -> dict:
        """Fallback response if SHAP calculation fails or is unavailable."""
        return {
            "top_features": ["Unknown (SHAP unavailable)"],
            "positive_contributions": [],
            "negative_contributions": [],
            "feature_importance": {},
            "summary": f"The model predicted {risk_label} Risk. (Detailed feature explanation is currently unavailable)."
        }
