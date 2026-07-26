"""
==========================================================
Smart Insole DFU Risk Prediction
REST API Routes

Exposes endpoints for Simulation, Prediction, SHAP XAI, 
History tracking, System Health, and System Reset.
==========================================================
"""

from flask import Blueprint, request, jsonify, current_app

from config import WINDOW_SIZE
from backend.simulator import SensorSimulator
from backend.preprocess_live import LivePreprocessor
from backend.shap_explainer import DfuShapExplainer
from backend.history import history_manager

api_bp = Blueprint('api_bp', __name__)

# ==========================================================
# Singletons for live state
# ==========================================================
simulator = SensorSimulator(mode="Normal")
preprocessor = LivePreprocessor()

# Initialize extensions dict safely in case it doesn't exist
def _get_extensions():
    if not hasattr(current_app, 'extensions'):
        current_app.extensions = {}
    return current_app.extensions


# ==========================================================
# Simulator Endpoint
# ==========================================================
@api_bp.route('/simulate', methods=['GET'])
def simulate():
    """
    Generates and returns one frame of simulated sensor data.
    Query param ?mode= can dynamically change the clinical state.
    """
    mode = request.args.get('mode')
    if mode:
        try:
            simulator.set_mode(mode)
        except ValueError as e:
            return jsonify({"error": str(e)}), 400
            
    reading = simulator.get_reading()
    return jsonify(reading), 200


# ==========================================================
# Prediction Endpoint
# ==========================================================
@api_bp.route('/predict', methods=['POST'])
def predict():
    """
    Primary pipeline endpoint:
    1. Receives raw sensor payload
    2. Runs feature engineering and adds to rolling window
    3. If window is full (30 secs), runs LSTM prediction
    4. Generates SHAP explainability
    5. Stores result in history
    """
    # 1. Strict Content-Type Validation
    if not request.is_json:
        return jsonify({"error": "Content-Type must be application/json."}), 415

    raw_data = request.get_json()
    if not raw_data:
        return jsonify({"error": "No JSON payload provided"}), 400

    # 2. Retrieve ML Predictor safely
    exts = _get_extensions()
    # Check both extensions dictionary (requested) and direct app attribute (legacy compatibility)
    predictor = exts.get("predictor", getattr(current_app, "predictor", None))
    
    if not predictor or not getattr(predictor, "is_ready", False):
        return jsonify({"error": "ML Predictor is not ready or failed to load."}), 503

    # 3. Preprocess
    try:
        result = preprocessor.process_reading(raw_data)
    except Exception as e:
        current_app.logger.exception("Preprocessing failed:")
        return jsonify({"error": f"Preprocessing failed: {str(e)}"}), 500

    # If window is not yet full, return a highly informative buffering status
    if result is None:
        samples = preprocessor.get_buffer_size()
        progress = round((samples / WINDOW_SIZE) * 100, 1) if WINDOW_SIZE > 0 else 0
        
        return jsonify({
            "status": "buffering",
            "message": "Collecting enough sensor samples for prediction.",
            "samples_received": samples,
            "required_samples": WINDOW_SIZE,
            "progress": progress
        }), 202

    sequence, base_risk_score = result

    # 4. Predict Risk
    try:
        prediction = predictor.predict(sequence, base_risk_score)
    except Exception as e:
        current_app.logger.exception("Prediction failed:")
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

    # 5. Explain (SHAP XAI)
    try:
        # Cache the explainer in current_app.extensions to reuse across requests
        if "shap_explainer" not in exts:
            exts["shap_explainer"] = DfuShapExplainer(predictor.model)
            
        shap_explainer = exts["shap_explainer"]
            
        class_name_to_idx = {"Low": 0, "Medium": 1, "High": 2}
        pred_idx = class_name_to_idx.get(prediction.get("risk_label", "Low"), 0)
        
        explanation = shap_explainer.explain(sequence, pred_idx, prediction.get("risk_label", "Low"))
        prediction["explanation"] = explanation
    except Exception as e:
        current_app.logger.exception("SHAP explanation failed:")
        prediction["explanation"] = {"error": "SHAP generation failed."}

    # 6. Save to History 
    try:
        history_manager.add(prediction)
    except Exception as e:
        current_app.logger.exception("Failed to save history:")

    return jsonify(prediction), 200


# ==========================================================
# History Endpoint
# ==========================================================
@api_bp.route('/history', methods=['GET'])
def get_history():
    """
    Returns the most recent prediction history.
    Supports query param ?limit=N to restrict the payload size.
    """
    limit_param = request.args.get('limit')
    
    if limit_param:
        try:
            limit = int(limit_param)
            return jsonify(history_manager.get_recent(limit)), 200
        except ValueError:
            return jsonify({"error": "Invalid limit parameter. Must be an integer."}), 400
            
    return jsonify(history_manager.get_all()), 200


# ==========================================================
# SHAP Endpoint
# ==========================================================
@api_bp.route('/shap', methods=['GET'])
def get_shap():
    """Returns the XAI explanation for the most recent prediction."""
    latest = history_manager.get_latest()
    if latest and "explanation" in latest:
        return jsonify(latest["explanation"]), 200
    return jsonify({"error": "No explanation available yet."}), 404


# ==========================================================
# Health Endpoint
# ==========================================================
@api_bp.route('/health', methods=['GET'])
def health_check():
    """
    Returns overall system status, model availability, 
    and sensor state without generating a prediction.
    """
    exts = _get_extensions()
    predictor = exts.get("predictor", getattr(current_app, "predictor", None))
    
    predictor_status = "ready" if predictor and getattr(predictor, "is_ready", False) else "unavailable"
    shap_status = "ready" if "shap_explainer" in exts else "unavailable"
    
    return jsonify({
        "status": "healthy",
        "predictor": predictor_status,
        "shap": shap_status,
        "history_records": history_manager.size(),
        "simulator_mode": simulator.mode
    }), 200


# ==========================================================
# Reset Endpoint
# ==========================================================
@api_bp.route('/reset', methods=['POST'])
def reset_system():
    """
    Wipes the system slate clean.
    Clears history, resets the live rolling window, reverts 
    simulator to Normal mode, and drops the SHAP cache.
    """
    try:
        # Clear dashboard history
        history_manager.clear()
        
        # Reset the live feature engineering buffers
        preprocessor.reset()
        
        # Revert simulator to default clinical state
        simulator.reset()
        
        # Remove the cached shap explainer (will be lazily rebuilt on next prediction)
        exts = _get_extensions()
        exts.pop("shap_explainer", None)
        
        return jsonify({"message": "System reset successfully."}), 200
    except Exception as e:
        current_app.logger.exception("Reset failed:")
        return jsonify({"error": f"Reset failed: {str(e)}"}), 500
