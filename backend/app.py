"""
==========================================================
Smart Insole DFU Risk Prediction
Backend App Initialization (Flask)
==========================================================
"""

import os
import sys

# Force TensorFlow to use Legacy Keras (v2) to prevent deserialization errors 
# (e.g. Unrecognized keyword arguments: ['batch_shape']) with models trained on older TF versions.
os.environ["TF_USE_LEGACY_KERAS"] = "1"
import shap

import logging
from flask import Flask, jsonify, render_template, redirect, url_for
from flask_cors import CORS

# ----------------------------------------------------------
# Project Root 
# ----------------------------------------------------------
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


def create_app() -> Flask:
    """Initialize and configure the Flask application."""
    
    app = Flask(
        __name__,
        template_folder="../frontend/templates",
        static_folder="../frontend/static"
    )
    
    # Enable CORS for all routes (important for frontend communication)
    CORS(app)

    # Configure Professional Logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    app.logger.info("Initializing Smart Insole DFU Backend...")

    # ==========================================================
    # 1. Load Model & Scaler Once
    # ==========================================================
    with app.app_context():
        try:
            # Import delayed to allow Flask to initialize without TF overhead immediately
            from backend.predictor import DfuPredictor
            app.predictor = DfuPredictor()
            app.logger.info("DFU Predictor (Model & Scaler) loaded successfully.")
        except ImportError as e:
            app.logger.warning(f"Predictor module missing (expected if predictor.py isn't built yet): {e}")
            app.predictor = None
        except Exception as e:
            app.logger.error(f"Failed to load DFU Predictor: {e}")
            app.predictor = None

    # ==========================================================
    # 2. Register Routes
    # ==========================================================
    with app.app_context():
        try:
            from backend.routes import api_bp
            app.register_blueprint(api_bp)
            app.logger.info("API Routes registered successfully.")
        except ImportError as e:
            app.logger.warning(f"Routes module missing (expected if routes.py isn't built yet): {e}")

    # ==========================================================
    # 3. Error Handling
    # ==========================================================
    
    @app.errorhandler(400)
    def bad_request_error(error):
        return jsonify({"error": "Bad Request", "message": str(error), "status": 400}), 400
        
    @app.errorhandler(404)
    def not_found_error(error):
        return jsonify({"error": "Resource not found", "status": 404}), 404

    @app.errorhandler(500)
    def internal_error(error):
        return jsonify({"error": "Internal server error", "status": 500}), 500

    @app.errorhandler(Exception)
    def unhandled_exception(e):
        app.logger.error(f"Unhandled Exception: {e}")
        return jsonify({"error": str(e), "status": 500}), 500

    # ==========================================================
    # 4. Root & Health Endpoints
    # ==========================================================
    
    @app.route("/", methods=["GET"])
    def root():
        """Root endpoint redirects to dashboard."""
        return redirect(url_for("dashboard"))

    @app.route("/dashboard", methods=["GET"])
    def dashboard():
        return render_template("dashboard.html")

    @app.route("/insole-feed", methods=["GET"])
    def insole_feed():
        return render_template("insole_feed.html")

    @app.route("/prediction", methods=["GET"])
    def prediction():
        return render_template("prediction.html")

    @app.route("/explainability", methods=["GET"])
    def explainability():
        return render_template("explainability.html")

    @app.route("/trends", methods=["GET"])
    def trends():
        return render_template("trends.html")

    @app.route("/prediction-log", methods=["GET"])
    def prediction_log():
        return render_template("prediction_log.html")

    @app.route("/patient", methods=["GET"])
    def patient():
        return render_template("patient.html")

    @app.route("/system", methods=["GET"])
    def system():
        return render_template("system.html")

    @app.route("/health", methods=["GET"])
    def health():
        """Health check endpoint to verify backend and ML model status."""
        model_status = "loaded" if getattr(app, 'predictor', None) and app.predictor.is_ready else "unavailable"
        
        status_code = 200 if model_status == "loaded" else 503
        
        return jsonify({
            "status": "healthy" if status_code == 200 else "degraded",
            "model": model_status
        }), status_code

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", port=5500, debug=True)
