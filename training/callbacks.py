"""
==========================================================
Smart Insole DFU Risk Prediction
Training Callbacks
==========================================================
"""

import os
import sys

from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint,
    TensorBoard,
    CSVLogger
)

from datetime import datetime

# Ensure project root is on path so config is importable
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from config import MODEL_PATH

# Resolve to absolute path
MODEL_PATH = os.path.join(_ROOT, MODEL_PATH)


# ==========================================================
# Create Directories
# ==========================================================

def create_directories():

    os.makedirs("models", exist_ok=True)

    os.makedirs("logs", exist_ok=True)

    os.makedirs("reports", exist_ok=True)


# ==========================================================
# Early Stopping
# ==========================================================

def get_early_stopping():

    return EarlyStopping(

        monitor="val_loss",

        patience=10,

        restore_best_weights=True,

        verbose=1

    )


# ==========================================================
# Reduce Learning Rate
# ==========================================================

def get_reduce_lr():

    return ReduceLROnPlateau(

        monitor="val_loss",

        factor=0.5,

        patience=5,

        min_lr=1e-6,

        verbose=1

    )


# ==========================================================
# Save Best Model
# ==========================================================

def get_checkpoint():

    return ModelCheckpoint(

        filepath=MODEL_PATH,

        monitor="val_accuracy",

        save_best_only=True,

        mode="max",

        verbose=1

    )


# ==========================================================
# TensorBoard
# ==========================================================

def get_tensorboard():

    log_dir = os.path.join(

        "logs",

        datetime.now().strftime("%Y%m%d-%H%M%S")

    )

    return TensorBoard(

        log_dir=log_dir,

        histogram_freq=1

    )


# ==========================================================
# CSV Logger
# ==========================================================

def get_csv_logger():

    return CSVLogger(

        "reports/training_history.csv",

        append=False

    )


# ==========================================================
# Get All Callbacks
# ==========================================================

def get_callbacks():

    create_directories()

    callbacks = [

        get_early_stopping(),

        get_reduce_lr(),

        get_checkpoint(),

        get_tensorboard(),

        get_csv_logger()

    ]

    return callbacks


# ==========================================================
# Test
# ==========================================================

if __name__ == "__main__":

    callbacks = get_callbacks()

    print("\nCallbacks Loaded Successfully\n")

    for cb in callbacks:

        print(type(cb).__name__)