"""
==========================================================
Smart Insole DFU Risk Prediction
Train LSTM Model
==========================================================
"""

import os
import sys
import random
import joblib
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# ----------------------------------------------------------
# Project Root  (works regardless of CWD)
# ----------------------------------------------------------

ROOT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, os.path.dirname(__file__))  # allow same-package imports

from model import build_model
from callbacks import get_callbacks

from config import (
    SEQUENCE_DATA_PATH,
    EPOCHS,
    BATCH_SIZE
)

# Resolve paths to absolute
SEQUENCE_DATA_PATH = os.path.join(ROOT_DIR, SEQUENCE_DATA_PATH)

# ==========================================================
# Reproducibility
# ==========================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ==========================================================
# Load Dataset
# ==========================================================

def load_data():

    print("\nLoading sequence datasets...\n")

    X_train = np.load(
        os.path.join(SEQUENCE_DATA_PATH, "X_train.npy")
    )

    X_test = np.load(
        os.path.join(SEQUENCE_DATA_PATH, "X_test.npy")
    )

    y_train = np.load(
        os.path.join(SEQUENCE_DATA_PATH, "y_train.npy")
    )

    y_test = np.load(
        os.path.join(SEQUENCE_DATA_PATH, "y_test.npy")
    )

    print("Training Shape :", X_train.shape)
    print("Testing Shape  :", X_test.shape)

    return X_train, X_test, y_train, y_test


# ==========================================================
# Load Class Weights
# ==========================================================

def load_class_weights():

    path = "models/class_weights.pkl"

    if os.path.exists(path):

        print("\nLoading Class Weights...\n")

        return joblib.load(path)

    print("\nNo class weights found.")

    return None


# ==========================================================
# Train
# ==========================================================

def train_model():

    X_train, X_test, y_train, y_test = load_data()

    class_weights = load_class_weights()

    model = build_model(

        input_shape=(

            X_train.shape[1],

            X_train.shape[2]

        )

    )

    model.summary()

    history = model.fit(

        X_train,

        y_train,

        validation_split=0.2,

        epochs=1,

        batch_size=BATCH_SIZE,

        shuffle=True,

        callbacks=get_callbacks(),

        class_weight=class_weights,

        verbose=1

    )

    print("\nTraining Completed.\n")

    # -----------------------------------------------------

    print("Evaluating Model...\n")

    loss, accuracy = model.evaluate(

        X_test,

        y_test,

        verbose=1

    )

    print(f"\nTest Accuracy : {accuracy:.4f}")

    print(f"Test Loss     : {loss:.4f}")

    # -----------------------------------------------------

    y_pred = model.predict(

        X_test,

        verbose=0

    )

    y_pred = np.argmax(

        y_pred,

        axis=1

    )

    print("\nClassification Report\n")

    print(

        classification_report(

            y_test,

            y_pred,

            target_names=[

                "Low",

                "Medium",

                "High"

            ]

        )

    )

    # -----------------------------------------------------

    cm = confusion_matrix(

        y_test,

        y_pred

    )

    disp = ConfusionMatrixDisplay(

        confusion_matrix=cm,

        display_labels=[

            "Low",

            "Medium",

            "High"

        ]

    )

    disp.plot()

    plt.savefig(

        "reports/confusion_matrix.png",

        dpi=300,

        bbox_inches="tight"

    )

    plt.close()

    # -----------------------------------------------------

    plt.figure(figsize=(8,5))

    plt.plot(

        history.history["accuracy"],

        label="Train"

    )

    plt.plot(

        history.history["val_accuracy"],

        label="Validation"

    )

    plt.xlabel("Epoch")

    plt.ylabel("Accuracy")

    plt.legend()

    plt.grid(True)

    plt.savefig(

        "reports/accuracy.png",

        dpi=300,

        bbox_inches="tight"

    )

    plt.close()

    # -----------------------------------------------------

    plt.figure(figsize=(8,5))

    plt.plot(

        history.history["loss"],

        label="Train"

    )

    plt.plot(

        history.history["val_loss"],

        label="Validation"

    )

    plt.xlabel("Epoch")

    plt.ylabel("Loss")

    plt.legend()

    plt.grid(True)

    plt.savefig(

        "reports/loss.png",

        dpi=300,

        bbox_inches="tight"

    )

    plt.close()

    print("\nTraining graphs saved.")

    return model


# ==========================================================
# Main
# ==========================================================

def main():

    train_model()


if __name__ == "__main__":

    main()