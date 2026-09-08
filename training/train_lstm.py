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
    ConfusionMatrixDisplay,
    precision_recall_fscore_support,
    roc_auc_score
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
    BATCH_SIZE,
    MODEL_PATH
)

# Resolve paths to absolute
SEQUENCE_DATA_PATH = os.path.join(ROOT_DIR, SEQUENCE_DATA_PATH)
MODEL_PATH = os.path.join(ROOT_DIR, MODEL_PATH)

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
    X_train = np.load(os.path.join(SEQUENCE_DATA_PATH, "X_train.npy"))
    X_val = np.load(os.path.join(SEQUENCE_DATA_PATH, "X_val.npy"))
    X_test = np.load(os.path.join(SEQUENCE_DATA_PATH, "X_test.npy"))
    
    y_train = np.load(os.path.join(SEQUENCE_DATA_PATH, "y_train.npy"))
    y_val = np.load(os.path.join(SEQUENCE_DATA_PATH, "y_val.npy"))
    y_test = np.load(os.path.join(SEQUENCE_DATA_PATH, "y_test.npy"))

    # Convert sparse integer labels to one-hot for categorical_crossentropy and Keras metrics
    y_train = tf.keras.utils.to_categorical(y_train, num_classes=3)
    y_val = tf.keras.utils.to_categorical(y_val, num_classes=3)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes=3)

    print("Training Shape   :", X_train.shape)
    print("Validation Shape :", X_val.shape)
    print("Testing Shape    :", X_test.shape)

    return X_train, X_val, X_test, y_train, y_val, y_test


# ==========================================================
# Load Class Weights
# ==========================================================

def load_class_weights():
    path = os.path.join(ROOT_DIR, "models", "class_weights.pkl")
    if os.path.exists(path):
        print("\nLoading Class Weights...\n")
        return joblib.load(path)
    
    print("\nNo class weights found. Continuing normally.")
    return None


# ==========================================================
# Train
# ==========================================================

def train_model():
    X_train, X_val, X_test, y_train, y_val, y_test = load_data()

    class_weights = load_class_weights()

    model = build_model(
        input_shape=(
            X_train.shape[1],
            X_train.shape[2]
        )
    )

    model.summary()

    # -----------------------------------------------------
    # Fit Model
    # -----------------------------------------------------

    # Task 1: Use epochs=EPOCHS
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        shuffle=True,
        callbacks=get_callbacks(),
        class_weight=class_weights,
        verbose=1
    )

    print("\nTraining Completed.\n")

    # -----------------------------------------------------
    # Evaluate Model
    # -----------------------------------------------------

    # Task 8: Reload the BEST checkpoint before evaluation
    if os.path.exists(MODEL_PATH):
        print("Reloading the best model before evaluation...\n")
        model = tf.keras.models.load_model(MODEL_PATH)
    
    print("Evaluating Model...\n")
    eval_results = model.evaluate(X_test, y_test, verbose=1)
    
    # Unpack based on metrics
    loss = eval_results[0]
    accuracy = eval_results[1]
    precision_keras = eval_results[2] if len(eval_results) > 2 else 0.0
    recall_keras = eval_results[3] if len(eval_results) > 3 else 0.0

    # -----------------------------------------------------
    # Predictions
    # -----------------------------------------------------

    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    if len(y_test.shape) > 1 and y_test.shape[1] > 1:
        y_test_labels = np.argmax(y_test, axis=1)
    else:
        y_test_labels = y_test

    # Task 9: Generate Precision, Recall, Macro F1, ROC-AUC
    precision_macro, recall_macro, f1, _ = precision_recall_fscore_support(
        y_test_labels, y_pred, average="macro"
    )
    
    roc_auc = "N/A"
    try:
        if len(y_test.shape) > 1 and y_test.shape[1] > 1:
            # y_test is one-hot
            roc_auc = roc_auc_score(y_test, y_pred_probs, multi_class='ovr')
        else:
            # y_test is integer encoded, create one-hot representation for ROC AUC
            y_test_oh = tf.keras.utils.to_categorical(y_test_labels, num_classes=3)
            roc_auc = roc_auc_score(y_test_oh, y_pred_probs, multi_class='ovr')
    except Exception as e:
        roc_auc = f"Error computing: {e}"
        
    print("\n--- Model Performance Metrics ---")
    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Loss      : {loss:.4f}")
    print(f"Precision : {precision_macro:.4f}")
    print(f"Recall    : {recall_macro:.4f}")
    print(f"Macro F1  : {f1:.4f}")
    if isinstance(roc_auc, float):
        print(f"ROC-AUC   : {roc_auc:.4f}")
    else:
        print(f"ROC-AUC   : {roc_auc}")
    print("---------------------------------")

    target_names = ["Low", "Medium", "High"]

    print("\nClassification Report\n")
    print(
        classification_report(
            y_test_labels,
            y_pred,
            target_names=target_names
        )
    )

    # -----------------------------------------------------
    # Save Outputs
    # -----------------------------------------------------
    
    reports_dir = os.path.join(ROOT_DIR, "reports")
    os.makedirs(reports_dir, exist_ok=True)
    
    # Task 10: Save confusion_matrix.png
    cm = confusion_matrix(y_test_labels, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.savefig(
        os.path.join(reports_dir, "confusion_matrix.png"),
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

    # Task 10: Save accuracy.png
    plt.figure(figsize=(8, 5))
    plt.plot(history.history["accuracy"], label="Train Accuracy")
    if "val_accuracy" in history.history:
        plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.title("Model Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.savefig(
        os.path.join(reports_dir, "accuracy.png"),
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

    # Task 10: Save loss.png
    plt.figure(figsize=(8, 5))
    plt.plot(history.history["loss"], label="Train Loss")
    if "val_loss" in history.history:
        plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title("Model Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(
        os.path.join(reports_dir, "loss.png"),
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

    print("\nAll tasks completed: Training graphs and confusion matrix saved in reports/")

    return model


# ==========================================================
# Main
# ==========================================================

def main():
    train_model()


if __name__ == "__main__":
    main()