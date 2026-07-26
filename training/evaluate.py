"""
==========================================================
Smart Insole DFU Risk Prediction
Model Evaluation

Generates and saves:
    reports/confusion_matrix.png
    reports/roc_curve.png
    reports/evaluation_report.txt
    reports/evaluation_summary.csv

Usage:
    python training/evaluate.py
==========================================================
"""

import os
import sys
from typing import Dict, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")          # headless — no display required
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score,
    roc_curve,
    auc,
)
from sklearn.preprocessing import label_binarize

import tensorflow as tf

# ----------------------------------------------------------
# Project Root  (works regardless of CWD)
# ----------------------------------------------------------

ROOT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, os.path.dirname(__file__))

from config import (
    SEQUENCE_DATA_PATH,
    MODEL_PATH,
)

# Resolve to absolute paths
SEQUENCE_DATA_PATH = os.path.join(ROOT_DIR, SEQUENCE_DATA_PATH)
MODEL_PATH         = os.path.join(ROOT_DIR, MODEL_PATH)
REPORTS_DIR        = os.path.join(ROOT_DIR, "reports")

# ----------------------------------------------------------
# Constants
# ----------------------------------------------------------

CLASS_NAMES  = ["Low", "Medium", "High"]
N_CLASSES    = len(CLASS_NAMES)
PALETTE      = ["#2ecc71", "#f39c12", "#e74c3c"]   # green / amber / red


# ==========================================================
# 1. Load Artifacts
# ==========================================================

def load_test_data() -> Tuple[np.ndarray, np.ndarray]:
    """Load X_test and y_test from data/sequences/."""
    print("\nLoading test data...")
    X_test = np.load(os.path.join(SEQUENCE_DATA_PATH, "X_test.npy"))
    y_test = np.load(os.path.join(SEQUENCE_DATA_PATH, "y_test.npy"))
    print(f"  X_test : {X_test.shape}")
    print(f"  y_test : {y_test.shape}")
    return X_test, y_test


def load_model() -> tf.keras.Model:
    """Load the saved Keras model."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Trained model not found at:\n  {MODEL_PATH}\n"
            "Run training/train_lstm.py first."
        )
    print(f"\nLoading model from:\n  {MODEL_PATH}")
    model = tf.keras.models.load_model(MODEL_PATH)
    print("  Model loaded successfully.")
    return model


# ==========================================================
# 2. Run Inference
# ==========================================================

def predict(
    model: tf.keras.Model,
    X_test: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run inference and return both hard predictions and class probabilities.

    Returns
    -------
    y_pred      : (n,)   — argmax class index
    y_prob      : (n, 3) — softmax probabilities
    """
    print("\nRunning inference...")
    y_prob = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_prob, axis=1)
    print("  Done.")
    return y_pred, y_prob


# ==========================================================
# 3. Core Metrics
# ==========================================================

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
) -> Dict:
    """
    Compute all evaluation metrics and return them as a dict.

    Metrics
    -------
    - Per-class precision, recall, F1
    - Per-class accuracy  (TP + TN) / N
    - Macro / weighted averages
    - Multiclass ROC-AUC  (One-vs-Rest, macro average)
    """
    report_dict = classification_report(
        y_true,
        y_pred,
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )

    # Per-class accuracy  = (TP + TN) / total
    per_class_acc = {}
    total = len(y_true)
    for i, name in enumerate(CLASS_NAMES):
        tp = np.sum((y_true == i) & (y_pred == i))
        tn = np.sum((y_true != i) & (y_pred != i))
        per_class_acc[name] = (tp + tn) / total

    # Multiclass ROC-AUC (One-vs-Rest)
    y_bin = label_binarize(y_true, classes=[0, 1, 2])
    try:
        roc_auc_macro = roc_auc_score(
            y_bin, y_prob, multi_class="ovr", average="macro"
        )
        roc_auc_per_class = {}
        for i, name in enumerate(CLASS_NAMES):
            roc_auc_per_class[name] = roc_auc_score(y_bin[:, i], y_prob[:, i])
    except ValueError as exc:
        print(f"  [!] ROC-AUC skipped: {exc}")
        roc_auc_macro = float("nan")
        roc_auc_per_class = {n: float("nan") for n in CLASS_NAMES}

    return {
        "report"          : report_dict,
        "per_class_acc"   : per_class_acc,
        "roc_auc_macro"   : roc_auc_macro,
        "roc_auc_per_class": roc_auc_per_class,
        "y_bin"           : y_bin,
        "y_prob"          : y_prob,
    }


# ==========================================================
# 4. Print Console Summary
# ==========================================================

def print_summary(metrics: Dict, y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """Print a formatted evaluation summary to the console."""
    rep = metrics["report"]

    print("\n" + "=" * 60)
    print("  DFU LSTM — Evaluation Report")
    print("=" * 60)

    # Overall
    overall_acc = rep["accuracy"]
    print(f"\n  Overall Accuracy  : {overall_acc:.4f}  ({overall_acc*100:.2f}%)")
    print(f"  Macro  ROC-AUC    : {metrics['roc_auc_macro']:.4f}")

    # Per-class table
    print("\n" + "-" * 60)
    print(f"  {'Class':<10} {'Precision':>10} {'Recall':>8} {'F1':>8} {'Acc':>8} {'AUC':>8}")
    print("-" * 60)
    for name in CLASS_NAMES:
        p   = rep[name]["precision"]
        r   = rep[name]["recall"]
        f1  = rep[name]["f1-score"]
        acc = metrics["per_class_acc"][name]
        auc_ = metrics["roc_auc_per_class"][name]
        print(f"  {name:<10} {p:>10.4f} {r:>8.4f} {f1:>8.4f} {acc:>8.4f} {auc_:>8.4f}")
    print("-" * 60)

    # Macro / weighted
    for avg in ("macro avg", "weighted avg"):
        p  = rep[avg]["precision"]
        r  = rep[avg]["recall"]
        f1 = rep[avg]["f1-score"]
        print(f"  {avg:<10} {p:>10.4f} {r:>8.4f} {f1:>8.4f}")

    print("=" * 60)


# ==========================================================
# 5. Confusion Matrix Plot
# ==========================================================

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> str:
    """Plot and save a styled confusion matrix. Returns save path."""
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(7, 6))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#1a1a2e")

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=CLASS_NAMES,
    )
    disp.plot(
        ax=ax,
        colorbar=True,
        cmap="Blues",
    )

    # Styling
    ax.set_title(
        "DFU Risk — Confusion Matrix",
        color="white", fontsize=14, fontweight="bold", pad=14,
    )
    ax.set_xlabel("Predicted Label", color="white", fontsize=11)
    ax.set_ylabel("True Label",      color="white", fontsize=11)
    ax.tick_params(colors="white")

    for text in disp.text_.ravel():
        text.set_color("white")
        text.set_fontsize(13)
        text.set_fontweight("bold")

    disp.im_.colorbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(disp.im_.colorbar.ax.yaxis.get_ticklabels(), color="white")

    plt.tight_layout()
    save_path = os.path.join(REPORTS_DIR, "confusion_matrix.png")
    plt.savefig(save_path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved -> {save_path}")
    return save_path


# ==========================================================
# 6. ROC Curve Plot  (One-vs-Rest per class)
# ==========================================================

def plot_roc_curves(
    y_bin: np.ndarray,
    y_prob: np.ndarray,
) -> str:
    """Plot per-class OvR ROC curves on a dark background. Returns save path."""
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#16213e")

    for i, (name, color) in enumerate(zip(CLASS_NAMES, PALETTE)):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
        roc_auc_val = auc(fpr, tpr)
        ax.plot(
            fpr, tpr,
            color=color,
            lw=2.5,
            label=f"{name}  (AUC = {roc_auc_val:.3f})",
        )

    # Diagonal reference
    ax.plot([0, 1], [0, 1], "w--", lw=1, alpha=0.4, label="Random")

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.02])
    ax.set_xlabel("False Positive Rate", color="white", fontsize=11)
    ax.set_ylabel("True Positive Rate",  color="white", fontsize=11)
    ax.set_title(
        "ROC Curves — One-vs-Rest (per class)",
        color="white", fontsize=13, fontweight="bold",
    )
    ax.tick_params(colors="white")
    ax.spines[:].set_color("#444466")

    legend = ax.legend(
        loc="lower right",
        facecolor="#0f3460",
        edgecolor="#444466",
        labelcolor="white",
        fontsize=10,
    )

    plt.tight_layout()
    save_path = os.path.join(REPORTS_DIR, "roc_curve.png")
    plt.savefig(save_path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved -> {save_path}")
    return save_path


# ==========================================================
# 7. Per-Class Metrics Bar Chart
# ==========================================================

def plot_per_class_metrics(metrics: Dict) -> str:
    """Bar chart comparing Precision / Recall / F1 / Accuracy per class."""
    rep = metrics["report"]

    x      = np.arange(N_CLASSES)
    width  = 0.20
    bars   = {
        "Precision": [rep[n]["precision"]    for n in CLASS_NAMES],
        "Recall"   : [rep[n]["recall"]       for n in CLASS_NAMES],
        "F1-Score" : [rep[n]["f1-score"]     for n in CLASS_NAMES],
        "Accuracy" : [metrics["per_class_acc"][n] for n in CLASS_NAMES],
    }
    bar_colors = ["#3498db", "#2ecc71", "#e67e22", "#9b59b6"]

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#16213e")

    for idx, (label, values) in enumerate(bars.items()):
        offset = (idx - 1.5) * width
        rects  = ax.bar(x + offset, values, width, label=label, color=bar_colors[idx], alpha=0.88)
        for rect in rects:
            h = rect.get_height()
            ax.annotate(
                f"{h:.2f}",
                xy=(rect.get_x() + rect.get_width() / 2, h),
                xytext=(0, 3), textcoords="offset points",
                ha="center", va="bottom", fontsize=7.5, color="white",
            )

    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_NAMES, color="white", fontsize=11)
    ax.set_yticks(np.arange(0, 1.1, 0.1))
    ax.set_ylim(0, 1.15)
    ax.tick_params(axis="y", colors="white")
    ax.set_xlabel("Risk Class",    color="white", fontsize=11)
    ax.set_ylabel("Score",         color="white", fontsize=11)
    ax.set_title(
        "Per-Class Metrics: Precision · Recall · F1 · Accuracy",
        color="white", fontsize=12, fontweight="bold",
    )
    ax.spines[:].set_color("#444466")
    ax.grid(axis="y", color="#444466", linestyle="--", alpha=0.4)

    legend = ax.legend(
        facecolor="#0f3460",
        edgecolor="#444466",
        labelcolor="white",
        fontsize=9,
    )

    plt.tight_layout()
    save_path = os.path.join(REPORTS_DIR, "per_class_metrics.png")
    plt.savefig(save_path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved -> {save_path}")
    return save_path


# ==========================================================
# 8. Save Text Report
# ==========================================================

def save_text_report(
    metrics: Dict,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> str:
    """Write a complete plain-text evaluation report."""
    rep     = metrics["report"]
    cr_text = classification_report(
        y_true, y_pred,
        target_names=CLASS_NAMES,
        zero_division=0,
    )

    lines = [
        "=" * 60,
        "  Smart Insole DFU Risk Prediction",
        "  Model Evaluation Report",
        "=" * 60,
        "",
        f"  Overall Accuracy   : {rep['accuracy']:.4f}",
        f"  Macro  ROC-AUC     : {metrics['roc_auc_macro']:.4f}",
        "",
        "-" * 60,
        "  Classification Report",
        "-" * 60,
        "",
        cr_text,
        "",
        "-" * 60,
        "  Per-Class Accuracy  (TP + TN) / N",
        "-" * 60,
    ]
    for name, acc in metrics["per_class_acc"].items():
        lines.append(f"  {name:<10} : {acc:.4f}  ({acc*100:.2f}%)")

    lines += [
        "",
        "-" * 60,
        "  ROC-AUC  (One-vs-Rest)",
        "-" * 60,
    ]
    for name, val in metrics["roc_auc_per_class"].items():
        lines.append(f"  {name:<10} : {val:.4f}")

    lines += ["", "=" * 60]

    save_path = os.path.join(REPORTS_DIR, "evaluation_report.txt")
    with open(save_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  Saved -> {save_path}")
    return save_path


# ==========================================================
# 9. Save CSV Summary
# ==========================================================

def save_csv_summary(metrics: Dict) -> str:
    """Save per-class metrics to a tidy CSV for downstream analysis."""
    import csv

    rep  = metrics["report"]
    rows = [["class", "precision", "recall", "f1_score", "per_class_accuracy", "roc_auc"]]

    for name in CLASS_NAMES:
        rows.append([
            name,
            f"{rep[name]['precision']:.4f}",
            f"{rep[name]['recall']:.4f}",
            f"{rep[name]['f1-score']:.4f}",
            f"{metrics['per_class_acc'][name]:.4f}",
            f"{metrics['roc_auc_per_class'][name]:.4f}",
        ])

    rows.append([
        "macro avg",
        f"{rep['macro avg']['precision']:.4f}",
        f"{rep['macro avg']['recall']:.4f}",
        f"{rep['macro avg']['f1-score']:.4f}",
        "",
        f"{metrics['roc_auc_macro']:.4f}",
    ])
    rows.append([
        "weighted avg",
        f"{rep['weighted avg']['precision']:.4f}",
        f"{rep['weighted avg']['recall']:.4f}",
        f"{rep['weighted avg']['f1-score']:.4f}",
        f"{rep['accuracy']:.4f}",
        "",
    ])

    save_path = os.path.join(REPORTS_DIR, "evaluation_summary.csv")
    with open(save_path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)
    print(f"  Saved -> {save_path}")
    return save_path


# ==========================================================
# Main
# ==========================================================

def main() -> None:
    """Full evaluation pipeline."""
    print("\n" + "=" * 60)
    print("  Smart Insole DFU — Model Evaluation")
    print("=" * 60)

    os.makedirs(REPORTS_DIR, exist_ok=True)

    # Load
    X_test, y_test = load_test_data()
    model          = load_model()

    # Predict
    y_pred, y_prob = predict(model, X_test)

    # Metrics
    print("\nComputing metrics...")
    metrics = compute_metrics(y_test, y_pred, y_prob)

    # Console summary
    print_summary(metrics, y_test, y_pred)

    # Plots
    print("\nGenerating plots...")
    plot_confusion_matrix(y_test, y_pred)
    plot_roc_curves(metrics["y_bin"], metrics["y_prob"])
    plot_per_class_metrics(metrics)

    # Reports
    print("\nSaving reports...")
    save_text_report(metrics, y_test, y_pred)
    save_csv_summary(metrics)

    print("\n" + "=" * 60)
    print("  Evaluation Complete")
    print(f"  All outputs saved to: {REPORTS_DIR}")
    print("=" * 60 + "\n")


# ==========================================================
# Entry Point
# ==========================================================

if __name__ == "__main__":
    main()
