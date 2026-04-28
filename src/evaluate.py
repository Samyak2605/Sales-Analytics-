"""
evaluate.py
============
Model evaluation utilities for the Sales Analytics Engine.

This module computes and reports classification metrics:
  - Accuracy
  - Precision
  - Recall
  - F1-Score (PRIMARY metric)
  - Confusion Matrix
  - Win Probability Analysis (Classification Confidence)

Evaluation Philosophy
---------------------

### Why F1-Score as the Primary Metric?

In binary classification with class imbalance, **accuracy alone is misleading**.

Consider a dataset with 90% positive class: a model that always predicts
"positive" achieves 90% accuracy but is completely useless for identifying
negatives. This is known as the **accuracy paradox**.

**F1-Score** is the harmonic mean of Precision and Recall:

  F1 = 2 × (Precision × Recall) / (Precision + Recall)

This metric penalizes models that sacrifice either precision or recall:
  - High precision but low recall → many missed positives (conservative model)
  - High recall but low precision → many false alarms (aggressive model)
  - High F1 → balanced performance on both dimensions

For sales analytics:
  - **Precision** answers: "Of deals we predicted as WON, how many actually won?"
    (avoiding false optimism in pipeline forecasts)
  - **Recall** answers: "Of deals that actually WON, how many did we predict?"
    (avoiding missed opportunities)

Both matter for sales operations, making F1 the appropriate primary metric.

### Confusion Matrix Interpretation

             Predicted
              0    1
Actual  0  [ TN | FP ]    TN = True Negative (correctly predicted LOST)
        1  [ FN | TP ]    FP = False Positive (predicted WON, actually LOST)
                          FN = False Negative (predicted LOST, actually WON)
                          TP = True Positive (correctly predicted WON)
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

# ─── Output Configuration ──────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
CONFUSION_MATRIX_PATH = os.path.join(OUTPUT_DIR, "confusion_matrix.png")


def evaluate_model(y_true, y_pred, y_prob=None) -> dict:
    """
    Compute all evaluation metrics.

    Parameters
    ----------
    y_true : array-like
        True labels.
    y_pred : array-like
        Predicted labels.
    y_prob : array-like, optional
        Predicted probabilities for the positive class.

    Returns
    -------
    dict
        Dictionary containing:
        - 'accuracy': Overall accuracy
        - 'precision': Precision for positive class
        - 'recall': Recall for positive class
        - 'f1_score': F1-score for positive class (PRIMARY)
        - 'confusion_matrix': 2×2 numpy array
        - 'classification_report': Detailed text report
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
        "classification_report": classification_report(
            y_true, y_pred,
            target_names=["LOST (0)", "WON (1)"],
            zero_division=0,
        ),
    }

    return metrics


def print_evaluation_report(metrics: dict):
    """
    Print a formatted evaluation report to the console.

    Parameters
    ----------
    metrics : dict
        Output from evaluate_model().
    """
    print("\n" + "=" * 60)
    print("  MODEL EVALUATION REPORT")
    print("=" * 60)
    print(f"\n  Accuracy:   {metrics['accuracy']:.4f}")
    print(f"  Precision:  {metrics['precision']:.4f}")
    print(f"  Recall:     {metrics['recall']:.4f}")
    print(f"  F1-Score:   {metrics['f1_score']:.4f}  ← PRIMARY METRIC")

    print(f"\n  Confusion Matrix:")
    cm = metrics["confusion_matrix"]
    print(f"                Predicted")
    print(f"               LOST  WON")
    print(f"  Actual LOST  [{cm[0][0]:4d}  {cm[0][1]:4d}]")
    print(f"  Actual WON   [{cm[1][0]:4d}  {cm[1][1]:4d}]")

    print(f"\n  Detailed Classification Report:")
    print(metrics["classification_report"])
    print("=" * 60)


def plot_confusion_matrix(metrics: dict, save_path: str = CONFUSION_MATRIX_PATH):
    """
    Generate and save a confusion matrix heatmap.

    Parameters
    ----------
    metrics : dict
        Output from evaluate_model().
    save_path : str
        File path to save the plot.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    cm = metrics["confusion_matrix"]

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["LOST (0)", "WON (1)"],
        yticklabels=["LOST (0)", "WON (1)"],
        ax=ax,
        linewidths=1,
        linecolor="gray",
        annot_kws={"size": 16, "weight": "bold"},
    )
    ax.set_xlabel("Predicted Label", fontsize=13, fontweight="bold")
    ax.set_ylabel("True Label", fontsize=13, fontweight="bold")
    ax.set_title("Confusion Matrix — Sales Deal Prediction", fontsize=15,
                 fontweight="bold", pad=15)

    # Add metric annotations
    accuracy = metrics["accuracy"]
    f1 = metrics["f1_score"]
    fig.text(0.5, -0.02, f"Accuracy: {accuracy:.4f}  |  F1-Score: {f1:.4f}",
             ha="center", fontsize=12, style="italic", color="gray")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n📊 Confusion matrix saved to: {save_path}")


def display_win_probabilities(y_prob, n_samples=5):
    """
    Display win probabilities for a subset of samples to match the
    'Win Probability' output shown in the architecture diagram.

    Parameters
    ----------
    y_prob : array-like
        Predicted probabilities for the WON class.
    n_samples : int
        Number of samples to display.
    """
    print("\n" + "=" * 60)
    print("  WIN PROBABIITY ANALYSIS (Sample Predictions)")
    print("=" * 60)
    
    indices = np.random.choice(len(y_prob), size=min(n_samples, len(y_prob)), replace=False)
    for idx in indices:
        prob = y_prob[idx]
        confidence = "HIGH LIKELIHOOD" if prob > 0.8 else "MODERATE" if prob > 0.5 else "LOW"
        print(f"  Deal #{idx:03d} | Win Probability: {prob*100:5.1f}% | Confidence: {confidence}")
    print("=" * 60)


def generate_full_report(y_true, y_pred, y_prob=None):
    """
    Run the complete evaluation pipeline: compute metrics, print report,
    and save the confusion matrix plot.

    Parameters
    ----------
    y_true : array-like
        True labels.
    y_pred : array-like
        Predicted labels.
    y_prob : array-like, optional
        Predicted probabilities.

    Returns
    -------
    dict
        Evaluation metrics dictionary.
    """
    metrics = evaluate_model(y_true, y_pred, y_prob)
    print_evaluation_report(metrics)
    plot_confusion_matrix(metrics)
    return metrics
