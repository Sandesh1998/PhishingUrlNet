"""
PhishNet - Visualiser
Sandesh Poudel | B01818884 | UWS MSc Cybersecurity

Produces all plots required by the project specification:
  - ROC curves (multi-model overlay)
  - Precision-Recall curves
  - Confusion matrices
  - Training history (loss/accuracy curves)
  - Feature importance (RF)
  - Latency comparison bar chart
  - Accuracy vs Latency scatter (replicating Figure 1 in interim report)
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from typing import Dict, List, Optional, Tuple

from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, auc

from src.utils.logger import get_logger

logger = get_logger(__name__)

# ── Consistent colour palette across all plots ───────────────────────────────
PALETTE = {
    "CNN":       "#2196F3",   # blue
    "CNN_TCN":   "#4CAF50",   # green
    "RF":        "#FF9800",   # orange
    "SVM":       "#9C27B0",   # purple
    "DNN":       "#F44336",   # red
    "default":   "#607D8B",
}

sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams.update({"figure.dpi": 150, "savefig.bbox": "tight"})


def _savefig(fig, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Plot saved → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# 1. ROC Curves
# ─────────────────────────────────────────────────────────────────────────────

def plot_roc_curves(
    roc_data: Dict[str, Tuple[np.ndarray, np.ndarray]],
    output_path: str,
    title: str = "ROC Curves – All Models",
) -> None:
    """
    Overlay ROC curves for multiple models.

    Args:
        roc_data: {model_name: (fpr_array, tpr_array, auc_score)}
        output_path: File save path (.png or .pdf)
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    for model_name, (fpr, tpr, auc_score) in roc_data.items():
        colour = PALETTE.get(model_name, PALETTE["default"])
        ax.plot(fpr, tpr, label=f"{model_name}  (AUC = {auc_score:.4f})",
                color=colour, lw=2)

    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random (AUC = 0.50)")
    ax.fill_between([0, 1], [0, 1], alpha=0.05, color="grey")

    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.01])

    _savefig(fig, output_path)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Precision-Recall Curves
# ─────────────────────────────────────────────────────────────────────────────

def plot_pr_curves(
    pr_data: Dict[str, Tuple[np.ndarray, np.ndarray, float]],
    output_path: str,
    title: str = "Precision-Recall Curves – All Models",
) -> None:
    """
    Overlay PR curves for multiple models.

    Args:
        pr_data: {model_name: (precision_array, recall_array, ap_score)}
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    for model_name, (precision, recall, ap) in pr_data.items():
        colour = PALETTE.get(model_name, PALETTE["default"])
        ax.plot(recall, precision, label=f"{model_name}  (AP = {ap:.4f})",
                color=colour, lw=2)

    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(loc="upper right", fontsize=10)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.01])

    _savefig(fig, output_path)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Confusion Matrix
# ─────────────────────────────────────────────────────────────────────────────

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    output_path: str,
    labels: List[str] = None,
) -> None:
    """Annotated heatmap confusion matrix for a single model."""
    if labels is None:
        labels = ["Legitimate", "Phishing"]

    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm_norm,
        annot=np.array([[f"{cm[i,j]}\n({cm_norm[i,j]:.1%})"
                         for j in range(cm.shape[1])]
                        for i in range(cm.shape[0])]),
        fmt="",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        linewidths=0.5,
        ax=ax,
        cbar_kws={"label": "Normalised Rate"},
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    ax.set_title(f"Confusion Matrix – {model_name}", fontsize=13, fontweight="bold")

    _savefig(fig, output_path)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Training History
# ─────────────────────────────────────────────────────────────────────────────

def plot_training_history(
    history: Dict[str, List[float]],
    model_name: str,
    output_path: str,
) -> None:
    """
    4-panel training curve: Loss, Accuracy, AUC, Learning Rate.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    epochs = range(1, len(history["loss"]) + 1)

    # Loss
    axes[0].plot(epochs, history["loss"], label="Train Loss",
                 color="#2196F3", lw=2)
    axes[0].plot(epochs, history["val_loss"], label="Val Loss",
                 color="#F44336", lw=2, linestyle="--")
    axes[0].set_title("Loss", fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Binary Cross-Entropy")
    axes[0].legend()

    # Accuracy
    acc_key = "accuracy" if "accuracy" in history else "acc"
    val_acc_key = "val_accuracy" if "val_accuracy" in history else "val_acc"
    axes[1].plot(epochs, history[acc_key], label="Train Acc",
                 color="#2196F3", lw=2)
    axes[1].plot(epochs, history[val_acc_key], label="Val Acc",
                 color="#F44336", lw=2, linestyle="--")
    axes[1].set_title("Accuracy", fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()

    fig.suptitle(f"Training History – {model_name}", fontsize=14, fontweight="bold")
    _savefig(fig, output_path)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Model Comparison Bar Chart
# ─────────────────────────────────────────────────────────────────────────────

def plot_model_comparison(
    metrics_df: pd.DataFrame,
    output_path: str,
    metrics: List[str] = None,
) -> None:
    """
    Grouped bar chart comparing all models across key classification metrics.

    Args:
        metrics_df: DataFrame with models as index and metrics as columns
        output_path: Save path
        metrics: List of metric column names to include
    """
    if metrics is None:
        metrics = ["accuracy", "precision", "recall", "f1", "auc_roc"]

    df_plot = metrics_df[metrics].copy()
    colours = [PALETTE.get(idx, PALETTE["default"]) for idx in df_plot.index]

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(metrics))
    n_models = len(df_plot)
    width = 0.8 / n_models

    for i, (model_name, row) in enumerate(df_plot.iterrows()):
        offset = (i - n_models / 2 + 0.5) * width
        bars = ax.bar(x + offset, row[metrics].values, width=width,
                      label=model_name, color=colours[i], edgecolor="white")
        # Value labels on bars
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005,
                    f"{h:.3f}", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("_", " ").upper() for m in metrics], fontsize=11)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Model Performance Comparison", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.axhline(y=0.97, color="red", linestyle=":", lw=1.5, label="Target 97%")

    _savefig(fig, output_path)


# ─────────────────────────────────────────────────────────────────────────────
# 6. Accuracy vs Latency Scatter
# ─────────────────────────────────────────────────────────────────────────────

def plot_accuracy_vs_latency(
    metrics_df: pd.DataFrame,
    output_path: str,
) -> None:
    """
    Scatter plot of accuracy vs inference latency (p50).
    Replicates Figure 1 concept from interim report.
    Bubble size = AUC-ROC score.
    """
    fig, ax = plt.subplots(figsize=(9, 6))

    for model_name, row in metrics_df.iterrows():
        colour = PALETTE.get(model_name, PALETTE["default"])
        acc = row.get("accuracy", 0)
        lat = row.get("latency_p50_ms", 0)
        auc_score = row.get("auc_roc", 0.5)

        ax.scatter(lat, acc, s=auc_score * 500, color=colour, alpha=0.85,
                   edgecolors="white", linewidths=1.5, zorder=3)
        ax.annotate(
            model_name,
            xy=(lat, acc),
            xytext=(8, 4),
            textcoords="offset points",
            fontsize=10,
            color=colour,
            fontweight="bold",
        )

    # Target zones
    ax.axhline(y=0.97, color="green", linestyle="--", lw=1.2,
               label="Accuracy target (97%)")
    ax.axvline(x=100, color="orange", linestyle="--", lw=1.2,
               label="Latency target (100 ms)")
    ax.fill_between([0, 100], [0.97, 0.97], [1.01, 1.01],
                    alpha=0.08, color="green", label="Target zone")

    ax.set_xlabel("Inference Latency p50 (ms)", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("Accuracy vs Inference Latency\n(bubble size = AUC-ROC)",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, loc="lower left")
    ax.set_ylim(0.85, 1.01)

    _savefig(fig, output_path)


# ─────────────────────────────────────────────────────────────────────────────
# 7. Feature Importance (Random Forest)
# ─────────────────────────────────────────────────────────────────────────────

def plot_feature_importance(
    feature_names: List[str],
    importances: np.ndarray,
    output_path: str,
    top_n: int = 20,
) -> None:
    """Horizontal bar chart of top-N Random Forest feature importances."""
    indices = np.argsort(importances)[-top_n:][::-1]
    top_names = [feature_names[i] for i in indices]
    top_vals = importances[indices]

    fig, ax = plt.subplots(figsize=(10, 7))
    colours = plt.cm.RdYlGn(np.linspace(0.2, 0.9, top_n))
    bars = ax.barh(range(top_n), top_vals[::-1], color=colours, edgecolor="white")
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(top_names[::-1], fontsize=10)
    ax.set_xlabel("Feature Importance (Gini)", fontsize=12)
    ax.set_title(f"Top {top_n} Feature Importances – Random Forest",
                 fontsize=13, fontweight="bold")

    _savefig(fig, output_path)
