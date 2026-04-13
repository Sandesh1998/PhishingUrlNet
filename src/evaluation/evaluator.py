"""
PhishNet - Model Evaluator
Sandesh Poudel | B01818884 | UWS MSc Cybersecurity

Computes the full evaluation suite required by the project specification:

Classification Metrics:
  Accuracy, Precision, Recall, F1-Score, AUC-ROC, AUC-PR, FPR, FNR

Deployment Metrics:
  Inference latency (p50, p95), Throughput (URLs/sec), Memory usage

Statistical Tests:
  McNemar test for pairwise model comparison significance (p < 0.05)

Cross-Dataset Validation:
  Train on PhishTank/UNB → test on UCI/Alexa (generalisation)
"""
import time
import os
import tracemalloc
import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Optional, Any
from scipy.stats import chi2_contingency

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
    confusion_matrix, classification_report,
    roc_curve, precision_recall_curve,
)
from sklearn.model_selection import StratifiedKFold

import tensorflow as tf

from src.utils.logger import get_logger
from src.utils.helpers import save_metrics_to_excel

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Core Metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    model_name: str = "model",
) -> Dict[str, float]:
    """
    Compute the full set of classification metrics.

    Args:
        y_true: Ground truth binary labels
        y_pred: Binary predictions (threshold 0.5)
        y_prob: Predicted probabilities for positive class
        model_name: Name for logging

    Returns:
        Dictionary of metric_name -> float
    """
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    metrics = {
        "accuracy":   accuracy_score(y_true, y_pred),
        "precision":  precision_score(y_true, y_pred, zero_division=0),
        "recall":     recall_score(y_true, y_pred, zero_division=0),
        "f1":         f1_score(y_true, y_pred, zero_division=0),
        "auc_roc":    roc_auc_score(y_true, y_prob),
        "auc_pr":     average_precision_score(y_true, y_prob),
        "fpr":        fp / (fp + tn + 1e-9),     # False Positive Rate
        "fnr":        fn / (fn + tp + 1e-9),     # False Negative Rate
        "tp": int(tp), "fp": int(fp),
        "tn": int(tn), "fn": int(fn),
    }

    logger.info(
        f"[{model_name}] Acc={metrics['accuracy']:.4f} | "
        f"F1={metrics['f1']:.4f} | AUC={metrics['auc_roc']:.4f} | "
        f"FPR={metrics['fpr']:.4f}"
    )
    return metrics


def get_roc_curve_data(
    y_true: np.ndarray, y_prob: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (fpr_array, tpr_array, thresholds) for ROC curve plotting."""
    return roc_curve(y_true, y_prob)


def get_pr_curve_data(
    y_true: np.ndarray, y_prob: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (precision_array, recall_array, thresholds) for PR curve plotting."""
    return precision_recall_curve(y_true, y_prob)


# ─────────────────────────────────────────────────────────────────────────────
# Latency Benchmarking
# ─────────────────────────────────────────────────────────────────────────────

def benchmark_inference_latency(
    model,
    X_sample: np.ndarray,
    warmup_runs: int = 10,
    benchmark_runs: int = 100,
    model_type: str = "keras",
) -> Dict[str, float]:
    """
    Measure per-URL inference latency in milliseconds.

    Performs warmup runs (discarded) then benchmark_runs timed predictions.
    Reports p50 (median) and p95 percentile latencies.

    Args:
        model: Keras model or sklearn estimator
        X_sample: Single URL input (shape (1, ...) )
        warmup_runs: Number of discarded warm-up predictions
        benchmark_runs: Number of timed predictions
        model_type: 'keras' or 'sklearn'

    Returns:
        Dict with latency_p50_ms, latency_p95_ms, throughput_urls_per_sec
    """
    # Warmup
    for _ in range(warmup_runs):
        if model_type == "keras":
            model.predict(X_sample, verbose=0)
        else:
            model.predict_proba(X_sample)

    # Benchmark
    latencies = []
    for _ in range(benchmark_runs):
        t0 = time.perf_counter()
        if model_type == "keras":
            model.predict(X_sample, verbose=0)
        else:
            model.predict_proba(X_sample)
        latencies.append((time.perf_counter() - t0) * 1000)  # ms

    arr = np.array(latencies)
    p50 = float(np.percentile(arr, 50))
    p95 = float(np.percentile(arr, 95))
    throughput = 1000.0 / p50  # URLs/sec at p50 latency

    logger.info(
        f"Latency → p50: {p50:.2f}ms | p95: {p95:.2f}ms | "
        f"throughput: {throughput:.0f} URLs/sec"
    )
    return {
        "latency_p50_ms": p50,
        "latency_p95_ms": p95,
        "throughput_urls_per_sec": throughput,
    }


def measure_memory_usage(
    model,
    X_sample: np.ndarray,
    model_type: str = "keras",
) -> Dict[str, float]:
    """
    Measure peak memory usage during inference using tracemalloc.

    Returns:
        Dict with peak_memory_mb
    """
    tracemalloc.start()
    if model_type == "keras":
        model.predict(X_sample, verbose=0)
    else:
        model.predict_proba(X_sample)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return {"peak_memory_mb": peak / 1024 / 1024}


# ─────────────────────────────────────────────────────────────────────────────
# Cross-Validation
# ─────────────────────────────────────────────────────────────────────────────

def cross_validate_sklearn(
    model,
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    model_name: str = "model",
) -> Dict[str, float]:
    """
    Stratified k-fold cross validation for sklearn models.
    Returns mean ± std of accuracy, F1, AUC-ROC.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    accs, f1s, aucs = [], [], []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        import copy
        m = copy.deepcopy(model)
        m.fit(X[train_idx], y[train_idx])
        y_pred = m.predict(X[val_idx])
        y_prob = m.predict_proba(X[val_idx])[:, 1]

        accs.append(accuracy_score(y[val_idx], y_pred))
        f1s.append(f1_score(y[val_idx], y_pred, zero_division=0))
        aucs.append(roc_auc_score(y[val_idx], y_prob))
        logger.info(f"[{model_name}] Fold {fold+1}/{n_splits} | acc={accs[-1]:.4f}")

    results = {
        "cv_accuracy_mean": float(np.mean(accs)),
        "cv_accuracy_std": float(np.std(accs)),
        "cv_f1_mean": float(np.mean(f1s)),
        "cv_f1_std": float(np.std(f1s)),
        "cv_auc_mean": float(np.mean(aucs)),
        "cv_auc_std": float(np.std(aucs)),
    }
    logger.info(
        f"[{model_name}] CV {n_splits}-fold → "
        f"Acc={results['cv_accuracy_mean']:.4f}±{results['cv_accuracy_std']:.4f} | "
        f"AUC={results['cv_auc_mean']:.4f}±{results['cv_auc_std']:.4f}"
    )
    return results


# ─────────────────────────────────────────────────────────────────────────────
# McNemar Statistical Significance Test
# ─────────────────────────────────────────────────────────────────────────────

def mcnemar_test(
    y_true: np.ndarray,
    y_pred_a: np.ndarray,
    y_pred_b: np.ndarray,
    model_a_name: str = "Model A",
    model_b_name: str = "Model B",
) -> Dict[str, float]:
    """
    McNemar test for statistical significance of performance difference
    between two classifiers.

    Contingency table:
        b01 = cases where A correct, B wrong
        b10 = cases where A wrong, B correct

    H0: Both classifiers have the same error rate.
    p < 0.05 → reject H0, difference is statistically significant.

    Reference: Interim report — "McNemar test (p < 0.05)"
    """
    correct_a = (y_pred_a == y_true).astype(int)
    correct_b = (y_pred_b == y_true).astype(int)

    b01 = np.sum((correct_a == 1) & (correct_b == 0))
    b10 = np.sum((correct_a == 0) & (correct_b == 1))
    b00 = np.sum((correct_a == 0) & (correct_b == 0))
    b11 = np.sum((correct_a == 1) & (correct_b == 1))

    contingency = np.array([[b11, b10], [b01, b00]])

    # McNemar with continuity correction
    if b01 + b10 == 0:
        chi2, p_value = 0.0, 1.0
    else:
        chi2 = (abs(b01 - b10) - 1) ** 2 / (b01 + b10)
        from scipy.stats import chi2 as chi2_dist
        p_value = 1 - chi2_dist.cdf(chi2, df=1)

    significant = p_value < 0.05
    logger.info(
        f"McNemar test: {model_a_name} vs {model_b_name} | "
        f"chi2={chi2:.4f}, p={p_value:.4f} | "
        f"{'SIGNIFICANT' if significant else 'not significant'}"
    )
    return {
        "chi2": float(chi2),
        "p_value": float(p_value),
        "significant": significant,
        "b01": int(b01),
        "b10": int(b10),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Full Evaluation Runner
# ─────────────────────────────────────────────────────────────────────────────

class ModelEvaluator:
    """
    Orchestrates the full evaluation pipeline for all PhishNet models.
    Collects metrics, benchmarks latency, and exports results to Excel.
    """

    def __init__(self, config: Dict, output_dir: str = "results/metrics"):
        self.config = config
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.results: Dict[str, Dict] = {}

    def evaluate_keras(
        self,
        model: tf.keras.Model,
        X_test: np.ndarray,
        y_test: np.ndarray,
        model_name: str,
        benchmark_latency: bool = True,
    ) -> Dict:
        """Full evaluation of a Keras model on the held-out test set."""
        logger.info(f"Evaluating {model_name}...")

        y_prob = model.predict(X_test, verbose=0).flatten()
        y_pred = (y_prob >= 0.5).astype(int)

        metrics = compute_classification_metrics(y_test, y_pred, y_prob, model_name)

        if benchmark_latency:
            sample = X_test[:1]
            lat = benchmark_inference_latency(
                model, sample, model_type="keras",
                warmup_runs=self.config["evaluation"]["latency_warmup_runs"],
                benchmark_runs=self.config["evaluation"]["latency_benchmark_runs"],
            )
            mem = measure_memory_usage(model, sample, model_type="keras")
            metrics.update(lat)
            metrics.update(mem)

        self.results[model_name] = metrics
        return metrics

    def evaluate_sklearn(
        self,
        model,
        X_test: np.ndarray,
        y_test: np.ndarray,
        model_name: str,
        benchmark_latency: bool = True,
    ) -> Dict:
        """Full evaluation of an sklearn model on the held-out test set."""
        logger.info(f"Evaluating {model_name}...")

        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)

        metrics = compute_classification_metrics(y_test, y_pred, y_prob, model_name)

        if benchmark_latency:
            sample = X_test[:1]
            lat = benchmark_inference_latency(
                model, sample, model_type="sklearn",
                warmup_runs=self.config["evaluation"]["latency_warmup_runs"],
                benchmark_runs=self.config["evaluation"]["latency_benchmark_runs"],
            )
            mem = measure_memory_usage(model, sample, model_type="sklearn")
            metrics.update(lat)
            metrics.update(mem)

        self.results[model_name] = metrics
        return metrics

    def export_to_excel(self, filename: str = "phishnet_results.xlsx") -> str:
        """Export all collected results to Excel spreadsheet."""
        path = os.path.join(self.output_dir, filename)
        save_metrics_to_excel(self.results, path)
        return path

    def get_comparison_dataframe(self) -> pd.DataFrame:
        """Return a clean comparison DataFrame of all models."""
        rows = []
        display_cols = [
            "accuracy", "precision", "recall", "f1",
            "auc_roc", "auc_pr", "fpr", "fnr",
            "latency_p50_ms", "latency_p95_ms", "throughput_urls_per_sec",
        ]
        for model_name, metrics in self.results.items():
            row = {"Model": model_name}
            for col in display_cols:
                row[col] = round(metrics.get(col, float("nan")), 4)
            rows.append(row)
        return pd.DataFrame(rows).set_index("Model")
