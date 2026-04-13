"""
PhishNet - Full Evaluation Pipeline
Sandesh Poudel | B01818884 | UWS MSc Cybersecurity

Loads trained models, runs the complete evaluation suite, and exports:
  - results/metrics/phishnet_results.xlsx  (Excel deliverable)
  - results/metrics/phishnet_results.json  (API metrics endpoint)
  - results/plots/                         (all visualisations)

Usage:
    python scripts/run_evaluation.py

Cross-dataset validation:
  Train PhishTank/UNB → test on UCI/Alexa (generalisation)
"""
import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.helpers import load_config, set_seed, batch_urls_to_sequences
from src.utils.logger import get_logger
from src.features.preprocessor import PhishNetPreprocessor
from src.evaluation.evaluator import (
    ModelEvaluator, compute_classification_metrics,
    get_roc_curve_data, get_pr_curve_data, mcnemar_test,
    cross_validate_sklearn,
)
from src.evaluation.visualiser import (
    plot_roc_curves, plot_pr_curves, plot_confusion_matrix,
    plot_training_history, plot_model_comparison,
    plot_accuracy_vs_latency, plot_feature_importance,
)
import tensorflow as tf
import joblib

logger = get_logger(__name__, log_dir="results/logs")


def load_test_data(config: dict):
    """Load the test set (features + sequences) from the processed data directory."""
    processed_dir = config["paths"]["data_processed"]
    raw_csv = os.path.join(config["paths"]["data_raw"], "combined_urls.csv")
    preprocessor_path = os.path.join(processed_dir, "preprocessor.pkl")
    features_csv = os.path.join(processed_dir, "features.csv")

    if not os.path.exists(preprocessor_path):
        raise FileNotFoundError("Preprocessor not found. Run train_all_models.py first.")

    preprocessor = PhishNetPreprocessor.load(preprocessor_path)

    if os.path.exists(features_csv):
        df_feat = pd.read_csv(features_csv)
    else:
        raise FileNotFoundError("features.csv not found. Run train_all_models.py first.")

    labels = df_feat["label"].values.astype(int)
    urls = df_feat["url"].tolist() if "url" in df_feat.columns else []
    X_tab = df_feat.drop(columns=["label", "url"], errors="ignore")
    X_tab_scaled = preprocessor.transform(X_tab)

    # Rebuild the same test split
    from sklearn.model_selection import StratifiedShuffleSplit
    n = len(labels)
    idx = np.arange(n)
    sss = StratifiedShuffleSplit(
        n_splits=1, test_size=config["preprocessing"]["test_size"], random_state=42
    )
    _, test_idx = next(sss.split(idx, labels))

    X_test_tab = X_tab_scaled[test_idx]
    y_test = labels[test_idx]
    test_urls = [urls[i] for i in test_idx] if urls else []

    X_test_seq = None
    if test_urls:
        X_test_seq = batch_urls_to_sequences(
            test_urls,
            config["features"]["max_url_length"],
            config["features"]["char_vocab_size"],
        )

    return X_test_tab, X_test_seq, y_test, test_urls, df_feat, labels


def main():
    config = load_config()
    set_seed(config["project"]["seed"])

    models_dir = config["paths"]["models"]
    plots_dir  = config["paths"]["plots"]
    metrics_dir = config["paths"]["metrics"]

    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)

    logger.info("=" * 60)
    logger.info("PhishNet Evaluation Pipeline")
    logger.info("=" * 60)

    # ── Load test data ───────────────────────────────────────────────────────
    X_test_tab, X_test_seq, y_test, test_urls, df_feat, all_labels = \
        load_test_data(config)

    evaluator = ModelEvaluator(config, output_dir=metrics_dir)
    roc_data = {}
    pr_data  = {}
    preds    = {}   # {model_name: (y_pred, y_prob)}

    # ── Evaluate Keras models (CNN, CNN-TCN, DNN) ───────────────────────────
    for model_name, filename, X_test in [
        ("CNN",     "cnn_final.keras",     X_test_seq),
        ("CNN_TCN", "cnn_tcn_final.keras", X_test_seq),
        ("DNN",     "dnn_final.keras",     X_test_tab),
    ]:
        path = os.path.join(models_dir, filename)
        if not os.path.exists(path):
            logger.warning(f"Model not found, skipping: {path}")
            continue
        if X_test is None:
            logger.warning(f"No sequence data for {model_name}, skipping.")
            continue

        logger.info(f"\n--- Evaluating {model_name} ---")
        model = tf.keras.models.load_model(path)
        metrics = evaluator.evaluate_keras(model, X_test, y_test, model_name)

        y_prob = model.predict(X_test, verbose=0).flatten()
        y_pred = (y_prob >= 0.5).astype(int)
        preds[model_name] = (y_pred, y_prob)

        fpr_arr, tpr_arr, _ = get_roc_curve_data(y_test, y_prob)
        roc_data[model_name] = (fpr_arr, tpr_arr, metrics["auc_roc"])

        prec_arr, rec_arr, _ = get_pr_curve_data(y_test, y_prob)
        pr_data[model_name] = (prec_arr, rec_arr, metrics["auc_pr"])

        # Confusion matrix
        plot_confusion_matrix(
            y_test, y_pred, model_name,
            os.path.join(plots_dir, f"cm_{model_name.lower()}.png")
        )

        # Training history
        hist_path = os.path.join(models_dir, f"{model_name.lower()}_history.json")
        if os.path.exists(hist_path):
            import json as _json
            with open(hist_path) as f:
                hist = _json.load(f)
            plot_training_history(
                hist, model_name,
                os.path.join(plots_dir, f"history_{model_name.lower()}.png")
            )

    # ── Evaluate sklearn models (RF, SVM) ────────────────────────────────────
    for model_name, filename in [("RF", "rf_model.pkl"), ("SVM", "svm_model.pkl")]:
        path = os.path.join(models_dir, filename)
        if not os.path.exists(path):
            logger.warning(f"Model not found, skipping: {path}")
            continue
        logger.info(f"\n--- Evaluating {model_name} ---")
        sk_model = joblib.load(path)
        metrics = evaluator.evaluate_sklearn(sk_model, X_test_tab, y_test, model_name)

        y_prob = sk_model.predict_proba(X_test_tab)[:, 1]
        y_pred = sk_model.predict(X_test_tab)
        preds[model_name] = (y_pred, y_prob)

        fpr_arr, tpr_arr, _ = get_roc_curve_data(y_test, y_prob)
        roc_data[model_name] = (fpr_arr, tpr_arr, metrics["auc_roc"])

        prec_arr, rec_arr, _ = get_pr_curve_data(y_test, y_prob)
        pr_data[model_name] = (prec_arr, rec_arr, metrics["auc_pr"])

        plot_confusion_matrix(
            y_test, y_pred, model_name,
            os.path.join(plots_dir, f"cm_{model_name.lower()}.png")
        )

        # Feature importance for RF
        if model_name == "RF" and hasattr(sk_model, "feature_importances_"):
            feat_names = df_feat.drop(
                columns=["label", "url"], errors="ignore"
            ).columns.tolist()
            plot_feature_importance(
                feat_names, sk_model.feature_importances_,
                os.path.join(plots_dir, "feature_importance_rf.png")
            )

    # ── Multi-model comparison plots ─────────────────────────────────────────
    if roc_data:
        plot_roc_curves(roc_data, os.path.join(plots_dir, "roc_all_models.png"))
        plot_pr_curves(pr_data,   os.path.join(plots_dir, "pr_all_models.png"))

    comp_df = evaluator.get_comparison_dataframe()
    if not comp_df.empty:
        plot_model_comparison(comp_df, os.path.join(plots_dir, "model_comparison.png"))
        plot_accuracy_vs_latency(comp_df, os.path.join(plots_dir, "accuracy_vs_latency.png"))

    # ── McNemar statistical tests ────────────────────────────────────────────
    if "CNN_TCN" in preds and "CNN" in preds:
        result = mcnemar_test(
            y_test, preds["CNN"][0], preds["CNN_TCN"][0],
            "CNN", "CNN_TCN"
        )
        logger.info(f"McNemar CNN vs CNN_TCN: {result}")

    if "CNN_TCN" in preds and "RF" in preds:
        result = mcnemar_test(
            y_test, preds["RF"][0], preds["CNN_TCN"][0],
            "RF", "CNN_TCN"
        )
        logger.info(f"McNemar RF vs CNN_TCN: {result}")

    # ── Export results ────────────────────────────────────────────────────────
    excel_path = evaluator.export_to_excel("phishnet_results.xlsx")
    logger.info(f"\n✓ Excel results saved → {excel_path}")

    # Save JSON for API /metrics endpoint
    json_path = os.path.join(metrics_dir, "phishnet_results.json")
    with open(json_path, "w") as f:
        json.dump(evaluator.results, f, indent=2, default=str)
    logger.info(f"✓ JSON metrics saved → {json_path}")

    # Print comparison table
    if not comp_df.empty:
        print("\n" + "=" * 80)
        print("PHISHNET MODEL COMPARISON RESULTS")
        print("=" * 80)
        print(comp_df[["accuracy", "precision", "recall", "f1",
                        "auc_roc", "fpr", "latency_p50_ms"]].to_string())
        print("=" * 80)

    logger.info("\n✓ Evaluation pipeline complete. See results/ directory.")


if __name__ == "__main__":
    main()
