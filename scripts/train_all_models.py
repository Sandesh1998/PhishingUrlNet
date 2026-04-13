"""
PhishNet - Full Training Pipeline
Sandesh Poudel | B01818884 | UWS MSc Cybersecurity

End-to-end training script. Run this after download_data.py.

Pipeline:
  1. Load and preprocess combined_urls.csv
  2. Extract all features (lexical, host-based, temporal)
  3. Split train / val / test (stratified)
  4. Apply SMOTE on training set
  5. Train 5 models: CNN, CNN-TCN, RF, SVM, DNN
  6. Save all model weights to results/models/

Usage:
    python scripts/train_all_models.py [--skip-features] [--models cnn cnn_tcn]

Time estimate (CPU only, ~50k samples):
  CNN:     ~15 min
  CNN-TCN: ~25 min
  RF:      ~3 min
  SVM:     ~5 min (use LinearSVC for large datasets)
  DNN:     ~10 min
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.helpers import load_config, set_seed
from src.utils.logger import get_logger
from src.features.feature_extractor import extract_all_features
from src.features.preprocessor import PhishNetPreprocessor
from src.models.cnn_model import build_from_config as build_cnn
from src.models.cnn_tcn_model import build_from_config as build_cnn_tcn
from src.models.baselines import build_random_forest, build_svm, build_dnn, save_sklearn_model
from src.models.trainer import train_keras_model, train_sklearn_model

logger = get_logger(__name__, log_dir="results/logs")


def _clean_training_rows(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Defensive clean-up for legacy noisy rows in existing combined datasets."""
    df = df_raw.copy()
    df = df.dropna(subset=["url", "label"]).copy()
    df["url"] = df["url"].astype(str).str.strip()
    df["label"] = df["label"].astype(int)

    # Remove synthetic placeholders introduced by old UCI URL fabrication.
    df = df[~df["url"].str.contains(r"example-\d+\.com", regex=True, na=False)]
    df = df[df["url"].str.len() > 7]
    df = df.drop_duplicates(subset="url")
    return df.reset_index(drop=True)


def parse_args():
    p = argparse.ArgumentParser(description="PhishNet Training Pipeline")
    p.add_argument("--skip-features", action="store_true",
                   help="Skip feature extraction (use cached processed data)")
    p.add_argument("--models", nargs="+",
                   default=["cnn", "cnn_tcn", "rf", "svm", "dnn"],
                   help="Which models to train")
    p.add_argument("--config", default="configs/config.yaml")
    return p.parse_args()


def load_and_preprocess(config: dict, skip_features: bool = False):
    """Load data, extract features, split, apply SMOTE."""
    raw_csv = os.path.join(config["paths"]["data_raw"], "combined_urls.csv")
    processed_dir = config["paths"]["data_processed"]
    os.makedirs(processed_dir, exist_ok=True)

    features_csv = os.path.join(processed_dir, "features.csv")
    preprocessor_path = os.path.join(processed_dir, "preprocessor.pkl")

    # ── Feature Extraction ──────────────────────────────────────────────────
    if skip_features and os.path.exists(features_csv):
        logger.info("Loading cached feature CSV...")
        df_feat = pd.read_csv(features_csv)
    else:
        if not os.path.exists(raw_csv):
            raise FileNotFoundError(
                f"{raw_csv} not found. Run `python scripts/download_data.py` first."
            )
        logger.info("Extracting features — this may take several minutes...")
        df_raw = pd.read_csv(raw_csv)
        before = len(df_raw)
        df_raw = _clean_training_rows(df_raw)
        logger.info(f"Cleaned training data: {before:,} -> {len(df_raw):,} rows")
        urls = df_raw["url"].tolist()
        labels = df_raw["label"].tolist()

        rows = []
        from tqdm import tqdm
        for url in tqdm(urls, desc="Feature extraction", unit="url"):
            try:
                feats = extract_all_features(url, include_host=False)   # fast mode
                rows.append(feats)
            except Exception:
                rows.append({})

        df_feat = pd.DataFrame(rows)
        df_feat["url"] = urls
        df_feat["label"] = labels
        df_feat.to_csv(features_csv, index=False)
        logger.info(f"Features saved → {features_csv}")

    labels = df_feat["label"].values.astype(int)
    urls = df_feat["url"].tolist() if "url" in df_feat.columns else []
    X_tab = df_feat.drop(columns=["label", "url"], errors="ignore")

    # ── Tabular Preprocessing ───────────────────────────────────────────────
    preprocessor = PhishNetPreprocessor(config)
    X_tab_scaled = preprocessor.fit_transform(X_tab, pd.Series(labels))
    preprocessor.save(preprocessor_path)

    # ── Train/Val/Test Split ────────────────────────────────────────────────
    splits = preprocessor.split(
        X_tab_scaled, labels,
        test_size=config["preprocessing"]["test_size"],
        val_size=config["preprocessing"]["val_size"],
    )
    X_train_tab, X_val_tab, X_test_tab, y_train, y_val, y_test = splits

    # ── SMOTE ───────────────────────────────────────────────────────────────
    if config["preprocessing"]["use_smote"]:
        X_train_tab, y_train = preprocessor.apply_smote(X_train_tab, y_train)

    # ── URL Sequences for CNN/CNN-TCN ───────────────────────────────────────
    if urls:
        # Re-split using same indices approach — simpler: just use all URLs
        n = len(urls)
        idx = np.arange(n)
        from sklearn.model_selection import StratifiedShuffleSplit
        sss = StratifiedShuffleSplit(
            n_splits=1,
            test_size=config["preprocessing"]["test_size"],
            random_state=config["preprocessing"]["random_state"]
        )
        train_idx, test_idx = next(sss.split(idx, labels))
        sss2 = StratifiedShuffleSplit(
            n_splits=1,
            test_size=config["preprocessing"]["val_size"],
            random_state=config["preprocessing"]["random_state"]
        )
        train_idx2, val_idx = next(sss2.split(train_idx, labels[train_idx]))
        actual_train_idx = train_idx[train_idx2]

        train_urls = [urls[i] for i in actual_train_idx]
        val_urls   = [urls[i] for i in val_idx]
        test_urls  = [urls[i] for i in test_idx]

        X_train_seq = preprocessor.urls_to_sequences(train_urls)
        X_val_seq   = preprocessor.urls_to_sequences(val_urls)
        X_test_seq  = preprocessor.urls_to_sequences(test_urls)

        y_train_seq = labels[actual_train_idx]
        y_val_seq   = labels[val_idx]
        y_test_seq  = labels[test_idx]
    else:
        X_train_seq = X_val_seq = X_test_seq = None
        y_train_seq = y_val_seq = y_test_seq = y_train

    return {
        "tabular": (X_train_tab, X_val_tab, X_test_tab, y_train, y_val, y_test),
        "sequence": (X_train_seq, X_val_seq, X_test_seq,
                     y_train_seq, y_val_seq, y_test_seq),
        "input_dim": X_train_tab.shape[1],
    }


def main():
    args = parse_args()
    config = load_config(args.config)
    set_seed(config["project"]["seed"])

    logger.info("=" * 60)
    logger.info("PhishNet Training Pipeline")
    logger.info(f"Models to train: {args.models}")
    logger.info("=" * 60)

    # Load & preprocess data
    data = load_and_preprocess(config, skip_features=args.skip_features)
    X_train_tab, X_val_tab, X_test_tab, y_train, y_val, y_test = data["tabular"]
    X_train_seq, X_val_seq, X_test_seq, y_train_seq, y_val_seq, y_test_seq = data["sequence"]
    input_dim = data["input_dim"]

    models_dir = config["paths"]["models"]
    os.makedirs(models_dir, exist_ok=True)

    # ── CNN ─────────────────────────────────────────────────────────────────
    if "cnn" in args.models and X_train_seq is not None:
        logger.info("\n>>> Training 1D CNN...")
        cnn = build_cnn(config)
        cnn.summary()
        train_keras_model(
            cnn, X_train_seq, y_train_seq,
            X_val_seq, y_val_seq,
            config, model_name="cnn",
        )

    # ── CNN-TCN ─────────────────────────────────────────────────────────────
    if "cnn_tcn" in args.models and X_train_seq is not None:
        logger.info("\n>>> Training CNN-TCN Hybrid...")
        cnn_tcn = build_cnn_tcn(config)
        cnn_tcn.summary()
        train_keras_model(
            cnn_tcn, X_train_seq, y_train_seq,
            X_val_seq, y_val_seq,
            config, model_name="cnn_tcn",
        )

    # ── Random Forest ────────────────────────────────────────────────────────
    if "rf" in args.models:
        logger.info("\n>>> Training Random Forest...")
        rf = build_random_forest(config)
        rf = train_sklearn_model(rf, X_train_tab, y_train, "rf", config)
        save_sklearn_model(rf, os.path.join(models_dir, "rf_model.pkl"))

    # ── SVM ──────────────────────────────────────────────────────────────────
    if "svm" in args.models:
        logger.info("\n>>> Training SVM...")
        svm = build_svm(config)
        rf = train_sklearn_model(svm, X_train_tab, y_train, "svm", config)
        save_sklearn_model(svm, os.path.join(models_dir, "svm_model.pkl"))

    # ── DNN ──────────────────────────────────────────────────────────────────
    if "dnn" in args.models:
        logger.info("\n>>> Training DNN baseline...")
        dnn = build_dnn(input_dim, config)
        train_keras_model(
            dnn, X_train_tab, y_train,
            X_val_tab, y_val,
            config, model_name="dnn",
        )

    logger.info("\n✓ Training pipeline complete. Run scripts/run_evaluation.py next.")


if __name__ == "__main__":
    main()
