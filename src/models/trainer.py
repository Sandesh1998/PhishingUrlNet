"""
PhishNet - Training Pipeline
Sandesh Poudel | B01818884 | UWS MSc Cybersecurity

Handles end-to-end training for all models:
  - Keras callbacks (EarlyStopping, ReduceLROnPlateau, ModelCheckpoint)
  - MLflow experiment tracking
  - Training history logging & plotting
  - Sklearn baseline training
"""
import os
import time
import json
import numpy as np
from typing import Dict, Tuple, Optional, Any

import tensorflow as tf
from tensorflow.keras import callbacks as keras_callbacks

try:
    import mlflow
    import mlflow.tensorflow
    import mlflow.sklearn
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

from src.utils.logger import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Keras Training
# ─────────────────────────────────────────────────────────────────────────────

def get_callbacks(
    model_name: str,
    checkpoint_dir: str,
    patience: int = 7,
    min_lr: float = 1e-6,
) -> list:
    """
    Build standard Keras callbacks:
      - EarlyStopping: stop when val_loss stops improving
      - ReduceLROnPlateau: halve LR when val_loss plateaus
      - ModelCheckpoint: save best weights
      - TensorBoard: training curve logging
    """
    os.makedirs(checkpoint_dir, exist_ok=True)

    cb_list = [
        keras_callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True,
            verbose=1,
        ),
        keras_callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=patience // 2,
            min_lr=min_lr,
            verbose=1,
        ),
        keras_callbacks.ModelCheckpoint(
            filepath=os.path.join(checkpoint_dir, f"{model_name}_best.keras"),
            monitor="val_auc",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        keras_callbacks.TensorBoard(
            log_dir=os.path.join(checkpoint_dir, "tensorboard", model_name),
            histogram_freq=0,
        ),
    ]
    return cb_list


def train_keras_model(
    model: tf.keras.Model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    config: Dict,
    model_name: str,
    use_mlflow: bool = True,
) -> Tuple[tf.keras.Model, Dict]:
    """
    Train a Keras model (CNN or CNN-TCN) with full callback suite.

    Args:
        model: Compiled Keras model
        X_train / y_train: Training data and labels
        X_val / y_val: Validation data and labels
        config: Project configuration dictionary
        model_name: Identifier string used for saving artifacts
        use_mlflow: Whether to log run to MLflow

    Returns:
        (trained_model, history_dict)
    """
    # Determine correct config section
    cfg_key = "cnn_tcn" if "TCN" in model.name.upper() else "cnn"
    if "DNN" in model.name.upper():
        cfg_key = "baselines"

    if cfg_key in ("cnn", "cnn_tcn"):
        cfg = config[cfg_key]
    else:
        cfg = config["baselines"]["dnn"]

    batch_size = cfg.get("batch_size", 256)
    epochs = cfg.get("epochs", 50)
    patience = cfg.get("early_stopping_patience", 7)

    checkpoint_dir = config["paths"]["models"]
    cb_list = get_callbacks(model_name, checkpoint_dir, patience)

    # Class weights to handle remaining imbalance after SMOTE
    unique, counts = np.unique(y_train, return_counts=True)
    total = len(y_train)
    class_weight = {int(k): total / (len(unique) * v) for k, v in zip(unique, counts)}
    logger.info(f"Class weights: {class_weight}")

    logger.info(f"[{model_name}] Training started | epochs={epochs}, batch={batch_size}")
    t0 = time.time()

    # ── MLflow tracking ────────────────────────────────────────────────────
    if use_mlflow and MLFLOW_AVAILABLE:
        mlflow.set_experiment("PhishNet")
        run = mlflow.start_run(run_name=model_name)
        mlflow.log_params({
            "model": model_name,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": cfg.get("learning_rate", 0.001),
            "dropout_rate": cfg.get("dropout_rate", 0.4),
        })

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=batch_size,
        epochs=epochs,
        callbacks=cb_list,
        class_weight=class_weight,
        verbose=1,
    )

    elapsed = time.time() - t0
    logger.info(f"[{model_name}] Training complete in {elapsed:.1f}s")

    # Save final model
    final_path = os.path.join(checkpoint_dir, f"{model_name}_final.keras")
    model.save(final_path)
    logger.info(f"Model saved → {final_path}")

    # Save history
    hist_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    hist_path = os.path.join(checkpoint_dir, f"{model_name}_history.json")
    with open(hist_path, "w") as f:
        json.dump(hist_dict, f, indent=2)

    if use_mlflow and MLFLOW_AVAILABLE:
        # Log final val metrics
        final_metrics = {k: v[-1] for k, v in hist_dict.items()}
        mlflow.log_metrics({f"final_{k}": v for k, v in final_metrics.items()})
        mlflow.tensorflow.log_model(model, model_name)
        mlflow.end_run()

    return model, hist_dict


# ─────────────────────────────────────────────────────────────────────────────
# Sklearn Baseline Training
# ─────────────────────────────────────────────────────────────────────────────

def train_sklearn_model(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    model_name: str,
    config: Dict,
    use_mlflow: bool = True,
) -> Any:
    """
    Fit an sklearn baseline model and log with MLflow.

    Args:
        model: sklearn estimator (RF, SVM, or Pipeline)
        X_train: Feature matrix
        y_train: Labels
        model_name: Identifier string
        config: Project config

    Returns:
        Fitted sklearn estimator
    """
    logger.info(f"[{model_name}] Training sklearn model...")
    t0 = time.time()

    if use_mlflow and MLFLOW_AVAILABLE:
        mlflow.set_experiment("PhishNet")
        with mlflow.start_run(run_name=model_name):
            model.fit(X_train, y_train)
            mlflow.sklearn.log_model(model, model_name)
    else:
        model.fit(X_train, y_train)

    elapsed = time.time() - t0
    logger.info(f"[{model_name}] Training complete in {elapsed:.1f}s")
    return model
