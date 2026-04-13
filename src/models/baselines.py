"""
PhishNet - Baseline Models
Sandesh Poudel | B01818884 | UWS MSc Cybersecurity

Implements comparison baselines:
  1. Random Forest
  2. Support Vector Machine (SVM)
  3. Standard Deep Neural Network (DNN)

These are compared against CNN and CNN-TCN as per the project specification.
"""
import os
import numpy as np
import joblib
from typing import Dict, Any

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks

from src.utils.logger import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Random Forest
# ─────────────────────────────────────────────────────────────────────────────

def build_random_forest(config: Dict) -> RandomForestClassifier:
    """
    Random Forest ensemble classifier.
    Serves as the primary traditional ML baseline.
    Reference: Abu-Nimeh et al. (2007) — 96.8% accuracy on phishing email corpus.
    """
    cfg = config["baselines"]["random_forest"]
    clf = RandomForestClassifier(
        n_estimators=cfg.get("n_estimators", 200),
        max_depth=cfg.get("max_depth", None),
        min_samples_split=cfg.get("min_samples_split", 2),
        random_state=config["project"]["seed"],
        n_jobs=cfg.get("n_jobs", -1),
        class_weight="balanced",     # handles imbalanced datasets
    )
    logger.info("Random Forest baseline built")
    return clf


# ─────────────────────────────────────────────────────────────────────────────
# 2. Support Vector Machine
# ─────────────────────────────────────────────────────────────────────────────

def build_svm(config: Dict) -> Pipeline:
    """
    SVM with RBF kernel, wrapped in a Pipeline with StandardScaler.
    probability=True enables predict_proba for ROC/PR curve computation.
    """
    cfg = config["baselines"]["svm"]
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(
            C=cfg.get("C", 1.0),
            kernel=cfg.get("kernel", "rbf"),
            gamma=cfg.get("gamma", "scale"),
            probability=cfg.get("probability", True),
            random_state=config["project"]["seed"],
            class_weight="balanced",
        )),
    ])
    logger.info("SVM baseline built")
    return pipeline


# ─────────────────────────────────────────────────────────────────────────────
# 3. Standard Deep Neural Network (DNN)
# ─────────────────────────────────────────────────────────────────────────────

def build_dnn(input_dim: int, config: Dict) -> tf.keras.Model:
    """
    Fully-connected DNN for tabular feature vectors.
    Serves as the deep learning baseline (without convolutions).

    Args:
        input_dim: Number of input features
        config: Project config dict
    """
    cfg = config["baselines"]["dnn"]
    hidden_layers = cfg.get("hidden_layers", [256, 128, 64])
    dropout_rate = cfg.get("dropout_rate", 0.3)
    learning_rate = cfg.get("learning_rate", 0.001)

    inp = layers.Input(shape=(input_dim,), name="feature_input")
    x = inp

    for i, units in enumerate(hidden_layers):
        x = layers.Dense(units, name=f"dense_{i+1}")(x)
        x = layers.BatchNormalization(name=f"bn_{i+1}")(x)
        x = layers.Activation("relu", name=f"relu_{i+1}")(x)
        x = layers.Dropout(rate=dropout_rate, name=f"drop_{i+1}")(x)

    out = layers.Dense(1, activation="sigmoid", name="output")(x)

    model = models.Model(inputs=inp, outputs=out, name="PhishNet_DNN")
    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )
    logger.info(f"DNN baseline built | input_dim={input_dim} | params: {model.count_params():,}")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Persistence helpers
# ─────────────────────────────────────────────────────────────────────────────

def save_sklearn_model(model, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    logger.info(f"Sklearn model saved → {path}")


def load_sklearn_model(path: str):
    model = joblib.load(path)
    logger.info(f"Sklearn model loaded ← {path}")
    return model
