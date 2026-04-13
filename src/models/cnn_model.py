"""
PhishNet - 1D CNN Model
Sandesh Poudel | B01818884 | UWS MSc Cybersecurity

1D Convolutional Neural Network operating on character-level URL sequences.
Architecture:
  Embedding → Conv1D blocks (ReLU + BatchNorm + MaxPool + Dropout)
  → GlobalMaxPool → Dense → Sigmoid

Reference: Project specification - "Train 1D CNN architecture based on URL sequence analysis"
"""
import os
import numpy as np
from typing import Dict, Tuple, Optional

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.regularizers import l2

from src.utils.logger import get_logger

logger = get_logger(__name__)


def build_cnn_model(
    vocab_size: int = 128,
    max_len: int = 200,
    embedding_dim: int = 64,
    filters: list = None,
    kernel_sizes: list = None,
    dense_units: list = None,
    dropout_rate: float = 0.4,
    learning_rate: float = 0.001,
) -> tf.keras.Model:
    """
    Build and compile the 1D CNN model.

    Architecture overview:
      1. Embedding layer: integer tokens → dense vectors
      2. Three Conv1D blocks: Conv → BatchNorm → ReLU → MaxPool → Dropout
      3. GlobalMaxPooling1D: collapse temporal dimension
      4. Two Dense layers with BatchNorm and Dropout
      5. Sigmoid output for binary classification

    Args:
        vocab_size: Character vocabulary size (default 128 for ASCII)
        max_len: URL sequence length after padding
        embedding_dim: Embedding vector dimension
        filters: List of filter sizes per Conv block
        kernel_sizes: List of kernel sizes per Conv block
        dense_units: Hidden units in fully-connected layers
        dropout_rate: Dropout probability
        learning_rate: Adam optimizer learning rate

    Returns:
        Compiled Keras model
    """
    if filters is None:
        filters = [128, 64, 32]
    if kernel_sizes is None:
        kernel_sizes = [3, 3, 3]
    if dense_units is None:
        dense_units = [128, 64]

    assert len(filters) == len(kernel_sizes), "filters and kernel_sizes must match"

    # ── Input ──────────────────────────────────────────────────────────────
    inp = layers.Input(shape=(max_len,), name="url_input")

    # ── Embedding ──────────────────────────────────────────────────────────
    x = layers.Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        input_length=max_len,
        name="char_embedding"
    )(inp)

    # ── Convolutional Blocks ───────────────────────────────────────────────
    for i, (f, k) in enumerate(zip(filters, kernel_sizes)):
        x = layers.Conv1D(
            filters=f,
            kernel_size=k,
            padding="same",
            activation="relu",
            kernel_regularizer=l2(1e-4),
            name=f"conv_{i+1}"
        )(x)
        x = layers.BatchNormalization(name=f"bn_{i+1}")(x)
        x = layers.MaxPooling1D(pool_size=2, name=f"pool_{i+1}")(x)
        x = layers.Dropout(rate=dropout_rate, name=f"drop_conv_{i+1}")(x)

    # ── Global Max Pooling ─────────────────────────────────────────────────
    x = layers.GlobalMaxPooling1D(name="global_max_pool")(x)

    # ── Dense Layers ──────────────────────────────────────────────────────
    for i, units in enumerate(dense_units):
        x = layers.Dense(units, name=f"dense_{i+1}")(x)
        x = layers.BatchNormalization(name=f"bn_dense_{i+1}")(x)
        x = layers.Activation("relu", name=f"relu_dense_{i+1}")(x)
        x = layers.Dropout(rate=dropout_rate, name=f"drop_dense_{i+1}")(x)

    # ── Output ─────────────────────────────────────────────────────────────
    out = layers.Dense(1, activation="sigmoid", name="output")(x)

    model = models.Model(inputs=inp, outputs=out, name="PhishNet_CNN")

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

    logger.info(f"CNN model built | params: {model.count_params():,}")
    return model


def build_from_config(config: Dict) -> tf.keras.Model:
    """Convenience wrapper: build CNN model from config dict."""
    cfg = config["cnn"]
    feat = config["features"]
    return build_cnn_model(
        vocab_size=feat["char_vocab_size"],
        max_len=feat["max_url_length"],
        embedding_dim=feat["embedding_dim"],
        filters=cfg["filters"],
        kernel_sizes=cfg["kernel_sizes"],
        dense_units=cfg["dense_units"],
        dropout_rate=cfg["dropout_rate"],
        learning_rate=cfg["learning_rate"],
    )
