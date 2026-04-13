"""
PhishNet - Hybrid CNN-TCN Model  *** NOVEL CONTRIBUTION ***
Sandesh Poudel | B01818884 | UWS MSc Cybersecurity

Hybrid architecture combining:
  - CNN layers:  spatial local-pattern extraction from URL characters
  - TCN layers:  dilated causal convolutions for temporal sequence modelling
  - Residual connections: stabilise gradient flow through deep stacks

This architecture addresses the core research gap identified in the literature
review: combining local character-level features (CNN) with long-range
sequential dependencies (TCN) to detect both static URL obfuscation patterns
AND dynamic campaign-level temporal behaviours.

Reference: Alorvor & Dadkhah (2025); project specification CNN-TCN proposal.
"""
import numpy as np
from typing import Dict, List

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.regularizers import l2

from src.utils.logger import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# TCN Residual Block
# ─────────────────────────────────────────────────────────────────────────────

def _tcn_residual_block(
    x: tf.Tensor,
    filters: int,
    kernel_size: int,
    dilation_rate: int,
    dropout_rate: float,
    block_id: int,
) -> tf.Tensor:
    """
    A single TCN residual block with dilated causal convolutions.

    Structure:
      Input ──► DilatedCausalConv1D ──► BatchNorm ──► ReLU ──► SpatialDropout
             ──► DilatedCausalConv1D ──► BatchNorm ──► ReLU ──► SpatialDropout
             ──► [+ 1x1 Conv residual connection] ──► ReLU ──► Output

    Dilation rate doubles per block to capture exponentially growing
    receptive fields (receptive field = 2 * kernel_size * dilation_rate).
    This allows the model to capture long-range URL structure patterns.
    """
    prefix = f"tcn_block{block_id}_d{dilation_rate}"

    # Skip connection: 1×1 conv to match channel dim if needed
    residual = layers.Conv1D(filters, 1, padding="same", name=f"{prefix}_skip")(x)

    # First dilated causal convolution
    x = layers.Conv1D(
        filters=filters,
        kernel_size=kernel_size,
        dilation_rate=dilation_rate,
        padding="causal",               # causal = no future leakage
        activation=None,
        kernel_regularizer=l2(1e-4),
        name=f"{prefix}_conv1",
    )(x)
    x = layers.BatchNormalization(name=f"{prefix}_bn1")(x)
    x = layers.Activation("relu", name=f"{prefix}_relu1")(x)
    x = layers.SpatialDropout1D(rate=dropout_rate, name=f"{prefix}_drop1")(x)

    # Second dilated causal convolution
    x = layers.Conv1D(
        filters=filters,
        kernel_size=kernel_size,
        dilation_rate=dilation_rate,
        padding="causal",
        activation=None,
        kernel_regularizer=l2(1e-4),
        name=f"{prefix}_conv2",
    )(x)
    x = layers.BatchNormalization(name=f"{prefix}_bn2")(x)
    x = layers.Activation("relu", name=f"{prefix}_relu2")(x)
    x = layers.SpatialDropout1D(rate=dropout_rate, name=f"{prefix}_drop2")(x)

    # Residual addition
    x = layers.Add(name=f"{prefix}_add")([x, residual])
    x = layers.Activation("relu", name=f"{prefix}_out_relu")(x)

    return x


# ─────────────────────────────────────────────────────────────────────────────
# Full CNN-TCN Model
# ─────────────────────────────────────────────────────────────────────────────

def build_cnn_tcn_model(
    vocab_size: int = 128,
    max_len: int = 200,
    embedding_dim: int = 64,
    cnn_filters: int = 128,
    cnn_kernel: int = 3,
    tcn_filters: int = 64,
    tcn_kernel_size: int = 3,
    tcn_dilations: List[int] = None,
    tcn_dropout: float = 0.2,
    dense_units: List[int] = None,
    dropout_rate: float = 0.4,
    learning_rate: float = 0.001,
) -> tf.keras.Model:
    """
    Build and compile the hybrid CNN-TCN model.

    Full architecture:
      Embedding
        ↓
      CNN Block (Conv1D → BatchNorm → ReLU → MaxPool → Dropout)
        ↓
      TCN Stack (N residual blocks with increasing dilation rates)
        ↓
      GlobalAvgPool1D
        ↓
      Dense → BatchNorm → ReLU → Dropout
        ↓
      Sigmoid output

    The CNN stage extracts local spatial patterns (e.g., suspicious
    substrings, character n-grams). The TCN stack then models the
    ordered sequential structure of the full URL, capturing temporal
    dependencies across character positions.

    Args:
        vocab_size: Character vocab size
        max_len: Input sequence length
        embedding_dim: Embedding dimension
        cnn_filters: Number of CNN filters
        cnn_kernel: CNN kernel size
        tcn_filters: TCN filters (same across all blocks)
        tcn_kernel_size: TCN kernel size
        tcn_dilations: List of dilation rates, e.g. [1,2,4,8,16]
        tcn_dropout: SpatialDropout rate within TCN blocks
        dense_units: Hidden units in classification head
        dropout_rate: Dropout in dense layers
        learning_rate: Adam learning rate

    Returns:
        Compiled Keras model
    """
    if tcn_dilations is None:
        tcn_dilations = [1, 2, 4, 8, 16]
    if dense_units is None:
        dense_units = [128, 64]

    # ── Input ──────────────────────────────────────────────────────────────
    inp = layers.Input(shape=(max_len,), name="url_input")

    # ── Embedding ──────────────────────────────────────────────────────────
    x = layers.Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        input_length=max_len,
        name="char_embedding",
    )(inp)
    # Shape: (batch, max_len, embedding_dim)

    # ── CNN Block (spatial feature extraction) ─────────────────────────────
    x = layers.Conv1D(
        filters=cnn_filters,
        kernel_size=cnn_kernel,
        padding="same",
        activation=None,
        kernel_regularizer=l2(1e-4),
        name="cnn_conv",
    )(x)
    x = layers.BatchNormalization(name="cnn_bn")(x)
    x = layers.Activation("relu", name="cnn_relu")(x)
    x = layers.MaxPooling1D(pool_size=2, name="cnn_pool")(x)
    x = layers.Dropout(rate=dropout_rate, name="cnn_drop")(x)
    # Shape: (batch, max_len/2, cnn_filters)

    # Project to TCN filter dimension
    if cnn_filters != tcn_filters:
        x = layers.Conv1D(tcn_filters, 1, padding="same", name="proj_conv")(x)

    # ── TCN Stack (temporal sequence modelling) ────────────────────────────
    for block_idx, dilation in enumerate(tcn_dilations):
        x = _tcn_residual_block(
            x,
            filters=tcn_filters,
            kernel_size=tcn_kernel_size,
            dilation_rate=dilation,
            dropout_rate=tcn_dropout,
            block_id=block_idx,
        )

    # ── Pooling ────────────────────────────────────────────────────────────
    x = layers.GlobalAveragePooling1D(name="global_avg_pool")(x)

    # ── Classification Head ────────────────────────────────────────────────
    for i, units in enumerate(dense_units):
        x = layers.Dense(units, name=f"dense_{i+1}")(x)
        x = layers.BatchNormalization(name=f"bn_dense_{i+1}")(x)
        x = layers.Activation("relu", name=f"relu_dense_{i+1}")(x)
        x = layers.Dropout(rate=dropout_rate, name=f"drop_dense_{i+1}")(x)

    # ── Output ─────────────────────────────────────────────────────────────
    out = layers.Dense(1, activation="sigmoid", name="output")(x)

    model = models.Model(inputs=inp, outputs=out, name="PhishNet_CNN_TCN")

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

    logger.info(
        f"CNN-TCN model built | TCN dilations: {tcn_dilations} | "
        f"params: {model.count_params():,}"
    )
    return model


def build_from_config(config: Dict) -> tf.keras.Model:
    """Convenience wrapper: build CNN-TCN model from config dict."""
    cfg = config["cnn_tcn"]
    feat = config["features"]
    return build_cnn_tcn_model(
        vocab_size=feat["char_vocab_size"],
        max_len=feat["max_url_length"],
        embedding_dim=feat["embedding_dim"],
        cnn_filters=cfg["cnn_filters"],
        cnn_kernel=cfg["cnn_kernel"],
        tcn_filters=cfg["tcn_filters"],
        tcn_kernel_size=cfg["tcn_kernel_size"],
        tcn_dilations=cfg["tcn_dilations"],
        tcn_dropout=cfg["tcn_dropout"],
        dense_units=cfg["dense_units"],
        dropout_rate=cfg["dropout_rate"],
        learning_rate=cfg["learning_rate"],
    )
