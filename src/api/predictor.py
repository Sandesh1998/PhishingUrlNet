"""
PhishNet - Model Inference Wrapper
Sandesh Poudel | B01818884 | UWS MSc Cybersecurity

Loads trained models at startup and provides a clean predict() interface
for the FastAPI layer. Handles:
  - Lazy model loading + caching
  - Feature extraction pipeline
  - Confidence → risk level mapping
  - Inference timing
"""
import os
import time
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple

import tensorflow as tf

from src.features.feature_extractor import (
    extract_all_features,
)
from src.features.preprocessor import PhishNetPreprocessor
from src.utils.helpers import url_to_char_sequence, load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Risk thresholds (calibrated on validation set)
RISK_THRESHOLDS = {
    "LOW":      (0.0,  0.30),
    "MEDIUM":   (0.30, 0.60),
    "HIGH":     (0.60, 0.85),
    "CRITICAL": (0.85, 1.01),
}


def _confidence_to_risk(confidence: float) -> str:
    for level, (lo, hi) in RISK_THRESHOLDS.items():
        if lo <= confidence < hi:
            return level
    return "CRITICAL"


class PhishNetPredictor:
    """
    Stateful model loader and inference engine.
    Instantiated once at API startup and reused across all requests.
    """

    def __init__(self, model_dir: str = "results/models"):
        self.project_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..")
        )
        self.model_dir = (
            model_dir
            if os.path.isabs(model_dir)
            else os.path.join(self.project_root, model_dir)
        )
        self.config = load_config()
        self._models: Dict = {}
        self._model_metrics: Dict = {}
        self.preprocessor: Optional[PhishNetPreprocessor] = None
        data_processed = self.config["paths"]["data_processed"]
        data_processed = (
            data_processed
            if os.path.isabs(data_processed)
            else os.path.join(self.project_root, data_processed)
        )
        self.preprocessor_path = os.path.join(
            data_processed,
            "preprocessor.pkl",
        )
        self.max_len = self.config["features"]["max_url_length"]
        self.vocab_size = self.config["features"]["char_vocab_size"]

    def load_all(self) -> None:
        """Load all available trained models from disk."""
        if os.path.exists(self.preprocessor_path):
            try:
                self.preprocessor = PhishNetPreprocessor.load(self.preprocessor_path)
            except Exception as e:
                logger.warning(f"Could not load preprocessor: {e}")

        model_map = {
            "cnn":     ("cnn_final.keras",   "keras"),
            "cnn_tcn": ("cnn_tcn_final.keras", "keras"),
            "rf":      ("rf_model.pkl",       "sklearn"),
            "svm":     ("svm_model.pkl",      "sklearn"),
            "dnn":     ("dnn_final.keras",    "keras"),
        }
        for name, (filename, mtype) in model_map.items():
            path = os.path.join(self.model_dir, filename)
            if os.path.exists(path):
                try:
                    if mtype == "keras":
                        self._models[name] = tf.keras.models.load_model(path)
                    else:
                        self._models[name] = joblib.load(path)
                    logger.info(f"Loaded model: {name} ← {path}")
                except Exception as e:
                    logger.warning(f"Could not load {name}: {e}")
            else:
                logger.warning(f"Model file not found: {path}")

    @property
    def loaded_models(self):
        return list(self._models.keys())

    def _extract_features_for_tabular(self, url: str) -> np.ndarray:
        """Extract and preprocess tabular features for sklearn / DNN models."""
        feats = extract_all_features(url, include_host=False)
        df = pd.DataFrame([feats])

        if self.preprocessor is not None:
            expected_cols = self.preprocessor.num_cols + self.preprocessor.cat_cols
            df = df.reindex(columns=expected_cols, fill_value=0.0)
            return self.preprocessor.transform(df).astype(np.float32)

        # Fallback when preprocessor is unavailable.
        return np.array(list(feats.values()), dtype=np.float32).reshape(1, -1)

    def _extract_sequence(self, url: str) -> np.ndarray:
        """Convert URL to padded integer sequence for CNN/CNN-TCN."""
        seq = url_to_char_sequence(url, self.max_len, self.vocab_size)
        return seq.reshape(1, -1)

    def predict(
        self,
        url: str,
        model_name: str = "cnn_tcn",
    ) -> Tuple[float, float]:
        """
        Run inference for a single URL.

        Returns:
            (confidence_score, latency_ms)
        """
        if model_name not in self._models:
            raise ValueError(
                f"Model '{model_name}' not loaded. "
                f"Available: {self.loaded_models}"
            )

        model = self._models[model_name]
        t0 = time.perf_counter()

        if model_name in ("cnn", "cnn_tcn"):
            X = self._extract_sequence(url)
            confidence = float(model.predict(X, verbose=0).flatten()[0])
        elif model_name == "dnn":
            X = self._extract_features_for_tabular(url)
            confidence = float(model.predict(X, verbose=0).flatten()[0])
        else:
            # sklearn (RF, SVM)
            X = self._extract_features_for_tabular(url)
            confidence = float(model.predict_proba(X)[0][1])

        latency_ms = (time.perf_counter() - t0) * 1000
        return confidence, latency_ms

    def predict_batch(
        self,
        urls: list,
        model_name: str = "cnn_tcn",
    ) -> Tuple[np.ndarray, float]:
        """
        Batch inference. Returns (confidences_array, avg_latency_ms).
        """
        if model_name not in self._models:
            raise ValueError(f"Model '{model_name}' not loaded.")

        model = self._models[model_name]
        t0 = time.perf_counter()

        if model_name in ("cnn", "cnn_tcn"):
            from src.utils.helpers import batch_urls_to_sequences
            X = batch_urls_to_sequences(urls, self.max_len, self.vocab_size)
            confidences = model.predict(X, verbose=0).flatten()
        elif model_name == "dnn":
            X = np.vstack([self._extract_features_for_tabular(u) for u in urls])
            confidences = model.predict(X, verbose=0).flatten()
        else:
            X = np.vstack([self._extract_features_for_tabular(u) for u in urls])
            confidences = model.predict_proba(X)[:, 1]

        elapsed_ms = (time.perf_counter() - t0) * 1000
        avg_latency = elapsed_ms / len(urls)
        return confidences, avg_latency
