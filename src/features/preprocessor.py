"""
PhishNet - Data Preprocessor
Sandesh Poudel | B01818884 | UWS MSc Cybersecurity

Handles:
  - Missing value imputation (median for numerical, mode for categorical)
  - StandardScaler normalisation
  - One-hot encoding of categoricals
  - SMOTE for class imbalance
  - Train/val/test splitting
  - URL → character sequence encoding for CNN input
"""
import os
import numpy as np
import pandas as pd
import joblib
from typing import Tuple, Dict, Optional

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE

from src.utils.logger import get_logger
from src.utils.helpers import batch_urls_to_sequences

logger = get_logger(__name__)


class PhishNetPreprocessor:
    """
    End-to-end preprocessing pipeline for PhishNet.

    Supports two input modes:
      - 'feature': Tabular feature vectors (for RF, SVM, DNN baselines)
      - 'sequence': Character-level URL sequences (for CNN, CNN-TCN)
    """

    def __init__(self, config: Dict):
        self.config = config
        self.scaler = StandardScaler()
        self.num_imputer = SimpleImputer(strategy="median")
        self.cat_imputer = SimpleImputer(strategy="most_frequent")
        self.label_encoder = LabelEncoder()
        self._fitted = False

        # Sequence params
        self.max_url_length = config["features"]["max_url_length"]
        self.vocab_size = config["features"]["char_vocab_size"]

    # ─────────────────────────────────────────────────────────────────────────
    # FEATURE-BASED PIPELINE
    # ─────────────────────────────────────────────────────────────────────────

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "PhishNetPreprocessor":
        """Fit all transformers on training data."""
        X_num = X.select_dtypes(include=[np.number])
        X_cat = X.select_dtypes(exclude=[np.number])

        self.num_cols = X_num.columns.tolist()
        self.cat_cols = X_cat.columns.tolist()

        self.num_imputer.fit(X_num)
        if self.cat_cols:
            self.cat_imputer.fit(X_cat)

        X_imputed = self._impute(X)
        self.scaler.fit(X_imputed)
        self._fitted = True
        logger.info(f"Preprocessor fitted | {len(self.num_cols)} numeric, "
                    f"{len(self.cat_cols)} categorical features")
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Apply imputation + scaling to feature dataframe."""
        if not self._fitted:
            raise RuntimeError("Preprocessor must be fitted before transform.")
        X_imp = self._impute(X)
        return self.scaler.transform(X_imp)

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> np.ndarray:
        return self.fit(X, y).transform(X)

    def _impute(self, X: pd.DataFrame) -> np.ndarray:
        """Impute numeric and categorical columns separately."""
        X_num = self.num_imputer.transform(X[self.num_cols]) if self.num_cols else np.empty((len(X), 0))
        if self.cat_cols:
            X_cat = self.cat_imputer.transform(X[self.cat_cols])
            # Simple ordinal encoding for categoricals
            for i in range(X_cat.shape[1]):
                col = X_cat[:, i]
                unique = np.unique(col[col != None])
                mapping = {v: idx for idx, v in enumerate(unique)}
                X_cat[:, i] = np.array([mapping.get(v, -1) for v in col], dtype=float)
            return np.hstack([X_num, X_cat.astype(float)])
        return X_num

    def apply_smote(self, X: np.ndarray, y: np.ndarray,
                    random_state: int = 42) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply SMOTE to handle class imbalance.
        Reference: Project specification — "implement SMOTE or class weighting"
        """
        unique, counts = np.unique(y, return_counts=True)
        logger.info(f"Before SMOTE: {dict(zip(unique, counts))}")
        smote = SMOTE(random_state=random_state)
        X_res, y_res = smote.fit_resample(X, y)
        unique_r, counts_r = np.unique(y_res, return_counts=True)
        logger.info(f"After SMOTE:  {dict(zip(unique_r, counts_r))}")
        return X_res, y_res

    # ─────────────────────────────────────────────────────────────────────────
    # SEQUENCE PIPELINE (for CNN / CNN-TCN)
    # ─────────────────────────────────────────────────────────────────────────

    def urls_to_sequences(self, urls: list) -> np.ndarray:
        """Convert list of URL strings to padded integer sequences."""
        return batch_urls_to_sequences(urls, self.max_url_length, self.vocab_size)

    # ─────────────────────────────────────────────────────────────────────────
    # TRAIN / VAL / TEST SPLIT
    # ─────────────────────────────────────────────────────────────────────────

    def split(self,
              X: np.ndarray,
              y: np.ndarray,
              test_size: float = 0.2,
              val_size: float = 0.1,
              stratify: bool = True,
              random_state: int = 42
              ) -> Tuple[np.ndarray, ...]:
        """
        Stratified train / validation / test split.

        Returns:
            X_train, X_val, X_test, y_train, y_val, y_test
        """
        strat = y if stratify else None
        X_train_full, X_test, y_train_full, y_test = train_test_split(
            X, y, test_size=test_size, stratify=strat, random_state=random_state
        )
        # val from remaining train
        val_adjusted = val_size / (1 - test_size)
        strat2 = y_train_full if stratify else None
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_full, y_train_full, test_size=val_adjusted,
            stratify=strat2, random_state=random_state
        )
        logger.info(
            f"Split → train: {len(X_train)}, val: {len(X_val)}, test: {len(X_test)}"
        )
        return X_train, X_val, X_test, y_train, y_val, y_test

    # ─────────────────────────────────────────────────────────────────────────
    # PERSISTENCE
    # ─────────────────────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self, path)
        logger.info(f"Preprocessor saved → {path}")

    @classmethod
    def load(cls, path: str) -> "PhishNetPreprocessor":
        obj = joblib.load(path)
        logger.info(f"Preprocessor loaded ← {path}")
        return obj
