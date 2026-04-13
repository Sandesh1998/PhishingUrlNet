"""
PhishNet - Model Architecture Tests
Sandesh Poudel | B01818884 | UWS MSc Cybersecurity
"""
import pytest
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

DUMMY_CONFIG = {
    "project": {"seed": 42},
    "features": {
        "max_url_length": 50,
        "char_vocab_size": 128,
        "embedding_dim": 32,
    },
    "cnn": {
        "filters": [64, 32],
        "kernel_sizes": [3, 3],
        "dropout_rate": 0.3,
        "batch_norm": True,
        "dense_units": [64],
        "learning_rate": 0.001,
        "batch_size": 32,
        "epochs": 2,
        "early_stopping_patience": 2,
        "optimizer": "adam",
    },
    "cnn_tcn": {
        "cnn_filters": 64,
        "cnn_kernel": 3,
        "tcn_filters": 32,
        "tcn_kernel_size": 3,
        "tcn_dilations": [1, 2, 4],
        "tcn_dropout": 0.1,
        "dense_units": [64],
        "dropout_rate": 0.3,
        "batch_norm": True,
        "learning_rate": 0.001,
        "batch_size": 32,
        "epochs": 2,
        "early_stopping_patience": 2,
        "optimizer": "adam",
    },
    "baselines": {
        "random_forest": {"n_estimators": 10, "max_depth": 3, "min_samples_split": 2, "n_jobs": 1},
        "svm": {"C": 1.0, "kernel": "rbf", "gamma": "scale", "probability": True},
        "dnn": {"hidden_layers": [64, 32], "dropout_rate": 0.2, "learning_rate": 0.001,
                "batch_size": 32, "epochs": 2},
    },
}

N = 100
MAX_LEN = DUMMY_CONFIG["features"]["max_url_length"]


class TestCNNModel:
    def test_builds_successfully(self):
        from src.models.cnn_model import build_from_config
        model = build_from_config(DUMMY_CONFIG)
        assert model is not None

    def test_output_shape(self):
        from src.models.cnn_model import build_from_config
        model = build_from_config(DUMMY_CONFIG)
        X = np.random.randint(0, 128, (N, MAX_LEN))
        preds = model.predict(X, verbose=0)
        assert preds.shape == (N, 1)
        assert np.all((preds >= 0) & (preds <= 1))

    def test_model_name(self):
        from src.models.cnn_model import build_from_config
        model = build_from_config(DUMMY_CONFIG)
        assert "CNN" in model.name.upper()


class TestCNNTCNModel:
    def test_builds_successfully(self):
        from src.models.cnn_tcn_model import build_from_config
        model = build_from_config(DUMMY_CONFIG)
        assert model is not None

    def test_output_shape(self):
        from src.models.cnn_tcn_model import build_from_config
        model = build_from_config(DUMMY_CONFIG)
        X = np.random.randint(0, 128, (N, MAX_LEN))
        preds = model.predict(X, verbose=0)
        assert preds.shape == (N, 1)
        assert np.all((preds >= 0) & (preds <= 1))

    def test_has_tcn_layers(self):
        from src.models.cnn_tcn_model import build_from_config
        model = build_from_config(DUMMY_CONFIG)
        layer_names = [l.name for l in model.layers]
        tcn_layers = [n for n in layer_names if "tcn" in n]
        assert len(tcn_layers) > 0, "No TCN layers found in model"

    def test_more_params_than_cnn(self):
        from src.models.cnn_model import build_from_config as build_cnn
        from src.models.cnn_tcn_model import build_from_config as build_cnn_tcn
        cnn = build_cnn(DUMMY_CONFIG)
        cnn_tcn = build_cnn_tcn(DUMMY_CONFIG)
        # CNN-TCN should have more parameters due to TCN stack
        assert cnn_tcn.count_params() > cnn.count_params()


class TestBaselineModels:
    def setup_method(self):
        self.X = np.random.randn(N, 20).astype(np.float32)
        self.y = np.random.randint(0, 2, N)

    def test_rf_fits_and_predicts(self):
        from src.models.baselines import build_random_forest
        rf = build_random_forest(DUMMY_CONFIG)
        rf.fit(self.X, self.y)
        preds = rf.predict(self.X)
        probs = rf.predict_proba(self.X)
        assert preds.shape == (N,)
        assert probs.shape == (N, 2)

    def test_svm_fits_and_predicts(self):
        from src.models.baselines import build_svm
        svm = build_svm(DUMMY_CONFIG)
        svm.fit(self.X, self.y)
        preds = svm.predict(self.X)
        probs = svm.predict_proba(self.X)
        assert preds.shape == (N,)

    def test_dnn_fits_and_predicts(self):
        from src.models.baselines import build_dnn
        dnn = build_dnn(input_dim=20, config=DUMMY_CONFIG)
        preds = dnn.predict(self.X, verbose=0)
        assert preds.shape == (N, 1)


class TestEvaluator:
    def test_compute_metrics(self):
        from src.evaluation.evaluator import compute_classification_metrics
        y_true = np.array([0, 1, 0, 1, 1, 0])
        y_pred = np.array([0, 1, 0, 1, 0, 0])
        y_prob = np.array([0.1, 0.9, 0.2, 0.8, 0.4, 0.3])
        metrics = compute_classification_metrics(y_true, y_pred, y_prob)
        assert "accuracy" in metrics
        assert "f1" in metrics
        assert "auc_roc" in metrics
        assert 0.0 <= metrics["accuracy"] <= 1.0
        assert 0.0 <= metrics["fpr"] <= 1.0

    def test_mcnemar_identical_models(self):
        from src.evaluation.evaluator import mcnemar_test
        y_true = np.random.randint(0, 2, 100)
        y_pred = np.random.randint(0, 2, 100)
        result = mcnemar_test(y_true, y_pred, y_pred)
        # Identical predictions → not significant
        assert result["p_value"] == 1.0 or result["b01"] == result["b10"] == 0
