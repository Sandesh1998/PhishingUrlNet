# PhishNet: Real-Time Phishing URL Detection with Hybrid Deep Learning
**Student:** Sandesh Poudel | **ID:** B01818884 | **UWS MSc Cybersecurity 2025/26**
**Supervisor:** Dr Manesh Thankappan

## Project Overview

PhishNet implements and compares **five models** for real-time phishing URL detection:

| Model | Type | Description |
|-------|------|-------------|
| **CNN** | Deep Learning | 1D CNN on character-level URL sequences |
| **CNN-TCN** ⭐ | Deep Learning (Novel) | Hybrid CNN + Temporal Convolutional Network |
| **Random Forest** | Traditional ML | Baseline ensemble classifier |
| **SVM** | Traditional ML | RBF kernel baseline |
| **DNN** | Deep Learning | Fully-connected baseline |

**Research targets:** >97% accuracy, <100ms inference latency, low false-positive rate.

---

## Quick Start

### Prerequisites
- Python 3.13+
- `pip` or a virtual environment manager
- Optional: Docker and Docker Compose

### Step 1 — Install dependencies
```bash
cd /home/kali/Documents/masterProject/PhishNet
source .venv/bin/activate
pip install -r requirements.txt
```

### Step 2 — Download datasets
```bash
python scripts/download_data.py
```
Downloads and prepares the raw datasets used by the project pipeline into `data/raw/`.

### Step 3 — Train all models
```bash
python scripts/train_all_models.py --models cnn cnn_tcn rf svm dnn
```
Or train specific models only:
```bash
python scripts/train_all_models.py --models cnn cnn_tcn
```

### Step 4 — Run full evaluation
```bash
python scripts/run_evaluation.py
```
Produces Excel spreadsheet, all plots, and JSON metrics in `results/`.

### Step 5 — Start API server
```bash
PYTHONPATH=/home/kali/Documents/masterProject/PhishNet python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8010
```
Open http://127.0.0.1:8010/ for the built-in URL checker or http://127.0.0.1:8010/docs for Swagger UI.

### Step 6 — Docker deployment
```bash
docker-compose -f docker/docker-compose.yml up --build
```

### Step 7 — Run tests
```bash
pytest tests/ -v
```

### Quick Health Check
```bash
curl http://127.0.0.1:8010/health
```
Expected response:
```json
{"status":"healthy","models_loaded":["cnn","cnn_tcn","rf","svm","dnn"],"version":"1.0.0"}
```

---

## Project Structure

```
PhishNet/
├── configs/config.yaml          # All hyperparameters
├── data/
│   ├── raw/                     # Downloaded CSVs (PhishTank, Tranco, legacy UCI reference data)
│   └── processed/               # Feature matrices + preprocessor
├── src/
│   ├── features/
│   │   ├── feature_extractor.py # 28 lexical + host-based + temporal features
│   │   └── preprocessor.py      # Imputation, scaling, SMOTE, train/val/test split
│   ├── models/
│   │   ├── cnn_model.py         # 1D CNN (Embedding → Conv blocks → Dense)
│   │   ├── cnn_tcn_model.py     # Hybrid CNN-TCN with dilated residual blocks
│   │   ├── baselines.py         # Random Forest, SVM, DNN
│   │   └── trainer.py           # Training loop + MLflow logging
│   ├── evaluation/
│   │   ├── evaluator.py         # All metrics + latency benchmarking + McNemar test
│   │   └── visualiser.py        # ROC, PR, confusion matrix, training history plots
│   ├── api/
│   │   ├── main.py              # FastAPI app
│   │   ├── schemas.py           # Pydantic request/response models
│   │   └── predictor.py         # Inference wrapper with model caching
│   └── utils/
│       ├── logger.py            # Logging setup
│       └── helpers.py           # Config loading, URL encoding utilities
├── scripts/
│   ├── download_data.py         # Dataset acquisition and cleaning
│   ├── train_all_models.py      # Full training pipeline
│   └── run_evaluation.py        # Full evaluation + export
├── tests/
│   ├── test_features.py         # Unit tests for feature extraction
│   ├── test_models.py           # Model architecture + evaluation tests
│   └── test_api.py              # API endpoint tests
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
└── results/                     # Auto-generated outputs
    ├── models/                  # Saved model weights (.keras, .pkl)
    ├── plots/                   # ROC, PR, confusion matrices, comparisons
    ├── metrics/                 # Excel spreadsheet + JSON
    └── logs/                    # Training logs
```

---

## CNN-TCN Architecture (Novel Contribution)

```
URL String
    ↓
Character Embedding (vocab=128, dim=64)
    ↓
CNN Block — Conv1D(128, k=3) → BN → ReLU → MaxPool → Dropout
    ↓  extracts local n-gram patterns ("paypal", "login", IP addresses)
Projection Conv — 128→64 channels
    ↓
TCN Block d=1  — Dilated Causal Conv × 2 + Residual
TCN Block d=2  — Dilated Causal Conv × 2 + Residual
TCN Block d=4  — Dilated Causal Conv × 2 + Residual
TCN Block d=8  — Dilated Causal Conv × 2 + Residual
TCN Block d=16 — Dilated Causal Conv × 2 + Residual
    ↓  models long-range sequential URL structure
GlobalAvgPool1D
    ↓
Dense(128) → BN → ReLU → Dropout(0.4)
Dense(64)  → BN → ReLU → Dropout(0.4)
    ↓
Sigmoid Output → P(phishing)
```

**Receptive field at d=16:** 2 × k × d = 2 × 3 × 16 = 96 characters per block.
Stacking 5 blocks gives effective RF >> URL length — the model sees the full URL context.

---

## Features Extracted (28 total)

| Category | Features |
|----------|----------|
| **Lexical (17)** | url_length, hostname_length, path_length, query_length, num_dots, num_hyphens, num_underscores, num_slashes, num_digits, num_special_chars, digit_ratio, uppercase_ratio, has_ip_address, num_subdomains, suspicious_keyword_count, url_entropy, path_entropy |
| **Host-based (7)** | has_dns_record, domain_age_days, is_https, domain_length, tld_in_common, whois_available, has_mx_record |
| **Temporal (4)** | domain_recently_registered, tld_is_new_gtld, url_has_date_pattern, campaign_novelty_score |

---

## Datasets

| Dataset | Role | Source |
|---------|------|--------|
| PhishTank | Phishing samples (train/test) | phishtank.org |
| Tranco Top-1M | Legitimate samples (train/test) | tranco-list.eu |
| UCI ML Repo | Reference dataset / legacy benchmark | archive.ics.uci.edu |

The API also includes a simple browser UI at `/` for quick URL checks, with a model selector and live prediction output.

---

## Evaluation Metrics

- **Classification:** Accuracy, Precision, Recall, F1, AUC-ROC, AUC-PR, FPR, FNR
- **Deployment:** Latency p50/p95 (ms), Throughput (URLs/sec), Peak memory (MB)
- **Statistical:** McNemar test (p < 0.05) for pairwise model significance
- **Generalisation:** Cross-dataset train/test splits for zero-day simulation

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/predict` | Single URL → label, confidence, risk level, latency |
| `POST` | `/predict/batch` | Batch up to 500 URLs |
| `GET`  | `/health` | API status + loaded models |
| `GET`  | `/metrics` | Stored evaluation metrics |
| `GET`  | `/` | Built-in browser UI |
| `GET`  | `/docs` | Swagger UI |

### Example
```bash
curl -X POST http://127.0.0.1:8010/predict \
  -H "Content-Type: application/json" \
  -d '{"url": "http://paypa1-secure.login.verify.xyz/account"}'
```

Response:
```json
{
  "url": "http://paypa1-secure.login.verify.xyz/account",
  "label": "phishing",
  "confidence": 0.9873,
  "is_phishing": true,
  "risk_level": "CRITICAL",
  "model_used": "cnn_tcn",
  "latency_ms": 12.4
}
```

