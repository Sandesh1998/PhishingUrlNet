"""
PhishNet - FastAPI Application
Sandesh Poudel | B01818884 | UWS MSc Cybersecurity

RESTful API exposing PhishNet phishing detection.
Endpoints:
  POST /predict        - Single URL classification
  POST /predict/batch  - Batch URL classification
  GET  /health         - API health and loaded models
  GET  /metrics        - Model performance metrics
  GET  /               - Built-in web UI
"""

import json
import os
from contextlib import asynccontextmanager
from typing import Optional
from urllib.parse import urlparse

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse

from src.api.predictor import PhishNetPredictor, _confidence_to_risk
from src.api.schemas import (
    BatchPredictionResult,
    BatchURLRequest,
    HealthResponse,
  PredictionResult,
  URLRequest,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Global predictor instance.
predictor = PhishNetPredictor(model_dir="results/models")

TRUSTED_DOMAINS = {
  "google.com", "facebook.com", "youtube.com", "wikipedia.org", "github.com",
  "microsoft.com", "apple.com", "amazon.com", "linkedin.com", "bbc.com",
  "reddit.com", "instagram.com", "netflix.com", "python.org", "pypi.org",
}

# Demo-focused thresholds. CNN shows the best separation in current checkpoints,
# so we use a stricter cutoff to reduce obvious false positives.
MODEL_PHISHING_THRESHOLDS = {
  "cnn": 0.90,
  "cnn_tcn": 0.50,
  "rf": 0.50,
  "svm": 0.50,
  "dnn": 0.50,
}


def _is_trusted_domain(url: str) -> bool:
  """Return True for trusted root domains and their subdomains."""
  try:
    host = (urlparse(url if "//" in url else f"https://{url}").hostname or "").lower()
  except Exception:
    return False
  for dom in TRUSTED_DOMAINS:
    if host == dom or host.endswith("." + dom):
      return True
  return False


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models at startup, clean up on shutdown."""
    logger.info("PhishNet API starting up - loading models...")
    predictor.load_all()
    logger.info(f"Models loaded: {predictor.loaded_models}")
    yield
    logger.info("PhishNet API shutting down.")


app = FastAPI(
    title="PhishNet API",
    description=(
        "Real-time phishing URL detection using CNN and CNN-TCN deep learning models. "
        "MSc Cybersecurity project - Sandesh Poudel (B01818884), UWS."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _select_model(requested_model: Optional[str]) -> str:
    """Pick requested model, or best available default."""
    if not predictor.loaded_models:
        raise HTTPException(
            status_code=503,
            detail="No models loaded. Run training pipeline first.",
        )

    if requested_model:
        if requested_model not in predictor.loaded_models:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Model '{requested_model}' not available. "
                    f"Loaded: {predictor.loaded_models}"
                ),
            )
        return requested_model

    for candidate in ("cnn_tcn", "cnn", "rf", "svm", "dnn"):
        if candidate in predictor.loaded_models:
            return candidate

    return predictor.loaded_models[0]


@app.get("/", response_class=HTMLResponse, tags=["UI"])
async def home_ui():
    """Built-in frontend for quick URL checks."""
    options = "\n".join(
        [f'<option value="{m}">{m}</option>' for m in predictor.loaded_models]
    )
    if not options:
        options = '<option value="">No model loaded</option>'

    html_template = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>PhishNet URL Checker</title>
  <style>
    :root {
      --bg: #f4f8f5;
      --panel: #ffffff;
      --text: #132218;
      --muted: #5c6d61;
      --accent: #1f7a4d;
      --danger: #af2f2f;
      --ok: #1f7a4d;
      --border: #d7e4db;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif;
      color: var(--text);
      background: radial-gradient(circle at 10% 20%, #e7f2ea, var(--bg));
      min-height: 100vh;
      display: grid;
      place-items: center;
      padding: 20px;
    }
    .card {
      width: min(760px, 100%);
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 14px;
      box-shadow: 0 12px 32px rgba(0, 0, 0, 0.06);
      padding: 22px;
    }
    h1 { margin: 0 0 6px; font-size: 1.5rem; }
    p { margin: 0 0 14px; color: var(--muted); }
    label { display: block; margin-bottom: 6px; font-weight: 600; }
    input, select, button {
      width: 100%;
      padding: 12px;
      border-radius: 10px;
      border: 1px solid var(--border);
      font-size: 0.98rem;
    }
    .row { display: grid; grid-template-columns: 1fr 170px; gap: 10px; }
    button {
      margin-top: 12px;
      background: var(--accent);
      color: #fff;
      border: none;
      cursor: pointer;
      font-weight: 600;
    }
    button:disabled { opacity: 0.6; cursor: not-allowed; }
    .result {
      margin-top: 16px;
      border: 1px dashed var(--border);
      border-radius: 10px;
      padding: 14px;
      background: #fbfdfb;
      display: none;
    }
    .badge {
      display: inline-block;
      padding: 4px 10px;
      border-radius: 999px;
      font-weight: 700;
      font-size: 0.85rem;
      color: #fff;
    }
    .phishing { background: var(--danger); }
    .legitimate { background: var(--ok); }
    .kv { margin-top: 8px; color: var(--muted); }
    .error { color: var(--danger); margin-top: 8px; min-height: 18px; }
  </style>
</head>
<body>
  <main class="card">
    <h1>PhishNet URL Checker</h1>
    <p>Enter a URL and check whether the model predicts phishing or legitimate.</p>

    <label for="url">URL</label>
    <div class="row">
      <input id="url" type="text" placeholder="https://www.google.com" />
      <select id="model">__MODEL_OPTIONS__</select>
    </div>
    <button id="checkBtn">Check URL</button>
    <div id="error" class="error"></div>

    <section id="result" class="result">
      <div><strong>Result:</strong> <span id="labelBadge" class="badge legitimate">-</span></div>
      <div class="kv">Confidence: <span id="confidence">-</span></div>
      <div class="kv">Risk Level: <span id="risk">-</span></div>
      <div class="kv">Model Used: <span id="modelUsed">-</span></div>
      <div class="kv">Latency: <span id="latency">-</span> ms</div>
    </section>
  </main>

  <script>
    const btn = document.getElementById("checkBtn");
    const urlInput = document.getElementById("url");
    const modelInput = document.getElementById("model");
    const errorEl = document.getElementById("error");
    const resultEl = document.getElementById("result");
    const labelBadge = document.getElementById("labelBadge");

    btn.addEventListener("click", async () => {
      const url = urlInput.value.trim();
      const model = modelInput.value;
      errorEl.textContent = "";
      resultEl.style.display = "none";

      if (!url) {
        errorEl.textContent = "Please enter a URL.";
        return;
      }

      btn.disabled = true;
      btn.textContent = "Checking...";

      try {
        const payload = model ? { url, model } : { url };
        const res = await fetch("/predict", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        });

        const data = await res.json();
        if (!res.ok) {
          throw new Error(data.detail || "Prediction failed");
        }

        labelBadge.textContent = data.label.toUpperCase();
        labelBadge.className = "badge " + (data.is_phishing ? "phishing" : "legitimate");
        document.getElementById("confidence").textContent = data.confidence;
        document.getElementById("risk").textContent = data.risk_level;
        document.getElementById("modelUsed").textContent = data.model_used;
        document.getElementById("latency").textContent = data.latency_ms;
        resultEl.style.display = "block";
      } catch (err) {
        errorEl.textContent = err.message;
      } finally {
        btn.disabled = false;
        btn.textContent = "Check URL";
      }
    });
  </script>
</body>
</html>
"""
    return HTMLResponse(content=html_template.replace("__MODEL_OPTIONS__", options))


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """API health check."""
    return HealthResponse(
        status="healthy" if predictor.loaded_models else "degraded",
        models_loaded=predictor.loaded_models,
        version="1.0.0",
    )


@app.post("/predict", response_model=PredictionResult, tags=["Detection"])
async def predict_single(request: URLRequest):
    """Classify one URL as phishing or legitimate."""
    model_name = _select_model(request.model)

    try:
        confidence, latency_ms = predictor.predict(request.url, model_name)
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    threshold = MODEL_PHISHING_THRESHOLDS.get(model_name, 0.5)
    is_phishing = confidence >= threshold

    # Guardrail for known trustworthy domains to reduce harmful false positives.
    if _is_trusted_domain(request.url) and confidence < 0.999999:
      is_phishing = False

    return PredictionResult(
        url=request.url,
        label="phishing" if is_phishing else "legitimate",
        confidence=round(confidence, 6),
        is_phishing=is_phishing,
        risk_level=_confidence_to_risk(confidence),
        model_used=model_name,
        latency_ms=round(latency_ms, 3),
    )


@app.post("/predict/batch", response_model=BatchPredictionResult, tags=["Detection"])
async def predict_batch(request: BatchURLRequest):
    """Classify up to 500 URLs in one request."""
    model_name = _select_model(request.model)
    urls = request.urls

    if len(urls) > 500:
        raise HTTPException(status_code=400, detail="Maximum 500 URLs per batch.")

    try:
        confidences, avg_latency = predictor.predict_batch(urls, model_name)
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    threshold = MODEL_PHISHING_THRESHOLDS.get(model_name, 0.5)
    results = []
    for url, confidence in zip(urls, confidences):
        is_phishing = bool(confidence >= threshold)
        if _is_trusted_domain(url) and float(confidence) < 0.999999:
            is_phishing = False
        results.append(
            PredictionResult(
                url=url,
                label="phishing" if is_phishing else "legitimate",
                confidence=round(float(confidence), 6),
                is_phishing=is_phishing,
                risk_level=_confidence_to_risk(float(confidence)),
                model_used=model_name,
                latency_ms=round(avg_latency, 3),
            )
        )

    phishing_count = sum(1 for r in results if r.is_phishing)
    return BatchPredictionResult(
        results=results,
        total=len(results),
        phishing_count=phishing_count,
        legitimate_count=len(results) - phishing_count,
        avg_latency_ms=round(avg_latency, 3),
    )


@app.get("/metrics", tags=["System"])
async def get_metrics():
    """Return stored evaluation metrics."""
    metrics_path = "results/metrics/phishnet_results.json"
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            return JSONResponse(content=json.load(f))
    return JSONResponse(
        content={"message": "No metrics file found. Run run_evaluation.py first."},
        status_code=404,
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(status_code=500, content={"detail": "Internal server error."})
