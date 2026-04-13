"""
PhishNet - API Request/Response Schemas
Sandesh Poudel | B01818884 | UWS MSc Cybersecurity
"""
from typing import List, Optional
from pydantic import BaseModel, Field, field_validator


class URLRequest(BaseModel):
    """Single URL classification request."""
    url: str = Field(
        ...,
        description="The URL to classify",
        json_schema_extra={
            "example": "http://paypa1-secure.login.verify.com/account/update"
        },
        min_length=4,
        max_length=2048,
    )
    model: Optional[str] = Field(
        default="cnn_tcn",
        description="Model to use: 'cnn', 'cnn_tcn', 'rf', 'svm', 'dnn'",
    )

    @field_validator("url")
    @classmethod
    def validate_url(cls, v: str) -> str:
        v = v.strip()
        if not (v.startswith("http://") or v.startswith("https://") or "." in v):
            raise ValueError("URL must contain a domain (e.g. example.com)")
        return v


class BatchURLRequest(BaseModel):
    """Batch URL classification request."""
    urls: List[str] = Field(
        ...,
        description="List of URLs to classify",
        min_length=1,
    )
    model: Optional[str] = Field(default="cnn_tcn")


class PredictionResult(BaseModel):
    """Single URL prediction result."""
    url: str
    label: str                   # "phishing" or "legitimate"
    confidence: float            # Probability of phishing (0.0 – 1.0)
    is_phishing: bool
    risk_level: str              # "LOW" | "MEDIUM" | "HIGH" | "CRITICAL"
    model_used: str
    latency_ms: float


class BatchPredictionResult(BaseModel):
    """Batch prediction response."""
    results: List[PredictionResult]
    total: int
    phishing_count: int
    legitimate_count: int
    avg_latency_ms: float


class HealthResponse(BaseModel):
    """API health check response."""
    status: str
    models_loaded: List[str]
    version: str


class MetricsResponse(BaseModel):
    """Model performance metrics response."""
    model_name: str
    accuracy: Optional[float]
    f1: Optional[float]
    auc_roc: Optional[float]
    latency_p50_ms: Optional[float]
    latency_p95_ms: Optional[float]
