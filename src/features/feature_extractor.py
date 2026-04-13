"""
PhishNet - Feature Extractor
Sandesh Poudel | B01818884 | UWS MSc Cybersecurity

Extracts three categories of features from URLs as specified in the project:
  1. Lexical features  - derived purely from the URL string
  2. Host-based features - DNS, WHOIS, certificate signals
  3. Temporal features - domain age, campaign timing signals

Reference: Chiew et al. (2019) multi-category feature engineering approach.
"""
import re
import math
import socket
import string
from urllib.parse import urlparse
from typing import Dict, Optional
from datetime import datetime

import numpy as np
import tldextract

from src.utils.logger import get_logger

logger = get_logger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# 1. LEXICAL FEATURES
# ─────────────────────────────────────────────────────────────────────────────

SUSPICIOUS_KEYWORDS = [
    "login", "signin", "verify", "account", "update", "secure", "banking",
    "paypal", "ebay", "amazon", "apple", "google", "microsoft", "confirm",
    "password", "credential", "wallet", "free", "prize", "winner", "urgent",
    "suspend", "unusual", "activity", "click", "here", "limited", "offer",
]


def _entropy(s: str) -> float:
    """Shannon entropy of a string — high entropy often signals obfuscated domains."""
    if not s:
        return 0.0
    freq = {c: s.count(c) / len(s) for c in set(s)}
    return -sum(p * math.log2(p) for p in freq.values())


def extract_lexical_features(url: str) -> Dict[str, float]:
    """
    Extract lexical features from the raw URL string.

    Features (17 total):
        url_length, hostname_length, path_length, query_length,
        num_dots, num_hyphens, num_underscores, num_slashes,
        num_digits, num_special_chars, digit_ratio, uppercase_ratio,
        has_ip_address, num_subdomains, suspicious_keyword_count,
        url_entropy, path_entropy
    """
    parsed = urlparse(url if url.startswith("http") else "http://" + url)
    hostname = parsed.hostname or ""
    path = parsed.path or ""
    query = parsed.query or ""
    ext = tldextract.extract(url)
    subdomain = ext.subdomain

    # IP address in hostname
    ip_pattern = re.compile(
        r"^(\d{1,3}\.){3}\d{1,3}$"
    )
    has_ip = 1 if ip_pattern.match(hostname) else 0

    # Suspicious keyword count
    url_lower = url.lower()
    kw_count = sum(1 for kw in SUSPICIOUS_KEYWORDS if kw in url_lower)

    # Subdomains (split on dots, minus TLD parts)
    sub_parts = subdomain.split(".") if subdomain else []
    num_subdomains = len(sub_parts)

    features = {
        "url_length": len(url),
        "hostname_length": len(hostname),
        "path_length": len(path),
        "query_length": len(query),
        "num_dots": url.count("."),
        "num_hyphens": url.count("-"),
        "num_underscores": url.count("_"),
        "num_slashes": url.count("/"),
        "num_digits": sum(c.isdigit() for c in url),
        "num_special_chars": sum(c in "@!#$%^&*()+=[]{}|;:,<>?" for c in url),
        "digit_ratio": sum(c.isdigit() for c in url) / max(len(url), 1),
        "uppercase_ratio": sum(c.isupper() for c in url) / max(len(url), 1),
        "has_ip_address": has_ip,
        "num_subdomains": num_subdomains,
        "suspicious_keyword_count": kw_count,
        "url_entropy": _entropy(url),
        "path_entropy": _entropy(path),
    }
    return features


# ─────────────────────────────────────────────────────────────────────────────
# 2. HOST-BASED FEATURES
# ─────────────────────────────────────────────────────────────────────────────

def extract_host_features(url: str, timeout: int = 3) -> Dict[str, float]:
    """
    Extract host-based features using DNS and WHOIS lookups.

    Features (7 total):
        has_dns_record, domain_age_days, is_https,
        domain_length, tld_in_alexa_top, whois_available, has_mx_record

    NOTE: DNS/WHOIS lookups add latency. In production the API caches
    these results. For offline feature engineering (training) this is
    acceptable. Gracefully degrades to -1 sentinel if lookup fails.
    """
    ext = tldextract.extract(url)
    domain = f"{ext.domain}.{ext.suffix}" if ext.suffix else ext.domain

    # DNS record check
    has_dns = 0
    try:
        socket.setdefaulttimeout(timeout)
        socket.gethostbyname(domain)
        has_dns = 1
    except Exception:
        has_dns = 0

    # HTTPS protocol
    is_https = 1 if url.lower().startswith("https") else 0

    # Domain age via WHOIS
    domain_age_days = -1
    whois_available = 0
    try:
        import whois as pywhois
        w = pywhois.whois(domain)
        whois_available = 1
        creation_date = w.creation_date
        if isinstance(creation_date, list):
            creation_date = creation_date[0]
        if creation_date:
            domain_age_days = (datetime.now() - creation_date).days
    except Exception:
        pass

    # MX record (email phishing indicator — phishing domains rarely have MX)
    has_mx = 0
    try:
        import dns.resolver
        dns.resolver.resolve(domain, "MX", lifetime=timeout)
        has_mx = 1
    except Exception:
        pass

    # Known benign TLDs
    COMMON_TLDS = {
        "com", "org", "net", "edu", "gov", "uk", "us", "ca", "au", "de", "fr"
    }
    tld_benign = 1 if ext.suffix in COMMON_TLDS else 0

    features = {
        "has_dns_record": has_dns,
        "domain_age_days": domain_age_days,
        "is_https": is_https,
        "domain_length": len(ext.domain),
        "tld_in_common": tld_benign,
        "whois_available": whois_available,
        "has_mx_record": has_mx,
    }
    return features


# ─────────────────────────────────────────────────────────────────────────────
# 3. TEMPORAL FEATURES
# ─────────────────────────────────────────────────────────────────────────────

def extract_temporal_features(url: str,
                               domain_age_days: Optional[float] = None) -> Dict[str, float]:
    """
    Extract temporal features indicating campaign-level patterns.

    Features (4 total):
        domain_recently_registered, tld_is_new_gtld,
        url_has_date_pattern, campaign_novelty_score

    Reference: Alorvor & Dadkhah (2025) TCN-driven URL sequence modelling.
    """
    # Recently registered domains are a strong phishing signal
    recently_registered = 0
    if domain_age_days is not None and domain_age_days >= 0:
        recently_registered = 1 if domain_age_days < 90 else 0
    elif domain_age_days == -1:
        # Unknown age — treat as suspicious (many phishing domains hide WHOIS)
        recently_registered = 1

    # New gTLDs commonly abused in phishing campaigns
    NEW_GTLDS = {
        "xyz", "top", "club", "online", "site", "tech", "info", "biz",
        "win", "click", "download", "stream", "gq", "ml", "cf", "ga", "tk"
    }
    ext = tldextract.extract(url)
    is_new_gtld = 1 if ext.suffix in NEW_GTLDS else 0

    # Date-like patterns embedded in URL (phishing campaign indicator)
    date_pattern = re.compile(
        r"\d{4}[-/]\d{2}[-/]\d{2}|"
        r"\d{2}[-/]\d{2}[-/]\d{4}|"
        r"20\d{2}"
    )
    has_date = 1 if date_pattern.search(url) else 0

    # Novelty score: combine signals into a 0–1 risk score
    novelty_score = (recently_registered + is_new_gtld + has_date) / 3.0

    features = {
        "domain_recently_registered": recently_registered,
        "tld_is_new_gtld": is_new_gtld,
        "url_has_date_pattern": has_date,
        "campaign_novelty_score": novelty_score,
    }
    return features


# ─────────────────────────────────────────────────────────────────────────────
# COMBINED EXTRACTOR
# ─────────────────────────────────────────────────────────────────────────────

def extract_all_features(url: str, include_host: bool = True) -> Dict[str, float]:
    """
    Extract all 28 features from a URL.

    Args:
        url: The URL string
        include_host: Whether to perform DNS/WHOIS lookups (slower, set False for batch offline)

    Returns:
        Dictionary of feature_name -> float value
    """
    features = {}
    features.update(extract_lexical_features(url))

    if include_host:
        host_feats = extract_host_features(url)
        features.update(host_feats)
        features.update(
            extract_temporal_features(url, domain_age_days=host_feats.get("domain_age_days"))
        )
    else:
        features.update(extract_temporal_features(url))

    return features


FEATURE_NAMES = (
    list(extract_lexical_features("http://example.com").keys()) +
    list(extract_host_features.__doc__.split("Features (7 total):\n")[0].splitlines()) +
    list(extract_temporal_features.__doc__.split("Features (4 total):\n")[0].splitlines())
)
