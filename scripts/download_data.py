"""
PhishNet - Dataset Acquisition Script
Sandesh Poudel | B01818884 | UWS MSc Cybersecurity

Downloads and prepares the four datasets specified in the project:
  1. PhishTank  - community-verified phishing URLs
  2. UNB (CIC)  - balanced legitimate/phishing with feature annotations
  3. UCI ML Repo - phishing websites dataset (benchmark)
  4. Tranco Top Sites - legitimate URLs (replaces Alexa, now discontinued)

Usage:
    python scripts/download_data.py

All raw files are saved to data/raw/.
Run preprocessor pipeline afterwards (scripts/train_all_models.py).
"""
import os
import sys
import zipfile
import time
import requests
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.logger import get_logger
from src.utils.helpers import load_config

logger = get_logger(__name__, log_dir="results/logs")
config = load_config()
RAW_DIR = config["paths"]["data_raw"]
os.makedirs(RAW_DIR, exist_ok=True)


def _download(url: str, dest: str, desc: str = "", force: bool = False) -> None:
    """Download a file with a progress bar."""
    if os.path.exists(dest) and not force:
        logger.info(f"Already exists, skipping: {dest}")
        return
    logger.info(f"Downloading: {url}")
    resp = requests.get(url, stream=True, timeout=60)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))
    with open(dest, "wb") as f, tqdm(total=total, unit="B", unit_scale=True,
                                      desc=desc, ncols=80) as bar:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
            bar.update(len(chunk))
    logger.info(f"Saved: {dest}")


def download_phishtank() -> pd.DataFrame:
    """
    PhishTank community-verified phishing URLs.
    Free download — no API key required for the CSV dump.
    Label: 1 (phishing)
    """
    dest = os.path.join(RAW_DIR, "phishtank.csv")
    url = "https://data.phishtank.com/data/online-valid.csv"
    try:
        # Keep PhishTank fresh for better real-world behavior.
        force_refresh = False
        if os.path.exists(dest):
            age_hours = (time.time() - os.path.getmtime(dest)) / 3600
            force_refresh = age_hours > 24
            if force_refresh:
                logger.info(f"PhishTank file is {age_hours:.1f}h old, refreshing...")
        _download(url, dest, desc="PhishTank", force=force_refresh)
        df = pd.read_csv(dest, usecols=["url", "verified"])
        df = df[df["verified"] == "yes"][["url"]].copy()
        df["label"] = 1
        df.columns = ["url", "label"]
        logger.info(f"PhishTank: {len(df):,} verified phishing URLs")
        return df
    except Exception as e:
        logger.warning(f"PhishTank refresh failed: {e}")

        # Use cached phishtank file if available.
        if os.path.exists(dest):
            try:
                df = pd.read_csv(dest, usecols=["url", "verified"])
                df = df[df["verified"] == "yes"][['url']].copy()
                df["label"] = 1
                df.columns = ["url", "label"]
                logger.info(f"Using cached PhishTank: {len(df):,} verified phishing URLs")
                return df
            except Exception as cache_err:
                logger.warning(f"Cached PhishTank parse failed: {cache_err}")

        # Backup source: OpenPhish feed.
        openphish_dest = os.path.join(RAW_DIR, "openphish.txt")
        openphish_url = "https://openphish.com/feed.txt"
        try:
            _download(openphish_url, openphish_dest, desc="OpenPhish")
            urls = pd.read_csv(openphish_dest, header=None, names=["url"])
            urls = urls.dropna().drop_duplicates()
            urls["label"] = 1
            logger.info(f"OpenPhish: {len(urls):,} phishing URLs")
            return urls[["url", "label"]]
        except Exception as openphish_err:
            logger.warning(f"OpenPhish download failed: {openphish_err}. Using synthetic fallback.")
            return _synthetic_phishing_fallback(1000)


def download_uci_phishing() -> pd.DataFrame:
    """
    UCI ML Repository – Website Phishing Dataset.
    Contains 30 features + target label (phishing=1, legitimate=-1/0).
    We load the ARFF and extract URL + label.
    """
    dest = os.path.join(RAW_DIR, "uci_phishing.arff")
    url = ("https://archive.ics.uci.edu/ml/machine-learning-databases/"
           "00327/Training%20Dataset.arff")
    try:
        _download(url, dest, desc="UCI")
        # UCI has no raw URLs. Using synthetic URL placeholders introduces
        # label noise and harms URL classifiers, so we keep UCI for download
        # completeness but do not inject it into combined URL training data.
        logger.info("UCI downloaded (feature dataset). Skipping merge into URL dataset.")
        return pd.DataFrame(columns=["url", "label"])
    except Exception as e:
        logger.warning(f"UCI download failed: {e}. Using synthetic fallback.")
        return _build_synthetic_dataset(2000)


def download_tranco_legit(n: int = 50_000) -> pd.DataFrame:
    """
    Tranco top-1M legitimate websites (replacement for discontinued Alexa list).
    We use the top-N as negative (legitimate) samples.
    Label: 0 (legitimate)
    """
    dest = os.path.join(RAW_DIR, "tranco_1m.csv.zip")
    url = "https://tranco-list.eu/top-1m.csv.zip"
    try:
        _download(url, dest, desc="Tranco")
        with zipfile.ZipFile(dest, "r") as zf:
            with zf.open("top-1m.csv") as f:
                df = pd.read_csv(f, header=None, names=["rank", "domain"],
                                 nrows=n)
        df["url"] = "https://" + df["domain"]
        df["label"] = 0
        df = df[["url", "label"]]
        logger.info(f"Tranco: {len(df):,} legitimate URLs")
        return df
    except Exception as e:
        logger.warning(f"Tranco download failed: {e}. Using synthetic fallback.")
        return _synthetic_legit_fallback(n)


def _synthetic_phishing_fallback(n: int = 5000) -> pd.DataFrame:
    """Generate synthetic phishing URLs for offline testing."""
    import random, string
    tlds = [".xyz", ".tk", ".ml", ".gq", ".cf", ".top"]
    paths = ["/login", "/verify", "/account", "/update", "/secure"]
    urls = []
    for _ in range(n):
        rand = "".join(random.choices(string.ascii_lowercase, k=10))
        tld = random.choice(tlds)
        path = random.choice(paths)
        urls.append(f"http://paypal-{rand}{tld}{path}")
    return pd.DataFrame({"url": urls, "label": 1})


def _synthetic_legit_fallback(n: int = 5000) -> pd.DataFrame:
    """Generate synthetic legitimate URLs for offline testing."""
    import random
    domains = ["google.com", "bbc.co.uk", "microsoft.com", "amazon.com",
               "wikipedia.org", "github.com", "stackoverflow.com", "nhs.uk"]
    paths = ["", "/about", "/news", "/products", "/search?q=test"]
    urls = [f"https://{random.choice(domains)}{random.choice(paths)}"
            for _ in range(n)]
    return pd.DataFrame({"url": urls, "label": 0})


def _build_synthetic_dataset(n: int = 2000) -> pd.DataFrame:
    ph = _synthetic_phishing_fallback(n // 2)
    leg = _synthetic_legit_fallback(n // 2)
    return pd.concat([ph, leg], ignore_index=True)


def _normalize_url(url: str) -> str:
    if not isinstance(url, str):
        return ""
    url = url.strip()
    if not url:
        return ""
    if not (url.startswith("http://") or url.startswith("https://")):
        url = "https://" + url
    return url


def _clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Remove malformed or synthetic-noise rows from URL datasets."""
    if df.empty:
        return df

    out = df.copy()
    out["url"] = out["url"].astype(str).map(_normalize_url)
    out = out[out["url"].str.len() > 7]

    # Drop synthetic placeholders used by older pipelines.
    out = out[~out["url"].str.contains(r"example-\d+\.com", regex=True, na=False)]
    out = out[~out["url"].str.contains(r"localhost|127\.0\.0\.1", regex=True, na=False)]

    # Keep http(s)-only URLs with host part.
    def valid(u: str) -> bool:
        try:
            p = urlparse(u)
            return p.scheme in ("http", "https") and bool(p.netloc)
        except Exception:
            return False

    out = out[out["url"].map(valid)]
    out = out.drop_duplicates(subset="url")
    out["label"] = out["label"].astype(int)
    return out.reset_index(drop=True)


def _trusted_anchor_urls() -> pd.DataFrame:
    """A small set of high-confidence legitimate URLs for calibration stability."""
    trusted = [
        "https://www.google.com",
        "https://www.facebook.com/?locale=en_GB",
        "https://www.youtube.com",
        "https://www.wikipedia.org",
        "https://www.github.com",
        "https://www.microsoft.com",
        "https://www.apple.com",
        "https://www.amazon.com",
        "https://www.linkedin.com",
        "https://www.bbc.com/news",
        "https://www.reddit.com",
        "https://www.instagram.com",
        "https://www.netflix.com",
        "https://docs.python.org",
        "https://pypi.org",
    ]
    return pd.DataFrame({"url": trusted, "label": 0})


def build_combined_dataset() -> pd.DataFrame:
    """
    Combine all datasets into a single DataFrame.
    Saves combined_urls.csv and per-source files to data/raw/.
    """
    logger.info("=== PhishNet Dataset Acquisition ===")

    phishtank_df = download_phishtank()
    tranco_df    = download_tranco_legit(n=len(phishtank_df))   # balanced
    uci_df       = download_uci_phishing()

    anchors_df   = _trusted_anchor_urls()

    # Combine and clean
    combined = pd.concat([phishtank_df, tranco_df, uci_df, anchors_df], ignore_index=True)
    combined = _clean_dataset(combined)
    combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)

    # Save
    out = os.path.join(RAW_DIR, "combined_urls.csv")
    combined.to_csv(out, index=False)
    logger.info(
        f"Combined dataset: {len(combined):,} URLs | "
        f"phishing={combined.label.sum():,} | "
        f"legitimate={(combined.label == 0).sum():,}"
    )
    logger.info(f"Saved → {out}")
    return combined


if __name__ == "__main__":
    df = build_combined_dataset()
    print(f"\n{'='*50}")
    print(f"Dataset ready: {len(df):,} total URLs")
    print(df["label"].value_counts().rename({0: "Legitimate", 1: "Phishing"}).to_string())
    print(f"{'='*50}")
