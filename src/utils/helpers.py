"""
PhishNet - Helper Utilities
Sandesh Poudel | B01818884 | UWS MSc Cybersecurity
"""
import os
import yaml
import numpy as np
import pandas as pd
from typing import Dict, Any


def load_config(config_path: str = "configs/config.yaml") -> Dict[str, Any]:
    """Load YAML configuration file from cwd or project root."""
    if os.path.isabs(config_path):
        resolved_path = config_path
    else:
        # Support running commands from outside the project root.
        cwd_candidate = os.path.join(os.getcwd(), config_path)
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        project_candidate = os.path.join(project_root, config_path)
        resolved_path = cwd_candidate if os.path.exists(cwd_candidate) else project_candidate

    with open(resolved_path, "r") as f:
        return yaml.safe_load(f)


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility across numpy, Python, TensorFlow."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass


def save_metrics_to_excel(metrics_dict: Dict[str, Any], output_path: str) -> None:
    """
    Save all model metrics to an Excel spreadsheet.
    Produces the deliverable required by the project specification.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    rows = []
    for model_name, metrics in metrics_dict.items():
        row = {"Model": model_name}
        row.update(metrics)
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_excel(output_path, index=False)
    print(f"[INFO] Metrics saved to {output_path}")


def url_to_char_sequence(url: str, max_len: int = 200, vocab_size: int = 128) -> np.ndarray:
    """
    Convert a URL string to a fixed-length integer sequence.
    Uses ASCII ordinal encoding, clamped to vocab_size.

    Args:
        url: Raw URL string
        max_len: Pad/truncate to this length
        vocab_size: Character vocabulary size

    Returns:
        numpy array of shape (max_len,) with integer character codes
    """
    seq = [min(ord(c), vocab_size - 1) for c in url[:max_len]]
    # Pad with zeros to max_len
    seq += [0] * (max_len - len(seq))
    return np.array(seq, dtype=np.int32)


def batch_urls_to_sequences(urls: list, max_len: int = 200, vocab_size: int = 128) -> np.ndarray:
    """Convert a list of URLs to a 2D numpy array of character sequences."""
    return np.vstack([url_to_char_sequence(u, max_len, vocab_size) for u in urls])
