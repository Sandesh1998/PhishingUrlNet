"""
PhishNet - Logger Utility
Sandesh Poudel | B01818884 | UWS MSc Cybersecurity
"""
import logging
import os
from datetime import datetime


def get_logger(name: str, log_dir: str = "results/logs") -> logging.Logger:
    """
    Create and return a configured logger.

    Args:
        name: Logger name (typically __name__ of calling module)
        log_dir: Directory to write log files

    Returns:
        Configured Python logger
    """
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    if logger.handlers:
        return logger

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch_fmt = logging.Formatter("[%(asctime)s] %(levelname)s - %(name)s - %(message)s",
                               datefmt="%H:%M:%S")
    ch.setFormatter(ch_fmt)

    # File handler
    timestamp = datetime.now().strftime("%Y%m%d")
    fh = logging.FileHandler(os.path.join(log_dir, f"phishnet_{timestamp}.log"))
    fh.setLevel(logging.DEBUG)
    fh_fmt = logging.Formatter(
        "[%(asctime)s] %(levelname)s - %(name)s:%(lineno)d - %(message)s"
    )
    fh.setFormatter(fh_fmt)

    logger.addHandler(ch)
    logger.addHandler(fh)

    return logger
