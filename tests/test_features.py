"""
PhishNet - Feature Extraction Tests
Sandesh Poudel | B01818884 | UWS MSc Cybersecurity
"""
import pytest
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features.feature_extractor import (
    extract_lexical_features,
    extract_temporal_features,
    extract_all_features,
)
from src.utils.helpers import url_to_char_sequence, batch_urls_to_sequences


PHISHING_URL = "http://paypa1-secure.login.verify-account.xyz/update?confirm=1"
LEGIT_URL    = "https://www.google.com"


class TestLexicalFeatures:
    def test_returns_dict(self):
        feats = extract_lexical_features(LEGIT_URL)
        assert isinstance(feats, dict)
        assert len(feats) > 0

    def test_feature_keys(self):
        feats = extract_lexical_features(LEGIT_URL)
        expected = ["url_length", "num_dots", "num_hyphens", "url_entropy",
                    "has_ip_address", "suspicious_keyword_count"]
        for key in expected:
            assert key in feats, f"Missing key: {key}"

    def test_phishing_higher_entropy(self):
        ph = extract_lexical_features(PHISHING_URL)
        lg = extract_lexical_features(LEGIT_URL)
        assert ph["url_length"] > lg["url_length"]

    def test_phishing_has_keywords(self):
        feats = extract_lexical_features(PHISHING_URL)
        assert feats["suspicious_keyword_count"] > 0

    def test_ip_detection(self):
        ip_url = "http://192.168.1.1/login"
        feats = extract_lexical_features(ip_url)
        assert feats["has_ip_address"] == 1

    def test_no_ip_for_domain(self):
        feats = extract_lexical_features(LEGIT_URL)
        assert feats["has_ip_address"] == 0

    def test_all_values_numeric(self):
        feats = extract_lexical_features(PHISHING_URL)
        for k, v in feats.items():
            assert isinstance(v, (int, float)), f"Non-numeric value for {k}: {v}"


class TestTemporalFeatures:
    def test_new_gtld_detected(self):
        feats = extract_temporal_features("http://example.xyz")
        assert feats["tld_is_new_gtld"] == 1

    def test_common_tld_not_new(self):
        feats = extract_temporal_features("http://example.com")
        assert feats["tld_is_new_gtld"] == 0

    def test_novelty_score_range(self):
        feats = extract_temporal_features(PHISHING_URL)
        assert 0.0 <= feats["campaign_novelty_score"] <= 1.0


class TestURLSequenceEncoding:
    def test_output_shape(self):
        seq = url_to_char_sequence(LEGIT_URL, max_len=200)
        assert seq.shape == (200,)

    def test_padding(self):
        seq = url_to_char_sequence("a.b", max_len=10)
        assert len(seq) == 10
        assert seq[-1] == 0   # padded with zeros

    def test_truncation(self):
        long_url = "http://" + "a" * 300 + ".com"
        seq = url_to_char_sequence(long_url, max_len=200)
        assert len(seq) == 200

    def test_batch_encoding(self):
        urls = [LEGIT_URL, PHISHING_URL, "http://test.org"]
        batch = batch_urls_to_sequences(urls, max_len=200)
        assert batch.shape == (3, 200)


class TestCombinedExtractor:
    def test_combine_returns_all_categories(self):
        feats = extract_all_features(LEGIT_URL, include_host=False)
        assert "url_length" in feats           # lexical
        assert "campaign_novelty_score" in feats  # temporal
