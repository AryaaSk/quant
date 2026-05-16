"""Tests for length_weighted pooling vs mean pooling in TextBlockBuilder.

All mocked. No network.
"""
from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd

from quant.state.text import TextBlockBuilder


class _MockVoyage:
    """Returns embedding[i] = unit_vector_along_axis_i for the i-th text."""
    def __init__(self, *args, **kwargs):
        pass

    def embed(self, texts: list[str]) -> np.ndarray:
        n = len(texts)
        out = np.zeros((n, 1024), dtype=np.float32)
        for i in range(n):
            out[i, i % 1024] = 1.0
        return out


def _make_two_doc_frame(short_text: str, long_text: str) -> pd.DataFrame:
    """Single timestamp row containing both texts joined by newline."""
    idx = pd.DatetimeIndex([pd.Timestamp("2024-01-01")])
    combined = short_text + "\n" + long_text
    return pd.DataFrame({"text__x": [combined]}, index=idx)


def test_length_weighted_pool_favors_longer_doc():
    """A 5000-char doc should contribute more weight to the pooled embedding than a 100-char one."""
    short = "x" * 100
    long = "y" * 5000
    df = _make_two_doc_frame(short, long)
    b = TextBlockBuilder(
        slot_width=1024, text_column="text__x", window_days=365,
        mode="voyage", pool_strategy="length_weighted", min_evidence_chars=10,
        use_pca=False,
    )
    with patch("quant.text.voyage_embedder.VoyageEmbedder", _MockVoyage):
        b.fit(df)
    out = b.build(pd.Timestamp("2024-06-01"), df)
    # short → axis 0, long → axis 1 (order they appear in texts list)
    # sqrt-length weights: sqrt(100)/sum vs sqrt(5000)/sum = 10 vs ~70.7
    # Normalized: short ≈ 10/80.7 ≈ 0.124, long ≈ 70.7/80.7 ≈ 0.876
    short_weight, long_weight = out[0], out[1]
    assert long_weight > short_weight, f"long weight ({long_weight:.4f}) should exceed short ({short_weight:.4f})"
    assert long_weight > 5 * short_weight, "5000-char doc should dominate 100-char doc significantly"


def test_mean_pool_treats_docs_equally():
    """In mean mode, a 5000-char doc gets exactly the same weight as a 100-char one."""
    short = "x" * 100
    long = "y" * 5000
    df = _make_two_doc_frame(short, long)
    b = TextBlockBuilder(
        slot_width=1024, text_column="text__x", window_days=365,
        mode="voyage", pool_strategy="mean", min_evidence_chars=10,
        use_pca=False,
    )
    with patch("quant.text.voyage_embedder.VoyageEmbedder", _MockVoyage):
        b.fit(df)
    out = b.build(pd.Timestamp("2024-06-01"), df)
    # Equal weights → each axis gets 0.5
    assert np.isclose(out[0], 0.5, atol=1e-5)
    assert np.isclose(out[1], 0.5, atol=1e-5)


def test_sqrt_weighting_is_sublinear():
    """sqrt(L) weighting is sub-linear: a 10x longer doc gets sqrt(10) ≈ 3.16x weight,
    NOT 10x weight. Prevents one giant doc from completely drowning out others."""
    short = "x" * 100
    medium = "y" * 1000  # 10x the short
    df = _make_two_doc_frame(short, medium)
    b = TextBlockBuilder(
        slot_width=1024, text_column="text__x", window_days=365,
        mode="voyage", pool_strategy="length_weighted", min_evidence_chars=10,
        use_pca=False,
    )
    with patch("quant.text.voyage_embedder.VoyageEmbedder", _MockVoyage):
        b.fit(df)
    out = b.build(pd.Timestamp("2024-06-01"), df)
    # sqrt(100)/sum = 10/40, sqrt(1000)/sum = ~31.6/40
    # short ≈ 0.24, medium ≈ 0.76, ratio = 3.16 (linear would be 10)
    short_w, med_w = out[0], out[1]
    ratio = med_w / short_w
    assert 2.5 < ratio < 3.8, f"sqrt-length ratio for 10x longer doc should be ~3.16, got {ratio:.2f}"
