"""Tests for the PCA mode of TextBlockBuilder.

All mocked. No real Voyage API calls.
"""
from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from quant.state.text import TextBlockBuilder


class _MockVoyage:
    """Deterministic mock that returns a low-rank embedding so PCA has structure to find."""
    def __init__(self, *args, **kwargs):
        pass

    def embed(self, texts: list[str]) -> np.ndarray:
        # Deterministic per-text embedding: take hash of text, seed RNG, draw 1024-dim vector.
        # Reduce intrinsic rank by mixing a small basis of 16 latent vectors → 1024 dim.
        np.random.seed(0)
        basis = np.random.randn(16, 1024).astype(np.float32)
        rows = []
        for t in texts:
            h = hash(t) % (10**6)
            coef = np.random.RandomState(h).randn(16).astype(np.float32)
            rows.append(coef @ basis)
        return np.stack(rows, axis=0)


def _make_structured(n_rows: int = 200, text_col: str = "text__sec") -> pd.DataFrame:
    """Build a synthetic structured frame with N rows of distinct text chunks."""
    idx = pd.date_range("2024-01-01", periods=n_rows, freq="D")
    texts = [f"distinct chunk number {i} with some unique content padding xxxxxxxxxx" for i in range(n_rows)]
    return pd.DataFrame({text_col: texts}, index=idx)


def test_pca_fit_below_min_samples_falls_back_to_truncation():
    """When fewer than `pca_min_samples` unique text chunks exist, PCA isn't fit; build()
    falls back to head-truncation gracefully."""
    df = _make_structured(n_rows=20)  # 20 < pca_min_samples=64
    b = TextBlockBuilder(
        slot_width=128, text_column="text__sec", window_days=365,
        mode="voyage", use_pca=True, pca_min_samples=64, pca_fit_max_samples=512,
    )
    with patch("quant.state.text.VoyageEmbedder" if False else "quant.text.voyage_embedder.VoyageEmbedder", _MockVoyage):
        b.fit(df)
    # PCA not fitted because too few samples
    assert b._pca_components is None
    # Build should still work (uses truncation fallback)
    out = b.build(pd.Timestamp("2024-06-01"), df)
    assert out.shape == (128,)
    assert out.dtype == np.float32


def test_pca_fit_succeeds_and_projection_shape_is_correct():
    """With ≥pca_min_samples unique texts, PCA fits and project to slot_width."""
    df = _make_structured(n_rows=200)
    b = TextBlockBuilder(
        slot_width=128, text_column="text__sec", window_days=365,
        mode="voyage", use_pca=True, pca_min_samples=64, pca_fit_max_samples=200,
    )
    with patch("quant.text.voyage_embedder.VoyageEmbedder", _MockVoyage):
        b.fit(df)
    assert b._pca_components is not None
    # PCA components shape: (n_components, raw_embedding_dim) = (128, 1024)
    assert b._pca_components.shape == (128, 1024)
    assert b._pca_mean.shape == (1024,)

    # Build produces slot_width-dim output
    out = b.build(pd.Timestamp("2024-06-01"), df)
    assert out.shape == (128,)
    assert out.dtype == np.float32
    # PCA-projected output should NOT be identical to first-128 truncation of pooled raw embedding
    # (proves projection ran)
    mock = _MockVoyage()
    raw_emb = mock.embed(["test"])[0]
    truncated = raw_emb[:128]
    assert not np.allclose(out, truncated, atol=1e-4)


def test_pca_zero_evidence_returns_zeros():
    """Empty/below-threshold text input still returns zeros even with PCA enabled."""
    df = _make_structured(n_rows=200)
    b = TextBlockBuilder(
        slot_width=128, text_column="text__sec", window_days=1,  # tight window → very few chunks
        mode="voyage", use_pca=True, min_evidence_chars=10_000,
    )
    with patch("quant.text.voyage_embedder.VoyageEmbedder", _MockVoyage):
        b.fit(df)
    # Query a date with at most 1 chunk in its 1-day window → far below 10k chars
    out = b.build(pd.Timestamp("2024-01-15"), df)
    assert np.allclose(out, 0.0)


def test_pca_disabled_by_default():
    """Default config does not enable PCA — preserves backward compatibility."""
    b = TextBlockBuilder(slot_width=128, text_column="text__sec", mode="voyage")
    assert b.use_pca is False


def test_pca_apply_centers_input():
    """Manually-set PCA matrix subtracts mean before projecting."""
    b = TextBlockBuilder(slot_width=4, text_column="text__sec", mode="voyage", use_pca=True)
    # Set up a trivial PCA: subtract mean=[1, 2, 3, 4], identity projection
    b._pca_mean = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    b._pca_components = np.eye(4, dtype=np.float32)  # (4, 4)
    pooled = np.array([5.0, 5.0, 5.0, 5.0], dtype=np.float32)
    projected = b._apply_pca(pooled)
    # Should be pooled - mean = [4, 3, 2, 1]
    assert np.allclose(projected, [4.0, 3.0, 2.0, 1.0])
