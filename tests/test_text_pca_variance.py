"""PCA variance-preservation test.

Verifies that PCA-projection from 1024-dim to slot_width preserves the bulk of variance
present in the original embeddings. Without this guarantee, PCA might not be meaningfully
better than naive head-truncation.

All mocked. No network.
"""
from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd

from quant.state.text import TextBlockBuilder


class _LowRankMockVoyage:
    """Mock that returns low-rank embeddings (1024 dim but real rank ~64) so PCA has structure to capture."""
    def __init__(self, *args, **kwargs):
        rng = np.random.default_rng(seed=42)
        self.basis = rng.standard_normal((64, 1024)).astype(np.float32)

    def embed(self, texts: list[str]) -> np.ndarray:
        rng = np.random.default_rng(seed=hash(tuple(texts)) % (2**32))
        rows = []
        for t in texts:
            coef = np.random.RandomState(hash(t) % (10**6)).standard_normal(64).astype(np.float32)
            rows.append(coef @ self.basis)
        return np.stack(rows, axis=0)


def _make_corpus(n: int = 200, col: str = "text__sec") -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    texts = [f"corpus chunk {i} padded with more content xxxxxxxxxx" for i in range(n)]
    return pd.DataFrame({col: texts}, index=idx)


def test_pca_preserves_at_least_80_pct_variance_at_slot_width_512():
    """When slot_width=512, PCA should preserve ≥80% of the input variance.

    Compared to naive truncation (which preserves only first 512 of 1024 dims = arbitrary
    50% of variance on a real-shaped embedding), PCA should compress information much
    more efficiently.
    """
    df = _make_corpus(n=200)
    b = TextBlockBuilder(
        slot_width=512, text_column="text__sec", window_days=365,
        mode="voyage", use_pca=True, pca_min_samples=64, pca_fit_max_samples=200,
    )
    with patch("quant.text.voyage_embedder.VoyageEmbedder", _LowRankMockVoyage):
        b.fit(df)

    assert b._pca is not None
    explained = float(b._pca.explained_variance_ratio_.sum())
    assert explained >= 0.80, f"PCA should preserve ≥80% variance at slot_width=512, got {explained:.2%}"


def test_pca_output_is_orthogonal_components():
    """PCA components should be orthonormal (sklearn guarantee). Defensive check."""
    df = _make_corpus(n=200)
    b = TextBlockBuilder(
        slot_width=128, text_column="text__sec", window_days=365,
        mode="voyage", use_pca=True, pca_min_samples=64, pca_fit_max_samples=200,
    )
    with patch("quant.text.voyage_embedder.VoyageEmbedder", _LowRankMockVoyage):
        b.fit(df)

    C = b._pca_components  # (n_comp, raw_dim)
    # Inner product matrix should be near-identity
    inner = C @ C.T
    expected = np.eye(C.shape[0], dtype=np.float32)
    np.testing.assert_allclose(inner, expected, atol=1e-4)


def test_pca_beats_truncation_on_reconstruction_error():
    """Reconstruction MSE: PCA projection + back-projection should give lower error
    than head-truncation + zero-pad, for an embedding sampled OUTSIDE the PCA fit set."""
    df = _make_corpus(n=200)
    b = TextBlockBuilder(
        slot_width=512, text_column="text__sec", window_days=365,
        mode="voyage", use_pca=True, pca_min_samples=64, pca_fit_max_samples=200,
    )
    with patch("quant.text.voyage_embedder.VoyageEmbedder", _LowRankMockVoyage):
        b.fit(df)

    mock = _LowRankMockVoyage()
    test_emb = mock.embed(["held-out test chunk that wasn't in training"])[0]  # (1024,)

    # PCA path: project, back-project
    centered = test_emb - b._pca_mean
    projected = centered @ b._pca_components.T  # (512,)
    reconstructed_pca = projected @ b._pca_components + b._pca_mean  # (1024,)
    pca_mse = float(np.mean((test_emb - reconstructed_pca) ** 2))

    # Truncation path: keep first 512, zero the rest
    reconstructed_trunc = test_emb.copy()
    reconstructed_trunc[512:] = 0
    trunc_mse = float(np.mean((test_emb - reconstructed_trunc) ** 2))

    assert pca_mse < trunc_mse, f"PCA MSE={pca_mse:.4f} should be < truncation MSE={trunc_mse:.4f}"
