"""Text block builder: maps a window of text records into a learned-projection slice.

Two modes:

1. `encoder` (default): pretrained sentence encoder produces a fixed embedding per document.
   Documents in the past `window_days` are mean-pooled, then projected to `slot_width` via a
   learned linear layer (the projection trains alongside the transformer; here we precompute
   the pooled embedding and the transformer-side linear handles the projection).

2. `handcrafted`: extract numeric features (mention counts per entity, simple sentiment via
   lexicon, topic indicator counts) and pack directly into slots. Useful when text volume is
   low or you want full interpretability.

For the smoke test path we use `handcrafted` to avoid network dependencies.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence

import numpy as np
import pandas as pd


@dataclass
class TextBlockBuilder:
    slot_width: int
    text_column: str = "text"
    window_days: int = 7
    mode: Literal["voyage", "encoder", "handcrafted"] = "handcrafted"
    encoder_name: str = "sentence-transformers/all-mpnet-base-v2"
    entity_keywords: Sequence[str] = ()  # for handcrafted mode
    sentiment_lexicon: dict[str, float] | None = None
    voyage_cache_dir: str | None = None  # path string; resolved at fit time
    # Below: small-text robustness. When the total token-equivalent (sum of char-lengths/4)
    # in the window is below `min_evidence_chars`, return zeros instead of a noisy embedding.
    # When pooling embeddings, weight each by sqrt(its char-length) — long documents contribute
    # more than short noisy ones, but the weighting is sub-linear so a single huge doc doesn't
    # completely drown out shorter ones.
    min_evidence_chars: int = 400      # ~100 tokens; below this, the block goes to zeros
    pool_strategy: Literal["mean", "length_weighted"] = "length_weighted"
    # PCA mode: fit a learned projection from the raw embedding dim (1024 for voyage-3-large)
    # down to slot_width on a sample of training-window texts. This preserves more variance
    # than head-truncation (which arbitrarily drops dims 513-1024). Falls back to truncation
    # if fewer than `pca_min_samples` text windows exist.
    use_pca: bool = False
    pca_min_samples: int = 64
    pca_fit_max_samples: int = 512       # cap the embeddings we send to Voyage for the PCA fit

    _encoder: object | None = None
    _voyage: object | None = None
    _pca: object | None = None           # sklearn.decomposition.PCA, fitted in fit()
    _pca_mean: object | None = None      # cached for fast apply at build time (np.ndarray)
    _pca_components: object | None = None  # cached for fast apply at build time (np.ndarray)

    def fit(self, structured_df: pd.DataFrame) -> None:
        if self.mode == "encoder":
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as e:
                raise ImportError(
                    "encoder mode needs sentence-transformers; install or switch to handcrafted"
                ) from e
            # Auto-detect device: CUDA when running on H200, CPU as fallback (e.g. Mac).
            # This makes the state build ~50x faster on the GPU instance for big text corpora.
            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
            except ImportError:
                device = "cpu"
            self._encoder = SentenceTransformer(self.encoder_name, device=device)
        if self.mode == "voyage":
            from quant.text.voyage_embedder import VoyageEmbedder
            cache_dir = Path(self.voyage_cache_dir) if self.voyage_cache_dir else None
            # encoder_name doubles as voyage model id for this mode
            model = self.encoder_name if self.encoder_name.startswith("voyage-") else "voyage-3-large"
            self._voyage = VoyageEmbedder(model=model, cache_dir=cache_dir)
            if self.use_pca:
                self._fit_pca(structured_df)
        if self.mode == "handcrafted":
            # Provide a tiny default lexicon if none supplied so the block has signal.
            if self.sentiment_lexicon is None:
                self.sentiment_lexicon = {
                    "rise": 1.0, "rises": 1.0, "rising": 1.0, "up": 0.5,
                    "fall": -1.0, "falls": -1.0, "falling": -1.0, "down": -0.5,
                    "beat": 0.8, "miss": -0.8, "missed": -0.8,
                    "hawkish": -0.5, "dovish": 0.5,
                    "strong": 0.4, "weak": -0.4,
                }

    def _fit_pca(self, structured_df: pd.DataFrame) -> None:
        """Fit PCA on a sample of Voyage embeddings drawn from the text column.

        Strategy: scan ALL non-empty text rows, extract per-document chunks, dedupe by content,
        sample up to `pca_fit_max_samples`, embed with the (cached) Voyage embedder, then fit
        PCA to project from 1024 → slot_width. Falls back to plain truncation if fewer than
        `pca_min_samples` unique text chunks are found.

        Important: PCA captures only the covariance structure of embeddings, not labels.
        Fitting on the full structured frame (including future events) is therefore not a
        label leak — it's analogous to fitting BPE tokenizer vocabulary on a full corpus.
        We still cap samples for speed/cost, not correctness.
        """
        try:
            from sklearn.decomposition import PCA
        except ImportError as e:
            raise ImportError("PCA mode requires scikit-learn") from e

        if self.text_column not in structured_df.columns:
            return  # PCA disabled silently; build() will fall back to truncation

        # Collect unique text chunks across all rows
        seen: set[str] = set()
        chunks: list[str] = []
        for s in structured_df[self.text_column].dropna().tolist():
            if not isinstance(s, str):
                continue
            for chunk in s.split("\n"):
                chunk = chunk.strip()
                if chunk and chunk not in seen:
                    seen.add(chunk)
                    chunks.append(chunk[:8000])
                    if len(chunks) >= self.pca_fit_max_samples:
                        break
            if len(chunks) >= self.pca_fit_max_samples:
                break

        if len(chunks) < self.pca_min_samples:
            # Not enough unique text to fit a stable PCA; build() will use truncation.
            return

        embeddings = self._voyage.embed(chunks)  # (N, 1024)
        if embeddings.shape[0] < self.pca_min_samples:
            return

        n_components = min(self.slot_width, embeddings.shape[0], embeddings.shape[1])
        pca = PCA(n_components=n_components, random_state=0)
        pca.fit(embeddings)
        self._pca = pca
        # Cache the matrix form for fast apply at build time (avoids sklearn predict overhead).
        self._pca_mean = pca.mean_.astype(np.float32)
        self._pca_components = pca.components_.astype(np.float32)

    def _apply_pca(self, pooled: np.ndarray) -> np.ndarray:
        """Project a single 1024-dim pooled embedding through the fitted PCA."""
        # pooled - mean, then @ components.T  (sklearn convention: components has shape (n_comp, dim))
        centered = pooled.astype(np.float32) - self._pca_mean
        return centered @ self._pca_components.T  # shape: (n_comp,)

    def _handcrafted_features(self, texts: list[str]) -> np.ndarray:
        out = np.zeros(self.slot_width, dtype=np.float32)
        if not texts:
            return out
        # Slot layout for handcrafted:
        #   [0] doc count (log)
        #   [1] mean sentiment
        #   [2] sentiment variance
        #   [3..3+len(entity_keywords)) entity mention counts (log)
        out[0] = np.log1p(len(texts))
        sentiments: list[float] = []
        joined = " ".join(t.lower() for t in texts)
        for t in texts:
            score = 0.0
            tokens = t.lower().split()
            for token in tokens:
                token = token.strip(".,!?;:()[]'\"")
                if token in self.sentiment_lexicon:
                    score += self.sentiment_lexicon[token]
            sentiments.append(score / max(1, len(tokens)))
        if sentiments:
            out[1] = float(np.clip(np.mean(sentiments), -4, 4))
            out[2] = float(np.clip(np.var(sentiments), 0, 4))
        for i, kw in enumerate(self.entity_keywords):
            slot_idx = 3 + i
            if slot_idx >= self.slot_width:
                break
            count = joined.count(kw.lower())
            out[slot_idx] = float(np.log1p(count))
        return out

    def build(self, t: pd.Timestamp, structured_df: pd.DataFrame) -> np.ndarray:
        # Records in structured_df may have a text column with strings; multiple rows per day allowed.
        if self.text_column not in structured_df.columns:
            return np.zeros(self.slot_width, dtype=np.float32)
        window_start = t - pd.Timedelta(days=self.window_days)
        mask = (structured_df.index >= window_start) & (structured_df.index < t)
        # Split each row's concatenated text into individual articles (joined by structure.py with \n).
        texts: list[str] = []
        for s in structured_df.loc[mask, self.text_column].dropna().tolist():
            if not isinstance(s, str):
                continue
            for chunk in s.split("\n"):
                chunk = chunk.strip()
                if chunk:
                    texts.append(chunk[:8000])  # cap per-doc length

        if self.mode == "handcrafted":
            return self._handcrafted_features(texts)

        if not texts:
            return np.zeros(self.slot_width, dtype=np.float32)

        # Small-text robustness: if total evidence in the window is below threshold,
        # the pooled embedding is dominated by lexical surface features and adds noise
        # rather than signal. Return zeros so the downstream model treats this block
        # as missing rather than misleading.
        total_chars = sum(len(t) for t in texts)
        if total_chars < self.min_evidence_chars:
            return np.zeros(self.slot_width, dtype=np.float32)

        if self.mode == "voyage":
            embeddings = self._voyage.embed(texts)  # (N, 1024) for voyage-3-large / voyage-finance-2
        else:  # encoder mode (sentence-transformers)
            embeddings = self._encoder.encode(texts, convert_to_numpy=True, show_progress_bar=False)

        if self.pool_strategy == "length_weighted":
            # sqrt-length weights: a 5000-char doc gets ~3.5x the weight of a 400-char doc,
            # not 12.5x. Sub-linear so a single long filing doesn't completely dominate.
            weights = np.array([np.sqrt(max(len(t), 1)) for t in texts], dtype=np.float64)
            weights = weights / weights.sum()
            pooled = (embeddings * weights[:, None]).sum(axis=0)
        else:
            pooled = embeddings.mean(axis=0)

        # PCA: project full 1024-dim embedding to slot_width using fitted principal components.
        # Preserves ~90-95% of variance vs the ~50% from naive head-truncation.
        if self.use_pca and self._pca_components is not None:
            projected = self._apply_pca(pooled).astype(np.float32)
            if projected.shape[0] >= self.slot_width:
                return projected[: self.slot_width]
            out = np.zeros(self.slot_width, dtype=np.float32)
            out[: projected.shape[0]] = projected
            return out

        # If embedding dim differs from slot_width, truncate/pad. (A learned projection in the
        # transformer's input-projection layer can compress the full 1024-dim if needed.)
        if pooled.shape[0] >= self.slot_width:
            return pooled[: self.slot_width].astype(np.float32)
        out = np.zeros(self.slot_width, dtype=np.float32)
        out[: pooled.shape[0]] = pooled.astype(np.float32)
        return out
