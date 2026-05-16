"""Voyage AI embeddings client with disk caching.

Voyage is top-of-MTEB. Two models we care about:
- `voyage-3-large` (1024-dim): newest general-purpose model.
- `voyage-finance-2` (1024-dim): finance-domain-tuned. Better for macro/earnings/oil text.

Cost: ~$0.12-0.18 per 1M tokens. For our scale (~24-2000 articles per market) the spend is small.

Disk cache: every (model, text) pair is hashed (sha256) and cached to
`data/raw/<slug>/_voyage_cache/<hash>.npy`. Re-runs hit cache; only new strings pay.

Usage:
    from quant.text import embed_with_voyage
    vecs = embed_with_voyage(["hello world", "another doc"], model="voyage-3-large", cache_dir=Path("..."))
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential


VOYAGE_URL = "https://api.voyageai.com/v1/embeddings"


class VoyageEmbedder:
    """Thin caching client around the Voyage embeddings endpoint."""

    def __init__(self, model: str = "voyage-3-large", cache_dir: Path | None = None, api_key: str | None = None):
        self.model = model
        self.cache_dir = cache_dir
        if cache_dir:
            cache_dir.mkdir(parents=True, exist_ok=True)
        self.api_key = api_key or os.environ.get("VOYAGE_API_KEY")
        if not self.api_key:
            raise RuntimeError(
                "VOYAGE_API_KEY is not set. Add to .env or export it.\n"
                "Sign up: https://dash.voyageai.com/"
            )

    def embed(self, texts: list[str]) -> np.ndarray:
        """Return (N, dim) ndarray. Cached per (model, text) sha256."""
        if not texts:
            return np.zeros((0, self._embed_dim()), dtype=np.float32)

        # Resolve cache hits first
        results: list[np.ndarray | None] = [None] * len(texts)
        uncached_indices: list[int] = []
        uncached_texts: list[str] = []
        for i, t in enumerate(texts):
            cached = self._cache_get(t)
            if cached is not None:
                results[i] = cached
            else:
                uncached_indices.append(i)
                uncached_texts.append(t)

        if uncached_texts:
            fetched = self._embed_uncached(uncached_texts)
            for j, idx in enumerate(uncached_indices):
                results[idx] = fetched[j]
                self._cache_put(uncached_texts[j], fetched[j])

        return np.stack([r for r in results if r is not None], axis=0)

    def _embed_dim(self) -> int:
        # voyage-3-large / voyage-finance-2 / voyage-3 are all 1024 by default
        return 1024

    def _cache_key(self, text: str) -> str:
        h = hashlib.sha256(f"{self.model}\n{text}".encode("utf-8")).hexdigest()
        return h

    def _cache_path(self, text: str) -> Path | None:
        if not self.cache_dir:
            return None
        return self.cache_dir / f"{self._cache_key(text)}.npy"

    def _cache_get(self, text: str) -> np.ndarray | None:
        p = self._cache_path(text)
        if p and p.exists():
            try:
                return np.load(p)
            except (OSError, ValueError):
                return None
        return None

    def _cache_put(self, text: str, vec: np.ndarray) -> None:
        p = self._cache_path(text)
        if p:
            np.save(p, vec.astype(np.float32))

    @retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=1, min=2, max=20))
    def _embed_uncached(self, texts: list[str]) -> np.ndarray:
        # Voyage accepts up to 128 inputs per call (older models) or 1000 (newer). Be safe: chunk by 64.
        out: list[np.ndarray] = []
        for chunk_start in range(0, len(texts), 64):
            chunk = texts[chunk_start : chunk_start + 64]
            # Voyage rejects empty strings; substitute a single space.
            chunk = [c if c.strip() else " " for c in chunk]
            payload = {"model": self.model, "input": chunk}
            with httpx.Client(timeout=60.0) as client:
                resp = client.post(
                    VOYAGE_URL,
                    json=payload,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                )
                resp.raise_for_status()
                data = resp.json()
            embeddings = [np.asarray(item["embedding"], dtype=np.float32) for item in data["data"]]
            out.extend(embeddings)
        return np.stack(out, axis=0) if out else np.zeros((0, self._embed_dim()), dtype=np.float32)


def embed_with_voyage(
    texts: list[str], *, model: str = "voyage-3-large", cache_dir: Path | None = None
) -> np.ndarray:
    """Convenience function: embed a list of texts via Voyage with disk caching."""
    return VoyageEmbedder(model=model, cache_dir=cache_dir).embed(texts)
