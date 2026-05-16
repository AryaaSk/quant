"""External text-encoder integrations.

- `voyage_embedder`: voyage-3-large / voyage-finance-2 (paid API, top-of-MTEB)
- (future) `gpu_embedder`: Qwen3-Embedding-8B on H200
"""
from quant.text.voyage_embedder import VoyageEmbedder, embed_with_voyage  # noqa: F401
