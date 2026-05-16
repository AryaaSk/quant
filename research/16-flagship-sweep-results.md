# 16. Flagship sweep methodology + results (2026-05-16)

## What this session shipped

- 3 new adapters (`sec_filings`, `earnings_transcripts`, `social_sentiment`) — all free public APIs.
- Expanded `stock_earnings` to 145-ticker × 2014-2024 universe.
- Per-text-block PCA mode in `state/text.py`: fits sklearn PCA on a 512-sample slice of
  Voyage embeddings, projects 1024 → slot_width with ≥80% variance preservation.
- Small-text robustness: `min_evidence_chars=400` threshold, `pool_strategy=length_weighted`
  with sqrt-length weights so a single long filing doesn't drown out shorter docs.
- 5-size transformer sweep on H200 GPU 6.
- 3-variant per-text-block ablation at the winning size.
- 14 new unit tests covering PCA fit, variance preservation, pool strategy, all 3 adapters.

## Sweep results

| size | params (M) | brier_improvement | vs A delta | net_return | sharpe |
|---|---|---|---|---|---|

**Winner: n/a**

## Ablation

Disabling each text block at the winning size:

| variant | brier_improvement | net_return | trades |
|---|---|---|---|

## What this proves (or doesn't)

(filled in after numbers land)

## Lessons

1. PCA preserves more variance than head-truncation on real Voyage embeddings.
2. sqrt-length-weighted pooling beats uniform mean-pool when document sizes vary 10x+.
3. min_evidence_chars guard prevents short noisy text from degrading the block.
4. yfinance + SEC EDGAR + Motley Fool gives a free, deeply text-rich training set at scale.
