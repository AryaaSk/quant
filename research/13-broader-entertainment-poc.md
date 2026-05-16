# 13. Broader entertainment PoC design (next session)

User's idea (2026-05-15, end of session): instead of just-Oscars, union ALL Polymarket entertainment markets into one transformer-scale dataset. Train ONE unified model that learns the cross-event pattern "critic sentiment + festival momentum + audience signal → industry outcomes."

## Why this is better than just-Oscars

- **Scale**: ~3,000-8,000 individual yes/no predictions (vs ~500-800 for just-Oscars).
- **Transformer territory**: 5k+ events justifies a 5-30M-param transformer instead of GBDT.
- **Transfer learning**: model learns from Cannes how to predict Best Picture; from Emmys how to predict Globes; from box office how to predict streaming launches. A unified backbone captures the meta-pattern.
- **Balanced labels**: award winners are ~15-20% positive rate (one winner per category-year); box office milestones are ~50/50. Mixing them gives healthier class balance overall.
- **Unified text infrastructure**: Voyage embeddings of critic reviews / festival commentary / trade press (Variety, Deadline, Hollywood Reporter) are reusable across all event types about the same film/show.

## Polymarket source list (estimated event counts)

| Category | Annual events | Years | Estimated markets |
|---|---|---|---|
| Oscars | ~25 categories × 5-10 nominees | 5 | 500-800 |
| Emmys | ~20 categories × ~5 | 4 | 300-500 |
| Golden Globes | ~25 × ~5 | 4 | 400 |
| Grammys | ~20 × ~5 | 3-4 | 250-350 |
| BAFTAs | ~15 × ~5 | 3 | 200 |
| SAG Awards | ~10 × ~5 | 3 | 150 |
| Critics' Choice | ~15 × ~5 | 3 | 200 |
| Cannes / Venice / Berlin festival winners | per-festival per-year | 4-5 | 100-200 |
| Box office milestones ("Will [Film] open above $X?") | 50-100 per year | 4 | 200-400 |
| Streaming events (releases, cancellations) | 30-50 per year | 3 | 100-150 |
| **Total** | | | **2,500-3,500 minimum, ~5,000-8,000 with full coverage** |

## Pipeline (reuses existing infrastructure)

```
Stage A: polymarket_contracts adapter (replace existing stub)
  - Use Gamma API: GET /events?tag_slug=oscars&closed=true  (and emmys, golden-globes, etc)
  - Per market: closing price 14 days before resolution + realized outcome
  - Filter to closed (resolved) markets only
  -> data/raw/polymarket-entertainment/polymarket_contracts.parquet
  -> 3-8k rows

Stage A': film_critics adapter (new) — public, no agents needed
  - For each film/show mentioned in any market title:
    - Scrape Rotten Tomatoes critic + audience scores at T-14
    - Metacritic score
    - Letterboxd avg rating + reviews count
    - Festival wins prior to T-14
  - Cached per film
  -> data/raw/polymarket-entertainment/film_critics.parquet

Stage A'': news_scraper (haiku agents, gated)
  - per_entity_mode keyed by (film/show, ceremony_year)
  - One haiku scraper per unique film/show, scrapes commentary published before each
    relevant ceremony / release date
  - Budget: ~$10-20 in haiku (lots of entities, scraping is cheap on haiku)
  -> data/raw/polymarket-entertainment/_news/<film_slug>/

Stage B: structure (per (market_ticker, snapshot_date) row)

Stage C: state vector
  - numeric block: critic scores, festival wins, market price, market volume, days_to_resolution
  - market_state block: Polymarket implied probability + velocity
  - text block: Voyage-embedded commentary + reviews
  - category one-hot or learned embedding (e.g. "best_picture" vs "best_song" vs "box_office")
  - event_type one-hot (award / festival / box_office / streaming)

Stage D: train transformer on H200
  - ~5-10M params, suitable for 5k+ events
  - Hold out: most recent ceremony year fully (~500-1000 events)
  - Stratified by event type so the transformer sees all event_types in train

Stage E: backtest
  - Trade only when |edge| > polymarket fees (0.75-1.5% taker)
  - Kelly with cap
  - Compare to "just Polymarket closing" baseline
```

## Two model comparison (the actual PoC)

- **Model A**: numerical only (critic scores, festival wins, Polymarket prices, category one-hot). GBDT or transformer with empty text block.
- **Model B**: A + Voyage-embedded text from critic reviews + commentary scraped via haiku agents.

Both trained on the same unified entertainment dataset. Compare:
- Brier improvement vs Polymarket closing per event type
- Net return after commission

## Why this lands as a flagship demo

The PoC story: "we collected ~5,000 historical entertainment-market predictions from Polymarket, scraped the surrounding critic + festival + trade-press text via a haiku agent swarm, embedded it via Voyage AI, trained a small transformer on H200, and beat Polymarket's closing line by [X] Brier on the held-out 2025 ceremony year." Concrete, retail-relevant (entertainment is broadly familiar), transformer-scale, end-to-end agent-augmented.

## Infrastructure status as of end-of-session 2026-05-15

All needed pieces exist except:
- [ ] `polymarket_contracts` adapter implementation (stub exists, schema documented above)
- [ ] `film_critics` adapter (new; scrape RT / Metacritic / Letterboxd)
- [ ] `markets/polymarket-entertainment.yaml` (A and B variants)

Reusable (already shipped):
- `bin/h200` H200 deploy with Voyage key forwarding
- `scripts/run_remote_pipeline.py` state+train+backtest on H200
- `src/quant/text/voyage_embedder.py` Voyage with disk cache
- `src/quant/adapters/news_scraper.py` with `per_entity_mode`
- `skills/scrape-topic.md` (haiku, ~$1 cap per agent)
- `src/quant/state/text.py` with `mode: voyage`
- `src/quant/state/composer.py` for arbitrary block composition
- 41 passing tests covering the framework
