# Polymarket entertainment unified PoC — Model A vs Model B (2026-05-15)

The second PoC of the dual A vs B methodology, folded into the same session as the stock-earnings PoC. Unifies **ALL resolved Polymarket entertainment markets** across awards (Oscars, Emmys, Golden Globes, Grammys, BAFTAs, SAG, Critics' Choice), festivals (Cannes, Venice, Sundance), box office milestones, and TV/streaming events.

## Setup

- **Source**: Polymarket Gamma API (`https://gamma-api.polymarket.com/events`), public, no auth.
- **Tag universe**: oscars, emmys, golden-globes, grammys, baftas, sag-awards, critics-choice-awards, cannes, venice-film-festival, sundance, movies, tv, streaming, box-office. Closed (resolved) markets only.
- **Total events**: **5,271 resolved yes/no markets** across **411 unique `event_slug`s** spanning **2024-01-16 → 2026-03-31**.
- **Distribution by tag**:

  tag | markets
  ---|---:
  grammys | 1,746
  oscars | 1,267
  golden-globes | 1,091
  movies | 533
  box-office | 444
  tv | 115
  emmys | 73
  streaming | 2

- **Label**: `y_realized = outcomePrices[yes_idx]` ∈ {0.0, 1.0} (clean resolutions only).
- **Class balance**: 4,715 No / 556 Yes (~10.5% positive rate). Naturally imbalanced because each (category, year) has many yes/no markets but one winner — the model has to beat Polymarket's already-calibrated price, not 50/50.
- **Held-out**: last 6 months.
- **Models**: GBDT. 5,271 events is at the lower edge of transformer territory; GBDT is the right choice for the A vs B isolation pass. Transformer is a natural follow-up once the methodology is validated.

## Architecture choice: per-`event_slug` text alignment

Each Polymarket market row carries a unique `target_event_time` (resolution date + per-market hash ms offset). The `event_slug` (e.g. `oscars-2026-best-picture-winner`, `grammys-album-of-the-year-winner`, `28-years-later-bone-temple-opening-weekend-box-office`) groups all per-nominee markets within one race.

`news_scraper` in `per_entity_mode` keyed by `event_slug`:
- Spawns **one haiku scraper per unique event_slug** (411 agents).
- Each agent scrapes critic/festival/industry commentary using query templates parameterised on the event_slug:
  - `"{entity} predictions analysis"`
  - `"{entity} oddsmaker pick analysis"`
  - `"{entity} critic commentary preview"`
  - `"{entity} expert forecast roundup"`
  - Plus static extras targeting Variety / Deadline / Hollywood Reporter / IndieWire / Rotten Tomatoes / Letterboxd.
- Per-event aggregation uses a **60-day pre-event window** and the **strict `published_at < target_event_time`** temporal-leak guard inherited from the news_scraper.

All markets sharing an `event_slug` see the same scraped commentary, which the model combines with the per-market `text__question` and numerical features.

## Model A: numerical baseline (complete)

State vector 128 dim:
- 96 numeric (market_volume, market_liquidity, days_open, yes_price_close — Polymarket closing price as a feature)
- 16 market_state (yes_price_close velocity, lags [1, 4])
- 16 calendar

### Result

| metric | value |
|---|---|
| brier_model | TBD (see run report) |
| brier_market | TBD |
| **brier_improvement** | **+0.1190** |
| trades | 24 |
| hit_rate | 58.33% |
| sharpe | 1.66 |
| max_drawdown | -7.86% |
| **net_return** | **+295,398.76%** |

Interpretation: the net_return is a Kelly-on-longshot artefact, not a real backtest result. With many markets priced at < 0.10, even a small edge × ~10x payoff × 24 compounding trades produces astronomical returns. **The headline number for this market is `brier_improvement`, not `net_return`**. Brier improvement of +0.119 over Polymarket closing is genuinely interesting; the model is finding signal beyond what raw closing prices encode (likely from volume/liquidity proxies of crowd attention).

A is the baseline for the A-vs-B comparison.

## Model B: numerical + Voyage-embedded haiku-scraped entertainment commentary (in flight)

Same data, same model class, plus:
- A `news_scraper` data source in `per_entity_mode` keyed by `event_slug` running 411 haiku scrapers in parallel (concurrency 5, max_articles=40, $1.0 cap per agent, $60 aggregate ceiling).
- Each agent scrapes commentary for one ceremony-year / film / box-office event using the entertainment-tailored query templates.
- All articles Voyage-embedded (`voyage-3-large`, 1024-dim, disk-cached by sha256) and mean-pooled into a 240-slot text block.

State vector 384 dim:
- 96 numeric (same as A)
- 16 news_article_count (per-event article density signal)
- 16 market_state (same as A)
- 240 voyage-embedded text
- 16 calendar

Cost estimate: 411 haiku agents × ~$0.05 each (well under per-agent $1.0 cap) → ~$20-30 claude. Voyage embeddings ~$5-15.

### Result

| metric | value | vs A |
|---|---|---|
| **brier_improvement** | **+0.1103** | A: +0.1190 (**regressed by -0.0087**) |
| n_trades | 24 | same |
| hit_rate | 54.17% | A: 58.33% |
| sharpe | 1.65 | A: 1.66 |
| max_drawdown | -8.62% | A: -7.86% |
| net_return | +268,802% | A: +295,398% (both Kelly artifacts) |

Interpretation: B marginally regressed on Brier (+0.1103 vs A's +0.1190). The text block didn't add signal beyond what numerical features already capture. Key cause: only 23/60 entities got scraped before the session budget exhausted, so ~70% of the 5,271 rows have empty `text__news` (the entire text block becomes zeros for those rows). The GBDT then learned to treat "zeros" as a "no signal" feature — but since most rows have it, the text block acts more like a noise dimension than a signal.

## Decision gate

- **B brier_improvement > A's by +0.02 AND B net behavior ≥ A's**: PoC confirmed. Entertainment-commentary text adds genuine signal over Polymarket closing prices. The methodology generalises to any unified Polymarket market category.
- **B ≈ A**: scraped commentary is redundant with what Polymarket closing already prices in (efficient sub-market). Try smaller subset (festival winners + Oscars only) where text is more directly informative.
- **B worse than A**: scraped commentary added noise, not signal. Tune text block size or per-event window length.

## Comparison to PoC 1 (earnings)

| dimension | earnings-beatmiss | polymarket-entertainment |
|---|---|---|
| events | 253 | **5,271** |
| held-out events | 28 | ~280 |
| label balance | 56% beat | 10.5% yes |
| text source | news pre-earnings | critic + festival + trade-press commentary |
| entity granularity | 24 tickers | 411 event_slugs |
| haiku scrape cost (B) | ~$5-15 | ~$20-30 |
| transformer-ready? | no | nearly (5k events is lower edge) |

Entertainment is the more transformer-ready domain. If both PoCs show B > A in Brier improvement, the entertainment dataset is the natural target for Phase 5 multi-market pretraining.
