# 01. Candidate market universe

Compiled 2026-05-15 from three parallel research streams (prediction markets, niche sports, niche crypto/finance). See `00-context.md` for the framing.

## Selection criteria

Each candidate must satisfy:

1. **Public data is reachable.** Historical odds + features can be obtained without paid market-data vendors.
2. **Liquidity ≥ $1k-$10k per event.** Enough to deploy our position size; small enough that institutional quants ignore it.
3. **Information asymmetry thesis.** A concrete reason why LLM agents can outperform the median market participant.
4. **Sample size.** Enough historical events to train a transformer (>500) or a LightGBM fallback (>100).
5. **Friction is bounded.** Commission + slippage + safety margin doesn't eat the realistic edge.

## Tier A — Prediction markets (structured, regulated, clean backtest)

| Slug | Platform | Event | Liquidity/event | Friction | Thesis |
|---|---|---|---|---|---|
| `kalshi-cpi` | Kalshi | Monthly CPI direction | $10M-$50M | 0.2% taker | Nowcasts (Atlanta Fed, Truflation, Redbook) + Fed-speaker tone diverge from consensus 1-3 days pre-release. Federal Reserve paper validated Kalshi beats Bloomberg consensus. |
| `kalshi-nfp` | Kalshi | Monthly non-farm payrolls | $10M-$50M | 0.2% taker | ADP leaks signal 2 days before BLS; jobless claims trend; regional Fed surveys. |
| `kalshi-fed-rate` | Kalshi | FOMC rate decision | $50M+ | 0.2% taker | Saturated; Fed funds futures dominate. LOW priority, included for completeness. |
| `kalshi-weather` | Kalshi | Hurricane / temperature records | $1M-$10M | 0.2% taker | NOAA/ECMWF ensemble disagreement = inefficiency. |
| `kalshi-fda` | Kalshi | FDA approval decisions | $0.5M-$2M | 0.2% taker | Clinical trial outcomes, AdCom leaks, SEC filings. LightGBM fallback (low event count). |
| `polymarket-oscars` | Polymarket | Award category outcomes | $5M-$30M/cat | 0.75% taker | Critic synthesis (Letterboxd, RT, festival results) + voter-bloc demographics. Beat Ben Zauzmer baseline. |
| `polymarket-emmys` | Polymarket | Award category outcomes | $1M-$10M/cat | 0.75% taker | Same as Oscars; fewer pro modelers. |
| `polymarket-elections` | Polymarket | Down-ballot (NOT national) | $1M-$10M | 0.75-1.0% taker | National saturated; state/local races have asymmetric polling + endorsements. |

## Tier B — Niche sports on exchanges (peer-to-peer, fan money)

| Slug | Platform | Event | Liquidity/event | Friction | Thesis |
|---|---|---|---|---|---|
| `betfair-brazil-serieb` | Betfair Exchange | Match outcome + cards/corners | £15k-£80k | 5% commission on profit | Portuguese-language press, regional injury reports, travel fatigue. |
| `betfair-eng-l1l2` | Betfair Exchange | English League One/Two match | £10k-£50k | 5% commission | Local club news, loan moves, weather on lower-tier pitches; pure recreational counterparty. |
| `betfair-wsl-nwsl` | Betfair Exchange | Women's football | £5k-£40k | 5% commission | Limited modeling by sportsbooks; team news fragmented. |
| `betfair-jleague` | Betfair Exchange | J-League match | £5k-£30k | 5% commission | Japanese-language press; advanced metrics rare in Western models. |
| `betfair-itf-challenger` | Betfair Exchange | ITF Challenger tennis | £2k-£15k | 5% commission | Independently confirmed market inefficiency; surface adjustments, scheduling fatigue. |
| `betfair-tt-csl` | Betfair Exchange | Chinese Super League table tennis | £2k-£15k | 5% commission | Mandarin-only press; ITTF archives. |
| `betfair-nrl` | Betfair Exchange | NRL rugby league | £10k-£50k | 5% commission | Australian press, weather, injury list timing, ref tendencies. |
| `betfair-corners-cards` | Betfair Exchange | Exotic markets in tier-B leagues | £3k-£15k | 5% commission | Pro sharps avoid exotics in non-elite leagues. |

## Tier C — Crypto/DeFi (high risk; survivorship bias warning)

| Slug | Platform | Event | Liquidity/event | Friction | Thesis |
|---|---|---|---|---|---|
| `hyperliquid-midcap` | Hyperliquid | Mid-cap alt perp 4-24h returns | $5M-$50M daily | ~0.05% taker | Discord/Telegram sentiment, GitHub commit velocity, on-chain whale flows. |
| `drift-solana` | Drift Protocol | Solana ecosystem perps, 6-48h | $2M-$20M daily | ~0.1% taker | Airdrop schedules, governance votes, Solana ecosystem Twitter. |
| `token-unlock` | Multiple | Pre/post token-unlock arbitrage | varies | combined CEX+DEX | Unlock dates public; LLM can distinguish "sellable" vs "locked" dilution. |

## Explicit exclusions (and why)

- **Premier League match outcome:** closing line near-efficient, professional sharp money dominant.
- **BTC/ETH perp on top CEXs:** institutional quants dominate; funding rates clustered around arbitrage.
- **S&P 500 / oil futures / gold:** efficient and capital-intensive.
- **US presidential Polymarket:** saturated + insider front-running history.
- **Deribit BTC/ETH options:** pro-dominated.

## Next step

Phase 1 screening (Phase 1 in the plan) runs an LLM agent per yaml to:
- verify data reachability (download one sample)
- estimate events/year
- sample 5 live events to estimate liquidity
- estimate text-info density
- produce a 0-100 feasibility score

See [04-screening-method.md](04-screening-method.md).
