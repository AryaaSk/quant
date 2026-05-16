"""tennis-data.co.uk adapter.

Source: http://www.tennis-data.co.uk/
Yearly Excel files cover ATP and WTA main-draw matches with results + closing odds.
Per-year URL: http://www.tennis-data.co.uk/{year}/{year}.xlsx (ATP)
              http://www.tennis-data.co.uk/{year}w/{year}.xlsx (WTA)

Columns include: Date, Tournament, Series, Court, Surface, Round, Best of, Winner, Loser,
WRank, LRank, WPts, LPts, W1..L5, Wsets, Lsets, Comment, B365W/L, EXW/L, LBW/L, PSW/PSL,
SJW/L, MaxW/L, AvgW/L. PSW/PSL are Pinnacle close odds; we prefer those.

Coverage caveat: tennis-data.co.uk is primarily ATP/WTA main draw. ITF Challenger coverage
is sparse to non-existent. For `betfair-itf-challenger`-style use, we treat ATP main draw
results as the v1 substitute and document the swap in the per-market notes. To get true
ITF/Challenger data, the next iteration should wire Jeff Sackmann's tennis_atp_challenger
GitHub repo (CSV-only, no odds — would need to merge with another odds source).
"""
from __future__ import annotations

from io import BytesIO
from pathlib import Path

import httpx
import numpy as np
import pandas as pd
from tenacity import retry, stop_after_attempt, wait_exponential

from quant.config import MarketConfig


TD_ATP_URL = "http://www.tennis-data.co.uk/{year}/{year}.xlsx"
TD_WTA_URL = "http://www.tennis-data.co.uk/{year}w/{year}.xlsx"


def _default_years() -> list[int]:
    # Default to a generous historical window. tennis-data.co.uk goes back to 2000+.
    return list(range(2015, 2026))


class TennisDataAdapter:
    def fetch(self, *, market: MarketConfig, source_params: dict) -> pd.DataFrame:
        tours = list(source_params.get("tours", ["ATP"]))
        years = list(source_params.get("years", _default_years()))
        # Optional filter: restrict to certain Tournament Series / surface / level.
        series_filter = source_params.get("series_filter")  # e.g. ["ATP250", "Masters 1000"]
        surface_filter = source_params.get("surface_filter")  # e.g. ["Hard", "Clay"]

        cache_dir = market.raw_dir() / "_td_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        scraped_at = pd.Timestamp.utcnow().tz_localize(None)

        frames: list[pd.DataFrame] = []
        for tour in tours:
            url_template = TD_ATP_URL if tour.upper() == "ATP" else TD_WTA_URL
            for year in years:
                df = self._fetch_year(tour, year, url_template, cache_dir)
                if df is None or df.empty:
                    continue
                df = df.copy()
                df["_tour"] = tour.upper()
                df["_year"] = year
                frames.append(df)

        if not frames:
            raise RuntimeError(f"tennis_data returned no matches for tours={tours} years={years}")

        matches = pd.concat(frames, ignore_index=True, copy=False)
        if series_filter:
            matches = matches[matches.get("Series", "").isin(series_filter)]
        if surface_filter:
            matches = matches[matches.get("Surface", "").isin(surface_filter)]
        if matches.empty:
            raise RuntimeError(f"tennis_data filter (series={series_filter} surface={surface_filter}) removed all rows")
        return self._to_records(matches, scraped_at)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=15))
    def _fetch_year(self, tour: str, year: int, url_template: str, cache_dir: Path) -> pd.DataFrame | None:
        cache_path = cache_dir / f"{tour.upper()}_{year}.xlsx"
        if cache_path.exists():
            try:
                return pd.read_excel(cache_path, engine="openpyxl")
            except Exception:
                cache_path.unlink(missing_ok=True)
        url = url_template.format(year=year)
        try:
            with httpx.Client(timeout=30.0, follow_redirects=True) as client:
                resp = client.get(url)
            if resp.status_code != 200 or not resp.content:
                return None
            cache_path.write_bytes(resp.content)
            return pd.read_excel(BytesIO(resp.content), engine="openpyxl")
        except (httpx.HTTPError, ValueError):
            return None

    def _to_records(self, matches: pd.DataFrame, scraped_at: pd.Timestamp) -> pd.DataFrame:
        if "Date" not in matches.columns:
            raise RuntimeError("tennis_data file missing Date column")

        matches = matches.copy()
        matches["_date"] = pd.to_datetime(matches["Date"], errors="coerce").dt.normalize()
        matches = matches.dropna(subset=["_date"])

        # We model the binary outcome P(player_A wins) where A is alphabetically first; this
        # avoids any leakage of "the winner is the one named first" by construction.
        # Tennis-data orders rows as (Winner, Loser) so we re-randomize labels here based
        # on (alphabetically first) becoming the predicted side.
        matches["_player_a"] = matches[["Winner", "Loser"]].min(axis=1)
        matches["_player_b"] = matches[["Winner", "Loser"]].max(axis=1)
        matches["_a_wins"] = (matches["_player_a"] == matches["Winner"]).astype(int)

        # Closing odds: prefer Pinnacle PSW/PSL, fall back to Bet365.
        for w_col, l_col in [("PSW", "PSL"), ("B365W", "B365L"), ("AvgW", "AvgL")]:
            if w_col in matches.columns and l_col in matches.columns:
                matches["_winner_odds"] = pd.to_numeric(matches[w_col], errors="coerce")
                matches["_loser_odds"] = pd.to_numeric(matches[l_col], errors="coerce")
                break
        else:
            raise RuntimeError("no closing-odds columns found in tennis_data file")

        # Re-orient odds onto (A, B).
        a_is_winner = matches["_a_wins"] == 1
        matches["_a_odds"] = np.where(a_is_winner, matches["_winner_odds"], matches["_loser_odds"])
        matches["_b_odds"] = np.where(a_is_winner, matches["_loser_odds"], matches["_winner_odds"])

        inv_a = 1.0 / matches["_a_odds"]
        inv_b = 1.0 / matches["_b_odds"]
        norm = inv_a + inv_b
        matches["_p_a_wins"] = (inv_a / norm).astype(float)

        # Rolling form per player (strictly past-only).
        matches = matches.sort_values("_date").reset_index(drop=True)
        wins_map: dict[str, list[int]] = {}
        days_since: dict[str, pd.Timestamp] = {}
        a_form = np.zeros(len(matches))
        b_form = np.zeros(len(matches))
        a_rest = np.zeros(len(matches))
        b_rest = np.zeros(len(matches))
        for i, row in matches.iterrows():
            a = row["_player_a"]
            b = row["_player_b"]
            a_form[i] = float(np.mean(wins_map.get(a, [])[-10:])) if wins_map.get(a) else 0.5
            b_form[i] = float(np.mean(wins_map.get(b, [])[-10:])) if wins_map.get(b) else 0.5
            a_rest[i] = float((row["_date"] - days_since[a]).days) if a in days_since else 30.0
            b_rest[i] = float((row["_date"] - days_since[b]).days) if b in days_since else 30.0
            # Update post-match
            wins_map.setdefault(a, []).append(int(row["_a_wins"]))
            wins_map.setdefault(b, []).append(int(1 - row["_a_wins"]))
            days_since[a] = row["_date"]
            days_since[b] = row["_date"]
        matches["_a_form_10"] = a_form
        matches["_b_form_10"] = b_form
        matches["_a_rest_days"] = np.clip(a_rest, 0, 60)
        matches["_b_rest_days"] = np.clip(b_rest, 0, 60)

        # Rankings (numeric)
        matches["_a_rank"] = np.where(a_is_winner, pd.to_numeric(matches.get("WRank"), errors="coerce"),
                                                    pd.to_numeric(matches.get("LRank"), errors="coerce"))
        matches["_b_rank"] = np.where(a_is_winner, pd.to_numeric(matches.get("LRank"), errors="coerce"),
                                                    pd.to_numeric(matches.get("WRank"), errors="coerce"))
        matches["_rank_diff"] = matches["_a_rank"].fillna(999) - matches["_b_rank"].fillna(999)

        rows: list[dict] = []
        for _, m in matches.iterrows():
            ts = m["_date"]
            if pd.isna(m.get("_a_odds")) or pd.isna(m.get("_p_a_wins")):
                continue
            url = (TD_ATP_URL if m["_tour"] == "ATP" else TD_WTA_URL).format(year=m["_year"])
            rows.append({
                "timestamp": ts,
                "source_published_at": ts,
                "scraped_at": scraped_at,
                "source_url": url,
                "source_type": f"tennis_data.{m['_tour']}",
                "target_event_time": ts,
                "y_realized": float(m["_a_wins"]),
                "p_market": float(m["_p_a_wins"]),
                "decimal_odds": float(m["_a_odds"]),
                "num__a_rank": float(m["_a_rank"]) if pd.notna(m["_a_rank"]) else 999.0,
                "num__b_rank": float(m["_b_rank"]) if pd.notna(m["_b_rank"]) else 999.0,
                "num__rank_diff": float(m["_rank_diff"]),
                "num__a_form_10": float(m["_a_form_10"]),
                "num__b_form_10": float(m["_b_form_10"]),
                "num__a_rest_days": float(m["_a_rest_days"]),
                "num__b_rest_days": float(m["_b_rest_days"]),
                "num__a_odds_inv": 1.0 / float(m["_a_odds"]),
                "num__b_odds_inv": 1.0 / float(m["_b_odds"]),
            })

        if not rows:
            raise RuntimeError("tennis_data produced 0 valid records")
        return pd.DataFrame(rows)
