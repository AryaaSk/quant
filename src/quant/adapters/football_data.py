"""Football-Data.co.uk adapter.

Downloads historical European football match results + closing odds from
https://www.football-data.co.uk/.

CSV format: one row per match with columns:
  Date, Time, HomeTeam, AwayTeam, FTHG, FTAG, FTR (H/D/A), HS, AS, ...
  Closing odds: PSCH (Pinnacle home close), PSCD (draw close), PSCA (away close).
  Fallbacks: B365CH, B365CD, B365CA (Bet365 close), or just B365H/D/A (early).

Each row emits a single match record with:
  timestamp = match date
  target_event_time = match date (one event per match)
  y_realized = 1 if home wins, 0 otherwise (for binary home_win label)
  p_market = 1 / closing_home_odds
  decimal_odds = closing_home_odds
  num__home_form_5, num__away_form_5 etc. computed pre-match (no leak)
"""
from __future__ import annotations

from io import BytesIO
from pathlib import Path

import httpx
import numpy as np
import pandas as pd
from tenacity import retry, stop_after_attempt, wait_exponential

from quant.config import MarketConfig


FD_BASE = "https://www.football-data.co.uk/mmz4281/{season}/{league}.csv"


# Season codes football-data uses: 2425 = 2024-25 season.
# We pull a default range (last 10 seasons) unless source_params says otherwise.
def _default_seasons() -> list[str]:
    out: list[str] = []
    for start_yy in range(15, 26):  # 2015-16 .. 2025-26
        end_yy = start_yy + 1
        out.append(f"{start_yy:02d}{end_yy:02d}")
    return out


class FootballDataAdapter:
    def fetch(self, *, market: MarketConfig, source_params: dict) -> pd.DataFrame:
        leagues = list(source_params.get("leagues", []))
        if not leagues:
            raise ValueError("football_data adapter needs source_params['leagues'] e.g. ['E2', 'E3']")
        seasons = list(source_params.get("seasons", _default_seasons()))

        cache_dir = market.raw_dir() / "_fd_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        scraped_at = pd.Timestamp.utcnow().tz_localize(None)

        all_matches: list[pd.DataFrame] = []
        for league in leagues:
            for season in seasons:
                df = self._fetch_csv(league, season, cache_dir)
                if df is None or df.empty:
                    continue
                df["_league"] = league
                df["_season"] = season
                all_matches.append(df)

        if not all_matches:
            raise RuntimeError(f"football_data returned no matches for {leagues} {seasons}")

        matches = pd.concat(all_matches, ignore_index=True, copy=False)
        return self._to_records(matches, scraped_at)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=15))
    def _fetch_csv(self, league: str, season: str, cache_dir: Path) -> pd.DataFrame | None:
        cache_path = cache_dir / f"{league}_{season}.csv"
        if cache_path.exists():
            try:
                return pd.read_csv(cache_path, encoding="utf-8")
            except UnicodeDecodeError:
                return pd.read_csv(cache_path, encoding="latin-1")
        url = FD_BASE.format(season=season, league=league)
        try:
            with httpx.Client(timeout=30.0, follow_redirects=True) as client:
                resp = client.get(url)
            if resp.status_code != 200 or not resp.content:
                return None
            try:
                df = pd.read_csv(BytesIO(resp.content), encoding="utf-8", low_memory=False)
            except UnicodeDecodeError:
                df = pd.read_csv(BytesIO(resp.content), encoding="latin-1", low_memory=False)
            cache_path.write_bytes(resp.content)
            # Defragment + tag with league/season so the caller does not later insert.
            df = df.copy()
            return df
        except (httpx.HTTPError, pd.errors.EmptyDataError):
            return None

    def _to_records(self, matches: pd.DataFrame, scraped_at: pd.Timestamp) -> pd.DataFrame:
        # Normalize date column.
        if "Date" not in matches.columns:
            raise RuntimeError("football_data CSV missing Date column")
        date_str = matches["Date"].astype(str)
        # Football-data uses dd/mm/yy or dd/mm/yyyy; pandas parses both with dayfirst=True.
        matches = matches.copy()
        matches["_date"] = pd.to_datetime(date_str, dayfirst=True, errors="coerce")
        matches = matches.dropna(subset=["_date"]).copy()
        matches["_date"] = matches["_date"].dt.normalize()

        # Result and outcome
        matches["_home_win"] = (matches.get("FTR", "") == "H").astype(int)

        # Closing odds: prefer Pinnacle close, then Bet365 close, then opening B365.
        for h, d, a in [("PSCH", "PSCD", "PSCA"), ("B365CH", "B365CD", "B365CA"), ("B365H", "B365D", "B365A")]:
            if h in matches.columns and d in matches.columns and a in matches.columns:
                matches["_close_home"] = pd.to_numeric(matches[h], errors="coerce")
                matches["_close_draw"] = pd.to_numeric(matches[d], errors="coerce")
                matches["_close_away"] = pd.to_numeric(matches[a], errors="coerce")
                break
        else:
            raise RuntimeError("no closing-odds columns found in football_data CSV")

        # p_market(home_win): normalize 1/odds across (H,D,A) to remove book overround.
        inv_h = 1.0 / matches["_close_home"]
        inv_d = 1.0 / matches["_close_draw"]
        inv_a = 1.0 / matches["_close_away"]
        norm = inv_h + inv_d + inv_a
        matches["_p_home_win"] = (inv_h / norm).astype(float)

        # Rolling form per team (computed in chronological order, strictly past-only).
        matches = matches.sort_values("_date").reset_index(drop=True)
        form_map: dict[str, list[int]] = {}
        home_form_5 = np.zeros(len(matches))
        away_form_5 = np.zeros(len(matches))
        for i, row in matches.iterrows():
            h_team = row.get("HomeTeam")
            a_team = row.get("AwayTeam")
            home_form_5[i] = np.mean(form_map.get(h_team, [])[-5:]) if form_map.get(h_team) else 0.0
            away_form_5[i] = np.mean(form_map.get(a_team, [])[-5:]) if form_map.get(a_team) else 0.0
            # Update post-match (do not leak: indexed by next match)
            home_pts = 3 if row.get("FTR") == "H" else (1 if row.get("FTR") == "D" else 0)
            away_pts = 3 if row.get("FTR") == "A" else (1 if row.get("FTR") == "D" else 0)
            form_map.setdefault(h_team, []).append(home_pts)
            form_map.setdefault(a_team, []).append(away_pts)
        matches["_home_form_5"] = home_form_5
        matches["_away_form_5"] = away_form_5

        # Goals scored / conceded rolling 5 + raw match stats (shots, corners, cards) rolling 5
        # Plus season-to-date totals (points, matches played, clean sheets) and rest days.
        # All STRICTLY past-only: we update the per-team buffers AFTER using them for the row.
        gs_map: dict[str, list[int]] = {}
        gc_map: dict[str, list[int]] = {}
        shots_map: dict[str, list[float]] = {}
        sot_map: dict[str, list[float]] = {}
        corners_map: dict[str, list[float]] = {}
        yellows_map: dict[str, list[float]] = {}
        clean_sheets_map: dict[str, list[int]] = {}
        last_date_map: dict[str, pd.Timestamp] = {}
        season_points_map: dict[tuple, int] = {}   # (team, season) -> running pts
        season_played_map: dict[tuple, int] = {}
        season_reds_map: dict[tuple, int] = {}

        # Allocate output arrays
        N = len(matches)
        home_gs = np.zeros(N); away_gs = np.zeros(N)
        home_gc = np.zeros(N); away_gc = np.zeros(N)
        home_shots = np.zeros(N); away_shots = np.zeros(N)
        home_sot = np.zeros(N); away_sot = np.zeros(N)
        home_corners = np.zeros(N); away_corners = np.zeros(N)
        home_yellows = np.zeros(N); away_yellows = np.zeros(N)
        home_clean_sheets = np.zeros(N); away_clean_sheets = np.zeros(N)
        home_rest_days = np.zeros(N); away_rest_days = np.zeros(N)
        home_points_std = np.zeros(N); away_points_std = np.zeros(N)
        home_played_std = np.zeros(N); away_played_std = np.zeros(N)
        home_reds_std = np.zeros(N); away_reds_std = np.zeros(N)

        def _mean5(lst):
            return float(np.mean(lst[-5:])) if lst else 0.0

        def _to_num(val):
            if val is None or (isinstance(val, float) and np.isnan(val)):
                return 0.0
            try:
                return float(val)
            except (ValueError, TypeError):
                return 0.0

        for i, row in matches.iterrows():
            h_team = row.get("HomeTeam"); a_team = row.get("AwayTeam")
            season = row.get("_season")
            date_now = row["_date"]

            # Read past-only buffers BEFORE updating
            home_gs[i] = _mean5(gs_map.get(h_team, []))
            home_gc[i] = _mean5(gc_map.get(h_team, []))
            away_gs[i] = _mean5(gs_map.get(a_team, []))
            away_gc[i] = _mean5(gc_map.get(a_team, []))

            home_shots[i] = _mean5(shots_map.get(h_team, []))
            away_shots[i] = _mean5(shots_map.get(a_team, []))
            home_sot[i] = _mean5(sot_map.get(h_team, []))
            away_sot[i] = _mean5(sot_map.get(a_team, []))
            home_corners[i] = _mean5(corners_map.get(h_team, []))
            away_corners[i] = _mean5(corners_map.get(a_team, []))
            home_yellows[i] = _mean5(yellows_map.get(h_team, []))
            away_yellows[i] = _mean5(yellows_map.get(a_team, []))
            home_clean_sheets[i] = float(sum(clean_sheets_map.get(h_team, [])[-5:]))
            away_clean_sheets[i] = float(sum(clean_sheets_map.get(a_team, [])[-5:]))

            # Rest days: gap since the team's last match
            home_rest_days[i] = float(np.clip((date_now - last_date_map[h_team]).days, 0, 60)) if h_team in last_date_map else 14.0
            away_rest_days[i] = float(np.clip((date_now - last_date_map[a_team]).days, 0, 60)) if a_team in last_date_map else 14.0

            # Season-to-date stats (reset each season because of (team, season) key)
            home_points_std[i] = float(season_points_map.get((h_team, season), 0))
            away_points_std[i] = float(season_points_map.get((a_team, season), 0))
            home_played_std[i] = float(season_played_map.get((h_team, season), 0))
            away_played_std[i] = float(season_played_map.get((a_team, season), 0))
            home_reds_std[i] = float(season_reds_map.get((h_team, season), 0))
            away_reds_std[i] = float(season_reds_map.get((a_team, season), 0))

            # NOW update the buffers with this match's results, so they affect FUTURE rows only
            fthg = int(_to_num(row.get("FTHG")))
            ftag = int(_to_num(row.get("FTAG")))
            gs_map.setdefault(h_team, []).append(fthg)
            gc_map.setdefault(h_team, []).append(ftag)
            gs_map.setdefault(a_team, []).append(ftag)
            gc_map.setdefault(a_team, []).append(fthg)
            shots_map.setdefault(h_team, []).append(_to_num(row.get("HS")))
            shots_map.setdefault(a_team, []).append(_to_num(row.get("AS")))
            sot_map.setdefault(h_team, []).append(_to_num(row.get("HST")))
            sot_map.setdefault(a_team, []).append(_to_num(row.get("AST")))
            corners_map.setdefault(h_team, []).append(_to_num(row.get("HC")))
            corners_map.setdefault(a_team, []).append(_to_num(row.get("AC")))
            yellows_map.setdefault(h_team, []).append(_to_num(row.get("HY")))
            yellows_map.setdefault(a_team, []).append(_to_num(row.get("AY")))
            clean_sheets_map.setdefault(h_team, []).append(int(ftag == 0))
            clean_sheets_map.setdefault(a_team, []).append(int(fthg == 0))
            last_date_map[h_team] = date_now
            last_date_map[a_team] = date_now

            # Season totals: points, matches played, reds
            ftr = row.get("FTR")
            h_pts = 3 if ftr == "H" else (1 if ftr == "D" else 0)
            a_pts = 3 if ftr == "A" else (1 if ftr == "D" else 0)
            season_points_map[(h_team, season)] = season_points_map.get((h_team, season), 0) + h_pts
            season_points_map[(a_team, season)] = season_points_map.get((a_team, season), 0) + a_pts
            season_played_map[(h_team, season)] = season_played_map.get((h_team, season), 0) + 1
            season_played_map[(a_team, season)] = season_played_map.get((a_team, season), 0) + 1
            season_reds_map[(h_team, season)] = season_reds_map.get((h_team, season), 0) + int(_to_num(row.get("HR")))
            season_reds_map[(a_team, season)] = season_reds_map.get((a_team, season), 0) + int(_to_num(row.get("AR")))

        matches["_home_gs_5"] = home_gs
        matches["_home_gc_5"] = home_gc
        matches["_away_gs_5"] = away_gs
        matches["_away_gc_5"] = away_gc
        matches["_home_shots_5"] = home_shots
        matches["_away_shots_5"] = away_shots
        matches["_home_sot_5"] = home_sot
        matches["_away_sot_5"] = away_sot
        matches["_home_corners_5"] = home_corners
        matches["_away_corners_5"] = away_corners
        matches["_home_yellows_5"] = home_yellows
        matches["_away_yellows_5"] = away_yellows
        matches["_home_clean_sheets_5"] = home_clean_sheets
        matches["_away_clean_sheets_5"] = away_clean_sheets
        matches["_home_rest_days"] = home_rest_days
        matches["_away_rest_days"] = away_rest_days
        matches["_home_points_std"] = home_points_std
        matches["_away_points_std"] = away_points_std
        matches["_home_played_std"] = home_played_std
        matches["_away_played_std"] = away_played_std
        matches["_home_reds_std"] = home_reds_std
        matches["_away_reds_std"] = away_reds_std

        rows: list[dict] = []
        for _, m in matches.iterrows():
            ts = m["_date"]
            if pd.isna(m.get("_close_home")) or pd.isna(m.get("_p_home_win")):
                continue
            rows.append({
                "timestamp": ts,
                "source_published_at": ts,
                "scraped_at": scraped_at,
                "source_url": FD_BASE.format(season=m["_season"], league=m["_league"]),
                "source_type": f"football_data.{m['_league']}",
                "target_event_time": ts,
                "y_realized": float(m["_home_win"]),
                "p_market": float(m["_p_home_win"]),
                "decimal_odds": float(m["_close_home"]),
                "num__home_form_5": float(m["_home_form_5"]),
                "num__away_form_5": float(m["_away_form_5"]),
                "num__home_gs_5": float(m["_home_gs_5"]),
                "num__home_gc_5": float(m["_home_gc_5"]),
                "num__away_gs_5": float(m["_away_gs_5"]),
                "num__away_gc_5": float(m["_away_gc_5"]),
                "num__home_shots_5": float(m["_home_shots_5"]),
                "num__away_shots_5": float(m["_away_shots_5"]),
                "num__home_sot_5": float(m["_home_sot_5"]),
                "num__away_sot_5": float(m["_away_sot_5"]),
                "num__home_corners_5": float(m["_home_corners_5"]),
                "num__away_corners_5": float(m["_away_corners_5"]),
                "num__home_yellows_5": float(m["_home_yellows_5"]),
                "num__away_yellows_5": float(m["_away_yellows_5"]),
                "num__home_clean_sheets_5": float(m["_home_clean_sheets_5"]),
                "num__away_clean_sheets_5": float(m["_away_clean_sheets_5"]),
                "num__home_rest_days": float(m["_home_rest_days"]),
                "num__away_rest_days": float(m["_away_rest_days"]),
                "num__home_points_std": float(m["_home_points_std"]),
                "num__away_points_std": float(m["_away_points_std"]),
                "num__home_played_std": float(m["_home_played_std"]),
                "num__away_played_std": float(m["_away_played_std"]),
                "num__home_reds_std": float(m["_home_reds_std"]),
                "num__away_reds_std": float(m["_away_reds_std"]),
                "num__close_home_inv": 1.0 / float(m["_close_home"]),
                "num__close_draw_inv": 1.0 / float(m["_close_draw"]) if pd.notna(m["_close_draw"]) else 0.0,
                "num__close_away_inv": 1.0 / float(m["_close_away"]) if pd.notna(m["_close_away"]) else 0.0,
            })

        if not rows:
            raise RuntimeError("football_data produced 0 valid match records")
        return pd.DataFrame(rows)
