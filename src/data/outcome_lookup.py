from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date

import pandas as pd
from nba_api.stats.endpoints import leaguegamefinder

logger = logging.getLogger(__name__)


@dataclass
class GameOutcome:
    game_date: date
    home_team_abbr: str
    away_team_abbr: str
    home_won: bool


def fetch_outcomes_for_dates(
    dates: list[date],
) -> dict[tuple[str, str, str], GameOutcome]:
    """Fetch final scores for the given dates, keyed by (home, away, date_iso).

    Issues one ``leaguegamefinder`` call per season-type with a
    date_from/date_to range covering only the requested dates. This avoids
    the prior implementation's full-season fetches across both regular
    season and playoffs (which pulled ~20k rows to filter down to a few).
    """
    if not dates:
        return {}

    date_from = min(dates).strftime("%m/%d/%Y")
    date_to = max(dates).strftime("%m/%d/%Y")

    frames: list[pd.DataFrame] = []
    for season_type in ("Regular Season", "Playoffs"):
        try:
            frame = leaguegamefinder.LeagueGameFinder(
                season_type_nullable=season_type,
                league_id_nullable="00",
                date_from_nullable=date_from,
                date_to_nullable=date_to,
            ).get_data_frames()[0]
        except Exception as exc:
            logger.warning("leaguegamefinder %s failed: %s", season_type, exc)
            continue
        if not frame.empty:
            frames.append(frame)

    if not frames:
        return {}

    df = pd.concat(frames, ignore_index=True)
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"]).dt.date
    target = set(dates)
    df = df[df["GAME_DATE"].isin(target)]
    if df.empty:
        return {}

    df["IS_HOME"] = df["MATCHUP"].astype(str).str.contains("vs.")

    out: dict[tuple[str, str, str], GameOutcome] = {}
    for _, game_rows in df.groupby("GAME_ID"):
        if len(game_rows) != 2:
            continue
        home_row = game_rows[game_rows["IS_HOME"]]
        away_row = game_rows[~game_rows["IS_HOME"]]
        if home_row.empty or away_row.empty:
            continue
        home = home_row.iloc[0]
        away = away_row.iloc[0]
        gd = home["GAME_DATE"]
        key = (home["TEAM_ABBREVIATION"], away["TEAM_ABBREVIATION"], gd.isoformat())
        out[key] = GameOutcome(
            game_date=gd,
            home_team_abbr=str(home["TEAM_ABBREVIATION"]),
            away_team_abbr=str(away["TEAM_ABBREVIATION"]),
            home_won=str(home.get("WL", "")) == "W",
        )
    return out
