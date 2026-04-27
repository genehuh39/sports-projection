from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date

import pandas as pd
from nba_api.stats.endpoints import leaguegamefinder

from src.models.trained_nba_model import NBAModelManager

logger = logging.getLogger(__name__)


@dataclass
class GameOutcome:
    game_date: date
    home_team_abbr: str
    away_team_abbr: str
    home_won: bool


def fetch_outcomes_for_dates(
    dates: list[date],
    seasons: list[str] | None = None,
) -> dict[tuple[str, str, str], GameOutcome]:
    """Fetch final scores for the given dates and key by (home, away, date).

    Pulls each season's worth of finished games via leaguegamefinder and
    filters locally — cheaper than per-date scoreboard calls when the
    journal spans many dates.
    """
    if not dates:
        return {}
    seasons = seasons or NBAModelManager.default_seasons()

    frames: list[pd.DataFrame] = []
    for season in seasons:
        try:
            frame = leaguegamefinder.LeagueGameFinder(
                season_nullable=season,
                season_type_nullable="Regular Season",
                league_id_nullable="00",
            ).get_data_frames()[0]
        except Exception as exc:
            logger.warning("leaguegamefinder failed for %s: %s", season, exc)
            continue
        if not frame.empty:
            frames.append(frame)

    # Also try playoffs — postseason games are common journal entries
    for season in seasons:
        try:
            frame = leaguegamefinder.LeagueGameFinder(
                season_nullable=season,
                season_type_nullable="Playoffs",
                league_id_nullable="00",
            ).get_data_frames()[0]
        except Exception as exc:
            logger.warning("leaguegamefinder playoffs failed for %s: %s", season, exc)
            continue
        if not frame.empty:
            frames.append(frame)

    if not frames:
        return {}

    df = pd.concat(frames, ignore_index=True)
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"]).dt.date
    target_dates = set(dates)
    df = df[df["GAME_DATE"].isin(target_dates)]
    if df.empty:
        return {}

    # Each game has two team rows. Identify home via "vs." in MATCHUP, away via "@".
    df["IS_HOME"] = df["MATCHUP"].astype(str).str.contains("vs.")

    out: dict[tuple[str, str, str], GameOutcome] = {}
    for game_id, game_rows in df.groupby("GAME_ID"):
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
        home_won = str(home.get("WL", "")) == "W"
        out[key] = GameOutcome(
            game_date=gd,
            home_team_abbr=str(home["TEAM_ABBREVIATION"]),
            away_team_abbr=str(away["TEAM_ABBREVIATION"]),
            home_won=home_won,
        )
    return out
