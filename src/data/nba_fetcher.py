import logging
from datetime import date, datetime, timedelta, timezone
from typing import Optional

import pandas as pd
import polars as pl
from nba_api.stats.endpoints import leaguegamefinder, scoreboardv3

from src.data.odds_providers import (
    EspnOddsProvider,
    TheOddsApiProvider,
    empty_odds_frame,
    normalize_team_code,
)

logger = logging.getLogger(__name__)


class NBAFetcher:
    """Fetches live NBA schedule, market odds, and recent team form."""

    def __init__(self, default_market_odds: int = -110):
        self.default_market_odds = default_market_odds
        self.odds_providers = [
            TheOddsApiProvider(priority=10),
            EspnOddsProvider(priority=20),
        ]

    @staticmethod
    def current_season(today: Optional[date] = None) -> str:
        today = today or datetime.now(timezone.utc).date()
        start_year = today.year if today.month >= 10 else today.year - 1
        end_year_short = str((start_year + 1) % 100).zfill(2)
        return f"{start_year}-{end_year_short}"

    @staticmethod
    def _coerce_date(game_date: Optional[date | str] = None) -> date:
        if game_date is None:
            return datetime.now(timezone.utc).date()
        if isinstance(game_date, date):
            return game_date
        return datetime.fromisoformat(str(game_date)).date()

    def _scoreboard_frames(
        self, game_date: Optional[date | str] = None
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        target_date = self._coerce_date(game_date)
        board = scoreboardv3.ScoreboardV3(game_date=target_date.isoformat())
        frames = board.get_data_frames()
        games_df = frames[1] if len(frames) > 1 else pd.DataFrame()
        teams_df = frames[2] if len(frames) > 2 else pd.DataFrame()
        leaders_df = frames[3] if len(frames) > 3 else pd.DataFrame()
        return games_df, teams_df, leaders_df

    @staticmethod
    def _resolve_home_away_teams(
        game_row: pd.Series, teams_df: pd.DataFrame, leaders_df: pd.DataFrame
    ) -> tuple[Optional[pd.Series], Optional[pd.Series]]:
        game_id = game_row["gameId"]
        game_teams = teams_df[teams_df["gameId"] == game_id]
        game_leaders = (
            leaders_df[leaders_df["gameId"] == game_id] if not leaders_df.empty else pd.DataFrame()
        )

        if not game_leaders.empty and "leaderType" in game_leaders.columns:
            home_team_ids = game_leaders.loc[
                game_leaders["leaderType"] == "home", "teamId"
            ].tolist()
            away_team_ids = game_leaders.loc[
                game_leaders["leaderType"] == "away", "teamId"
            ].tolist()
            if home_team_ids and away_team_ids:
                home_row = game_teams[game_teams["teamId"] == home_team_ids[0]]
                away_row = game_teams[game_teams["teamId"] == away_team_ids[0]]
                if not home_row.empty and not away_row.empty:
                    return home_row.iloc[0], away_row.iloc[0]

        game_code = str(game_row.get("gameCode", ""))
        tricodes = game_code.split("/")[-1]
        away_code, home_code = tricodes[:3], tricodes[3:6]
        home_row = game_teams[game_teams["teamTricode"] == home_code]
        away_row = game_teams[game_teams["teamTricode"] == away_code]
        if not home_row.empty and not away_row.empty:
            return home_row.iloc[0], away_row.iloc[0]

        return None, None


    def get_market_odds(self, game_date: Optional[date | str] = None) -> pl.DataFrame:
        target_date = self._coerce_date(game_date)
        odds_frames = [provider.fetch(target_date) for provider in self.odds_providers]
        odds_frames = [frame for frame in odds_frames if not frame.is_empty()]

        if not odds_frames:
            return empty_odds_frame()

        merged = pl.concat(odds_frames, how="vertical_relaxed").sort("provider_priority")
        return merged.unique(
            subset=["game_date", "away_team_code", "home_team_code"],
            keep="first",
            maintain_order=True,
        ).drop("provider_priority")

    def get_upcoming_games(
        self, start_date: Optional[date | str] = None, days_ahead: int = 7
    ) -> pl.DataFrame:
        """Returns upcoming NBA games for the next few days with market odds when available."""
        start = self._coerce_date(start_date)

        empty_schema = {
            "game_id": pl.Utf8,
            "game_date": pl.Utf8,
            "status": pl.Utf8,
            "home_team_id": pl.Int64,
            "away_team_id": pl.Int64,
            "home_team_code": pl.Utf8,
            "away_team_code": pl.Utf8,
            "home_team_name": pl.Utf8,
            "away_team_name": pl.Utf8,
            "market_odds": pl.Int64,
            "away_market_odds": pl.Int64,
            "market_spread": pl.Float64,
            "market_total": pl.Float64,
            "market_provider": pl.Utf8,
            "market_bookmaker": pl.Utf8,
            "market_source": pl.Utf8,
        }

        for offset in range(days_ahead + 1):
            target_date = start + timedelta(days=offset)
            try:
                games_df, teams_df, leaders_df = self._scoreboard_frames(target_date)
            except Exception as exc:
                logger.warning("Failed to fetch scoreboard for %s: %s", target_date, exc)
                continue

            if games_df.empty or teams_df.empty:
                continue

            records: list[dict] = []
            for _, game_row in games_df.iterrows():
                home_team, away_team = self._resolve_home_away_teams(
                    game_row, teams_df, leaders_df
                )
                if home_team is None or away_team is None:
                    continue

                records.append(
                    {
                        "game_id": str(game_row["gameId"]),
                        "game_date": target_date.isoformat(),
                        "status": str(game_row.get("gameStatusText", "")),
                        "home_team_id": int(home_team["teamId"]),
                        "away_team_id": int(away_team["teamId"]),
                        "home_team_code": normalize_team_code(str(home_team["teamTricode"])),
                        "away_team_code": normalize_team_code(str(away_team["teamTricode"])),
                        "home_team_name": f"{home_team['teamCity']} {home_team['teamName']}",
                        "away_team_name": f"{away_team['teamCity']} {away_team['teamName']}",
                    }
                )

            if not records:
                continue

            schedule_df = pl.DataFrame(records).unique(subset=["game_id"]).sort(
                ["game_date", "game_id"]
            )
            odds_df = self.get_market_odds(target_date)

            if odds_df.is_empty():
                return schedule_df.with_columns(
                    [
                        pl.lit(self.default_market_odds).cast(pl.Int64).alias("market_odds"),
                        pl.lit(None).cast(pl.Int64).alias("away_market_odds"),
                        pl.lit(None).cast(pl.Float64).alias("market_spread"),
                        pl.lit(None).cast(pl.Float64).alias("market_total"),
                        pl.lit("default").alias("market_provider"),
                        pl.lit("default").alias("market_bookmaker"),
                        pl.lit("default").alias("market_source"),
                    ]
                )

            return (
                schedule_df.join(
                    odds_df,
                    on=["game_date", "away_team_code", "home_team_code"],
                    how="left",
                )
                .with_columns(
                    [
                        pl.col("market_odds")
                        .fill_null(self.default_market_odds)
                        .cast(pl.Int64),
                        pl.col("market_provider").fill_null("default"),
                        pl.col("market_bookmaker").fill_null("default"),
                        pl.col("market_source").fill_null("default"),
                    ]
                )
                .with_columns(
                    [
                        pl.col("away_market_odds").cast(pl.Int64, strict=False),
                        pl.col("market_spread").cast(pl.Float64, strict=False),
                        pl.col("market_total").cast(pl.Float64, strict=False),
                    ]
                )
            )

        return pl.DataFrame(schema=empty_schema)

    def get_team_recent_form(
        self, season: Optional[str] = None, last_n_games: int = 10
    ) -> pl.DataFrame:
        """Returns recent points scored and allowed by team for the current season."""
        season = season or self.current_season()

        try:
            raw_df = leaguegamefinder.LeagueGameFinder(
                season_nullable=season,
                league_id_nullable="00",
            ).get_data_frames()[0]
        except Exception as exc:
            logger.warning("Failed to fetch recent team form for season %s: %s", season, exc)
            return pl.DataFrame(
                schema={
                    "team_id": pl.Int64,
                    "team_code": pl.Utf8,
                    "team_name": pl.Utf8,
                    "avg_points": pl.Float64,
                    "avg_points_allowed": pl.Float64,
                    "games_used": pl.Int64,
                }
            )

        if raw_df.empty:
            return pl.DataFrame()

        games_df = raw_df[
            ["GAME_ID", "GAME_DATE", "TEAM_ID", "TEAM_ABBREVIATION", "TEAM_NAME", "PTS"]
        ].copy()
        games_df["GAME_DATE"] = pd.to_datetime(games_df["GAME_DATE"])

        opponents_df = games_df[["GAME_ID", "TEAM_ID", "PTS"]].rename(
            columns={"TEAM_ID": "OPP_TEAM_ID", "PTS": "OPP_PTS"}
        )

        paired_df = games_df.merge(opponents_df, on="GAME_ID", how="inner")
        paired_df = paired_df[paired_df["TEAM_ID"] != paired_df["OPP_TEAM_ID"]]
        paired_df = paired_df.sort_values(["TEAM_ID", "GAME_DATE"], ascending=[True, False])
        paired_df = paired_df.groupby("TEAM_ID", group_keys=False).head(last_n_games)

        summary_df = (
            paired_df.groupby(["TEAM_ID", "TEAM_ABBREVIATION", "TEAM_NAME"], as_index=False)
            .agg(
                avg_points=("PTS", "mean"),
                avg_points_allowed=("OPP_PTS", "mean"),
                games_used=("GAME_ID", "nunique"),
            )
            .rename(
                columns={
                    "TEAM_ID": "team_id",
                    "TEAM_ABBREVIATION": "team_code",
                    "TEAM_NAME": "team_name",
                }
            )
        )

        if summary_df.empty:
            return pl.DataFrame()

        summary_df = summary_df.astype(
            {
                "team_id": "int64",
                "team_code": "string",
                "team_name": "string",
                "avg_points": "float64",
                "avg_points_allowed": "float64",
                "games_used": "int64",
            }
        )
        return pl.DataFrame(summary_df.to_dict(orient="records"))

    def get_upcoming_games_with_context(
        self,
        start_date: Optional[date | str] = None,
        days_ahead: int = 7,
        season: Optional[str] = None,
        last_n_games: int = 10,
    ) -> pl.DataFrame:
        upcoming_df = self.get_upcoming_games(start_date=start_date, days_ahead=days_ahead)
        if upcoming_df.is_empty():
            return upcoming_df

        recent_form_df = self.get_team_recent_form(season=season, last_n_games=last_n_games)
        if recent_form_df.is_empty():
            return upcoming_df

        home_form_df = recent_form_df.rename(
            {
                "team_id": "home_team_id",
                "team_code": "home_form_code",
                "team_name": "home_form_name",
                "avg_points": "home_avg_points",
                "avg_points_allowed": "home_avg_allowed",
                "games_used": "home_games_used",
            }
        )
        away_form_df = recent_form_df.rename(
            {
                "team_id": "away_team_id",
                "team_code": "away_form_code",
                "team_name": "away_form_name",
                "avg_points": "away_avg_points",
                "avg_points_allowed": "away_avg_allowed",
                "games_used": "away_games_used",
            }
        )

        return upcoming_df.join(home_form_df, on="home_team_id", how="left").join(
            away_form_df, on="away_team_id", how="left"
        )


if __name__ == "__main__":
    fetcher = NBAFetcher()
    print(fetcher.get_upcoming_games_with_context())
