from __future__ import annotations

import json
import logging
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime, timezone
from urllib.request import Request, urlopen

import polars as pl
from nba_api.stats.endpoints import leaguedashplayerstats
from nba_api.stats.static import teams

logger = logging.getLogger(__name__)

ESPN_INJURIES_URL = (
    "https://site.web.api.espn.com/apis/site/v2/sports/basketball/nba/injuries"
)

STATUS_WEIGHT: dict[str, float] = {
    "Out": 1.0,
    "Out For Season": 1.0,
    "Doubtful": 0.75,
    "Questionable": 0.3,
    "Day-To-Day": 0.2,
    "Day to Day": 0.2,
    "Probable": 0.0,
}

# Calibrated via sports-calibrate against walk-forward CV on 2024-25 + 2025-26.
# Public research suggested ~0.35 but our data shows it overcorrects: the
# RECENT_* rolling features already absorb most of the injury signal, so a
# small damping is all that's useful. At 0.05 the improvement over zero
# adjustment is within the fold-to-fold noise band, while 0.35 materially
# degrades MAE.
IMPACT_DAMPING = 0.05


def _normalize_name(name: str) -> str:
    return (
        unicodedata.normalize("NFKD", name)
        .encode("ascii", "ignore")
        .decode("ascii")
        .casefold()
        .strip()
    )


@dataclass
class InjuryReport:
    points_absent_by_team_id: dict[int, float] = field(default_factory=dict)
    unmatched_players: list[str] = field(default_factory=list)
    filtered_long_term: list[str] = field(default_factory=list)
    source: str = "none"


class InjuryProvider:
    def __init__(self, season: str | None = None, timeout: float = 10.0):
        self._season = season
        self._timeout = timeout
        self._ppg_cache: dict[str, float] | None = None
        self._recent_active_cache: set[str] | None = None
        self._team_name_to_id: dict[str, int] = {
            t["full_name"]: int(t["id"]) for t in teams.get_teams()
        }

    def fetch(self) -> InjuryReport:
        try:
            injuries = self._fetch_espn()
        except Exception as exc:
            logger.warning("injury fetch failed: %s", exc)
            return InjuryReport()

        try:
            ppg = self._player_ppg()
            recently_active = self._recent_active_players()
        except Exception as exc:
            logger.warning("player PPG fetch failed: %s", exc)
            return InjuryReport()

        points_absent: dict[int, float] = {}
        unmatched: list[str] = []
        filtered_long_term: list[str] = []
        for team_name, items in injuries.items():
            team_id = self._team_name_to_id.get(team_name)
            if team_id is None:
                continue
            total = 0.0
            for player_name, status in items:
                weight = STATUS_WEIGHT.get(status, 0.0)
                if weight == 0.0:
                    continue
                key = _normalize_name(player_name)
                ppg_val = ppg.get(key)
                if ppg_val is None:
                    unmatched.append(player_name)
                    continue
                if key not in recently_active:
                    filtered_long_term.append(player_name)
                    continue
                total += ppg_val * weight
            if total > 0:
                points_absent[team_id] = total
        return InjuryReport(
            points_absent_by_team_id=points_absent,
            unmatched_players=unmatched,
            filtered_long_term=filtered_long_term,
            source="espn",
        )

    def _fetch_espn(self) -> dict[str, list[tuple[str, str]]]:
        req = Request(ESPN_INJURIES_URL, headers={"User-Agent": "Mozilla/5.0"})
        with urlopen(req, timeout=self._timeout) as resp:
            data = json.loads(resp.read())

        result: dict[str, list[tuple[str, str]]] = {}
        for team_block in data.get("injuries", []):
            team_name = team_block.get("displayName", "")
            pairs: list[tuple[str, str]] = []
            for inj in team_block.get("injuries", []) or []:
                athlete = inj.get("athlete") or {}
                name = athlete.get("displayName", "")
                status = inj.get("status", "")
                if name and status:
                    pairs.append((name, status))
            if team_name and pairs:
                result[team_name] = pairs
        return result

    def _player_ppg(self) -> dict[str, float]:
        """Season PPG for every player who has appeared this season."""
        if self._ppg_cache is not None:
            return self._ppg_cache
        season = self._season or self._current_season()
        df = leaguedashplayerstats.LeagueDashPlayerStats(
            season=season,
            season_type_all_star="Regular Season",
        ).get_data_frames()[0]
        ppg: dict[str, float] = {}
        for _, row in df.iterrows():
            gp = float(row.get("GP", 0) or 0)
            if gp > 0:
                ppg[_normalize_name(row["PLAYER_NAME"])] = float(row["PTS"]) / gp
        self._ppg_cache = ppg
        return ppg

    def _recent_active_players(self) -> set[str]:
        """Players with any appearances in the last 10 games.

        Long-term injuries are already baked into the model's RECENT_* rolling
        features, so only players who were playing recently should contribute
        an injury adjustment.
        """
        if self._recent_active_cache is not None:
            return self._recent_active_cache
        season = self._season or self._current_season()
        df = leaguedashplayerstats.LeagueDashPlayerStats(
            season=season,
            season_type_all_star="Regular Season",
            last_n_games=10,
        ).get_data_frames()[0]
        active = {
            _normalize_name(row["PLAYER_NAME"])
            for _, row in df.iterrows()
            if float(row.get("GP", 0) or 0) > 0
        }
        self._recent_active_cache = active
        return active

    @staticmethod
    def _current_season() -> str:
        today = datetime.now(timezone.utc).date()
        start_year = today.year if today.month >= 10 else today.year - 1
        end_year_short = str((start_year + 1) % 100).zfill(2)
        return f"{start_year}-{end_year_short}"


def adjust_projections_for_injuries(
    projections_df: pl.DataFrame,
    report: InjuryReport,
    damping: float = IMPACT_DAMPING,
) -> pl.DataFrame:
    """Attach per-team points-absent and subtract a damped adjustment from expected scores."""
    required = {"home_team_id", "away_team_id", "home_expected_score", "away_expected_score"}
    if not required.issubset(projections_df.columns):
        return projections_df

    absent = report.points_absent_by_team_id

    def lookup(series: pl.Series) -> pl.Series:
        values: list[float] = []
        for raw in series.to_list():
            try:
                key = int(raw) if raw is not None else None
            except (TypeError, ValueError):
                key = None
            values.append(float(absent.get(key, 0.0)) if key is not None else 0.0)
        return pl.Series(values, dtype=pl.Float64)

    home_abs = lookup(projections_df["home_team_id"])
    away_abs = lookup(projections_df["away_team_id"])

    projections_df = projections_df.with_columns(
        [
            home_abs.alias("home_points_absent"),
            away_abs.alias("away_points_absent"),
        ]
    )
    if not absent:
        return projections_df

    return projections_df.with_columns(
        [
            (pl.col("home_expected_score") - pl.col("home_points_absent") * damping).alias(
                "home_expected_score"
            ),
            (pl.col("away_expected_score") - pl.col("away_points_absent") * damping).alias(
                "away_expected_score"
            ),
        ]
    )
