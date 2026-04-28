from __future__ import annotations

from datetime import date as date_cls
from typing import Iterable, Protocol

import polars as pl

from src.data.nba_teams import team_abbr_to_id


class _MarketLike(Protocol):
    game_date: date_cls
    home_team_abbr: str
    away_team_abbr: str


def build_upcoming_from_market_games(
    markets: Iterable[_MarketLike],
    *,
    future_only: bool = False,
    today: date_cls | None = None,
    id_attr: str = "event_ticker",
    fallback_id_attr: str | None = "slug",
) -> pl.DataFrame:
    """Construct the upcoming-games DataFrame the projection engine consumes.

    Each input market only needs ``game_date`` / ``home_team_abbr`` /
    ``away_team_abbr`` plus a stable identifier (``id_attr``, with optional
    fallback). Markets whose abbreviations don't resolve to NBA team_ids
    (e.g. WNBA, all-star events) are skipped.
    """
    abbr_to_id = team_abbr_to_id()
    cutoff = today or date_cls.today()
    rows: list[dict] = []
    for market in markets:
        if future_only and market.game_date < cutoff:
            continue
        home_id = abbr_to_id.get(market.home_team_abbr)
        away_id = abbr_to_id.get(market.away_team_abbr)
        if home_id is None or away_id is None:
            continue
        market_id = getattr(market, id_attr, None)
        if market_id is None and fallback_id_attr is not None:
            market_id = getattr(market, fallback_id_attr, None)
        rows.append(
            {
                "game_id": market_id,
                "game_date": market.game_date.isoformat(),
                "home_team_id": home_id,
                "away_team_id": away_id,
                "home_team_code": market.home_team_abbr,
                "away_team_code": market.away_team_abbr,
                "market_odds": -110,
            }
        )
    return pl.DataFrame(rows) if rows else pl.DataFrame()
