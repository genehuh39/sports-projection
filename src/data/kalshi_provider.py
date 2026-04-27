from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from urllib.parse import urlencode
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)

KALSHI_API_BASE = "https://api.elections.kalshi.com/trade-api/v2"
NBA_GAME_SERIES = "KXNBAGAME"

_MONTH_ABBREV = {
    "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
    "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
}

# KXNBAGAME-{YY}{MMM}{DD}{AWAY3}{HOME3}
_EVENT_TICKER_RE = re.compile(
    r"^KXNBAGAME-(\d{2})([A-Z]{3})(\d{2})([A-Z]{3})([A-Z]{3})$"
)


@dataclass
class KalshiGameMarket:
    """A single NBA game on Kalshi with both teams' YES contract prices.

    Prices are in 0-1 probability units (Kalshi reports cents 1-99 over the
    wire; we normalize). ``None`` indicates no resting order on that side.
    """

    event_ticker: str
    game_date: date
    home_team_abbr: str
    away_team_abbr: str
    home_market_ticker: str
    away_market_ticker: str
    home_yes_bid: float | None = None
    home_yes_ask: float | None = None
    away_yes_bid: float | None = None
    away_yes_ask: float | None = None
    home_volume: int | None = None
    away_volume: int | None = None
    title: str = ""


@dataclass
class KalshiSnapshot:
    games: list[KalshiGameMarket] = field(default_factory=list)
    source: str = "none"


def _parse_event_ticker(ticker: str) -> tuple[date, str, str] | None:
    """Return (game_date, away_abbr, home_abbr) for a KXNBAGAME event ticker."""
    match = _EVENT_TICKER_RE.match(ticker)
    if not match:
        return None
    yy, mmm, dd, away, home = match.groups()
    month = _MONTH_ABBREV.get(mmm)
    if month is None:
        return None
    try:
        return date(2000 + int(yy), month, int(dd)), away, home
    except ValueError:
        return None


def _to_prob(cents: int | float | None) -> float | None:
    if cents is None:
        return None
    try:
        return float(cents) / 100.0
    except (TypeError, ValueError):
        return None


class KalshiProvider:
    """Read-only client for Kalshi's public NBA game markets."""

    def __init__(self, timeout: float = 10.0, page_size: int = 100):
        self._timeout = timeout
        self._page_size = page_size

    def fetch(self) -> KalshiSnapshot:
        try:
            events = self._list_events()
        except Exception as exc:
            logger.warning("kalshi events fetch failed: %s", exc)
            return KalshiSnapshot()

        try:
            markets = self._list_markets_for_series()
        except Exception as exc:
            logger.warning("kalshi markets fetch failed: %s", exc)
            return KalshiSnapshot()

        markets_by_event: dict[str, list[dict]] = {}
        for m in markets:
            markets_by_event.setdefault(m.get("event_ticker", ""), []).append(m)

        games: list[KalshiGameMarket] = []
        for event in events:
            ticker = event.get("event_ticker", "")
            parsed = _parse_event_ticker(ticker)
            if parsed is None:
                continue
            game_date, away_abbr, home_abbr = parsed
            event_markets = markets_by_event.get(ticker, [])
            home_market = next(
                (m for m in event_markets if m.get("ticker", "").endswith(f"-{home_abbr}")),
                None,
            )
            away_market = next(
                (m for m in event_markets if m.get("ticker", "").endswith(f"-{away_abbr}")),
                None,
            )
            if home_market is None or away_market is None:
                continue

            games.append(
                KalshiGameMarket(
                    event_ticker=ticker,
                    game_date=game_date,
                    home_team_abbr=home_abbr,
                    away_team_abbr=away_abbr,
                    home_market_ticker=home_market["ticker"],
                    away_market_ticker=away_market["ticker"],
                    home_yes_bid=_to_prob(home_market.get("yes_bid")),
                    home_yes_ask=_to_prob(home_market.get("yes_ask")),
                    away_yes_bid=_to_prob(away_market.get("yes_bid")),
                    away_yes_ask=_to_prob(away_market.get("yes_ask")),
                    home_volume=home_market.get("volume"),
                    away_volume=away_market.get("volume"),
                    title=event.get("title") or event.get("sub_title", ""),
                )
            )

        return KalshiSnapshot(games=games, source="kalshi")

    def _list_events(self) -> list[dict]:
        events: list[dict] = []
        cursor: str | None = None
        for _ in range(50):
            params: dict[str, str | int] = {
                "series_ticker": NBA_GAME_SERIES,
                "status": "open",
                "limit": self._page_size,
            }
            if cursor:
                params["cursor"] = cursor
            payload = self._get("/events", params)
            page = payload.get("events", []) or []
            events.extend(page)
            cursor = payload.get("cursor") or None
            if not cursor or not page:
                break
        return events

    def _list_markets_for_series(self) -> list[dict]:
        markets: list[dict] = []
        cursor: str | None = None
        for _ in range(50):
            params: dict[str, str | int] = {
                "series_ticker": NBA_GAME_SERIES,
                "status": "open",
                "limit": self._page_size,
            }
            if cursor:
                params["cursor"] = cursor
            payload = self._get("/markets", params)
            page = payload.get("markets", []) or []
            markets.extend(page)
            cursor = payload.get("cursor") or None
            if not cursor or not page:
                break
        return markets

    def _get(self, path: str, params: dict[str, str | int] | None = None) -> dict:
        url = f"{KALSHI_API_BASE}{path}"
        if params:
            url = f"{url}?{urlencode(params)}"
        req = Request(url, headers={"User-Agent": "sports-projection/0.5"})
        with urlopen(req, timeout=self._timeout) as resp:
            return json.loads(resp.read())


def compute_kalshi_edges(
    snapshot: KalshiSnapshot,
    games_df,
    home_abbr_col: str = "home_team_code",
    away_abbr_col: str = "away_team_code",
    game_date_col: str = "game_date",
    home_prob_col: str = "home_win_prob",
):
    """Annotate ``games_df`` with Kalshi quotes and per-side edge.

    ``games_df`` should be a polars DataFrame containing at least the team
    abbreviation columns and a model home_win_prob column. Returns a new
    DataFrame with home_yes_bid/ask, away_yes_bid/ask, edge_home,
    edge_away columns appended (None when unmatched).
    """
    import polars as pl

    if not snapshot.games:
        return games_df.with_columns(
            [
                pl.lit(None).cast(pl.Float64).alias("kalshi_home_yes_ask"),
                pl.lit(None).cast(pl.Float64).alias("kalshi_away_yes_ask"),
                pl.lit(None).cast(pl.Float64).alias("edge_home"),
                pl.lit(None).cast(pl.Float64).alias("edge_away"),
            ]
        )

    by_key: dict[tuple[str, str, str], KalshiGameMarket] = {}
    for g in snapshot.games:
        key = (g.home_team_abbr, g.away_team_abbr, g.game_date.isoformat())
        by_key[key] = g

    rows = games_df.to_dicts()
    out = []
    for row in rows:
        home = (row.get(home_abbr_col) or "").upper()
        away = (row.get(away_abbr_col) or "").upper()
        gd = row.get(game_date_col)
        if isinstance(gd, (date, datetime)):
            gd_iso = gd.isoformat()[:10]
        else:
            gd_iso = str(gd)[:10]
        match = by_key.get((home, away, gd_iso))
        model_p = row.get(home_prob_col)
        home_ask = match.home_yes_ask if match else None
        away_ask = match.away_yes_ask if match else None
        edge_home = (
            (model_p - home_ask)
            if (match and home_ask is not None and model_p is not None)
            else None
        )
        edge_away = (
            ((1.0 - model_p) - away_ask)
            if (match and away_ask is not None and model_p is not None)
            else None
        )
        row.update(
            {
                "kalshi_home_yes_bid": match.home_yes_bid if match else None,
                "kalshi_home_yes_ask": home_ask,
                "kalshi_away_yes_bid": match.away_yes_bid if match else None,
                "kalshi_away_yes_ask": away_ask,
                "kalshi_event_ticker": match.event_ticker if match else None,
                "edge_home": edge_home,
                "edge_away": edge_away,
            }
        )
        out.append(row)
    return pl.DataFrame(out)
