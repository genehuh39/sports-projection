from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import date, datetime
from urllib.parse import urlencode
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)

POLYMARKET_GAMMA_BASE = "https://gamma-api.polymarket.com"

# Game-line slugs look like "nba-det-orl-2026-04-27".
# Spread / totals slugs look like "spread-pistons-3-5-2026-04-27" — skip those.
_NBA_SLUG_RE = re.compile(
    r"^nba-([a-z]{3})-([a-z]{3})-(\d{4}-\d{2}-\d{2})$"
)


@dataclass
class PolymarketGameMarket:
    """A Polymarket NBA moneyline game market.

    Polymarket's gamma API exposes a single ``outcomePrices`` per market
    (last trade price for each side), not a bid/ask book. We surface
    those as ``home_yes_ask`` / ``away_yes_ask`` to keep the same shape
    as Kalshi's market struct so downstream join code can be shared,
    but they are *last-trade* prices, not actionable asks.
    """

    market_id: str
    slug: str
    game_date: date
    home_team_abbr: str
    away_team_abbr: str
    home_yes_bid: float | None = None
    home_yes_ask: float | None = None
    away_yes_bid: float | None = None
    away_yes_ask: float | None = None
    home_volume: float | None = None
    away_volume: float | None = None
    title: str = ""


@dataclass
class PolymarketSnapshot:
    games: list[PolymarketGameMarket] = field(default_factory=list)
    source: str = "none"


def _parse_nba_slug(slug: str) -> tuple[date, str, str] | None:
    match = _NBA_SLUG_RE.match(slug)
    if not match:
        return None
    away_abbr, home_abbr, date_str = match.groups()
    try:
        return date.fromisoformat(date_str), away_abbr.upper(), home_abbr.upper()
    except ValueError:
        return None


def _parse_outcome_prices(raw) -> list[float] | None:
    """outcomePrices is sometimes a JSON-string, sometimes a list."""
    if raw is None:
        return None
    if isinstance(raw, list):
        items = raw
    elif isinstance(raw, str):
        try:
            items = json.loads(raw)
        except json.JSONDecodeError:
            return None
    else:
        return None
    try:
        return [float(x) for x in items]
    except (TypeError, ValueError):
        return None


class PolymarketProvider:
    """Read-only client for Polymarket's NBA moneyline game markets."""

    def __init__(self, timeout: float = 10.0, page_size: int = 200):
        self._timeout = timeout
        self._page_size = page_size

    def fetch(self) -> PolymarketSnapshot:
        try:
            markets = self._list_active_markets()
        except Exception as exc:
            logger.warning("polymarket fetch failed: %s", exc)
            return PolymarketSnapshot()

        games: list[PolymarketGameMarket] = []
        for m in markets:
            slug = m.get("slug") or ""
            parsed = _parse_nba_slug(slug)
            if parsed is None:
                continue
            game_date, away_abbr, home_abbr = parsed

            prices = _parse_outcome_prices(m.get("outcomePrices"))
            if prices is None or len(prices) < 2:
                continue
            # Outcomes order matches question order: away first, home second
            away_price = prices[0]
            home_price = prices[1]

            games.append(
                PolymarketGameMarket(
                    market_id=str(m.get("conditionId") or m.get("id") or slug),
                    slug=slug,
                    game_date=game_date,
                    home_team_abbr=home_abbr,
                    away_team_abbr=away_abbr,
                    home_yes_bid=home_price,
                    home_yes_ask=home_price,
                    away_yes_bid=away_price,
                    away_yes_ask=away_price,
                    home_volume=m.get("volume24hr"),
                    away_volume=m.get("volume24hr"),
                    title=m.get("question") or "",
                )
            )

        return PolymarketSnapshot(games=games, source="polymarket")

    def _list_active_markets(self) -> list[dict]:
        # Sorted by 24-hour volume so the highest-liquidity NBA markets surface
        # in the first page; we filter to NBA slugs after the fact. One pass is
        # usually enough; paginate as a safety net.
        markets: list[dict] = []
        offset = 0
        for _ in range(20):
            params = {
                "active": "true",
                "closed": "false",
                "order": "volume24hr",
                "ascending": "false",
                "limit": self._page_size,
                "offset": offset,
            }
            payload = self._get("/markets", params)
            page = payload if isinstance(payload, list) else payload.get("markets", [])
            if not page:
                break
            markets.extend(page)
            # Stop early once the page is dominated by non-NBA markets.
            nba_in_page = sum(1 for m in page if (m.get("slug") or "").startswith("nba-"))
            if nba_in_page == 0 and offset >= 200:
                break
            offset += self._page_size
        return markets

    def _get(self, path: str, params: dict[str, str | int]):
        url = f"{POLYMARKET_GAMMA_BASE}{path}?{urlencode(params)}"
        req = Request(url, headers={"User-Agent": "sports-projection/0.6"})
        with urlopen(req, timeout=self._timeout) as resp:
            return json.loads(resp.read())


def compute_polymarket_edges(
    snapshot: PolymarketSnapshot,
    games_df,
    home_abbr_col: str = "home_team_code",
    away_abbr_col: str = "away_team_code",
    game_date_col: str = "game_date",
    home_prob_col: str = "home_win_prob",
):
    """Annotate ``games_df`` with Polymarket prices and per-side edge.

    Mirrors compute_kalshi_edges. Returns a DataFrame with new columns
    polymarket_home_price, polymarket_away_price, polymarket_slug,
    edge_home_pm, edge_away_pm.
    """
    import polars as pl

    if not snapshot.games:
        return games_df.with_columns(
            [
                pl.lit(None).cast(pl.Float64).alias("polymarket_home_price"),
                pl.lit(None).cast(pl.Float64).alias("polymarket_away_price"),
                pl.lit(None).cast(pl.Utf8).alias("polymarket_slug"),
                pl.lit(None).cast(pl.Float64).alias("edge_home_pm"),
                pl.lit(None).cast(pl.Float64).alias("edge_away_pm"),
            ]
        )

    by_key: dict[tuple[str, str, str], PolymarketGameMarket] = {}
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
        home_price = match.home_yes_ask if match else None
        away_price = match.away_yes_ask if match else None
        edge_home = (
            (model_p - home_price)
            if (match and home_price is not None and model_p is not None)
            else None
        )
        edge_away = (
            ((1.0 - model_p) - away_price)
            if (match and away_price is not None and model_p is not None)
            else None
        )
        row.update(
            {
                "polymarket_home_price": home_price,
                "polymarket_away_price": away_price,
                "polymarket_slug": match.slug if match else None,
                "edge_home_pm": edge_home,
                "edge_away_pm": edge_away,
            }
        )
        out.append(row)
    return pl.DataFrame(out)
