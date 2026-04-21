import json
import logging
import os
from datetime import date, datetime, timezone
from typing import Optional
from urllib.parse import urlencode
from urllib.request import urlopen

import polars as pl

logger = logging.getLogger(__name__)

EMPTY_ODDS_SCHEMA = {
    "game_date": pl.Utf8,
    "home_team_code": pl.Utf8,
    "away_team_code": pl.Utf8,
    "market_odds": pl.Int64,
    "away_market_odds": pl.Int64,
    "market_spread": pl.Float64,
    "market_total": pl.Float64,
    "market_provider": pl.Utf8,
    "market_bookmaker": pl.Utf8,
    "market_source": pl.Utf8,
    "provider_priority": pl.Int64,
}

TEAM_CODE_ALIASES = {
    "SA": "SAS",
    "GS": "GSW",
    "NO": "NOP",
    "NY": "NYK",
}

TEAM_NAME_TO_CODE = {
    "ATLANTA HAWKS": "ATL",
    "BOSTON CELTICS": "BOS",
    "BROOKLYN NETS": "BKN",
    "CHARLOTTE HORNETS": "CHA",
    "CHICAGO BULLS": "CHI",
    "CLEVELAND CAVALIERS": "CLE",
    "DALLAS MAVERICKS": "DAL",
    "DENVER NUGGETS": "DEN",
    "DETROIT PISTONS": "DET",
    "GOLDEN STATE WARRIORS": "GSW",
    "HOUSTON ROCKETS": "HOU",
    "INDIANA PACERS": "IND",
    "LOS ANGELES CLIPPERS": "LAC",
    "LOS ANGELES LAKERS": "LAL",
    "MEMPHIS GRIZZLIES": "MEM",
    "MIAMI HEAT": "MIA",
    "MILWAUKEE BUCKS": "MIL",
    "MINNESOTA TIMBERWOLVES": "MIN",
    "NEW ORLEANS PELICANS": "NOP",
    "NEW YORK KNICKS": "NYK",
    "OKLAHOMA CITY THUNDER": "OKC",
    "ORLANDO MAGIC": "ORL",
    "PHILADELPHIA 76ERS": "PHI",
    "PHOENIX SUNS": "PHX",
    "PORTLAND TRAIL BLAZERS": "POR",
    "SACRAMENTO KINGS": "SAC",
    "SAN ANTONIO SPURS": "SAS",
    "TORONTO RAPTORS": "TOR",
    "UTAH JAZZ": "UTA",
    "WASHINGTON WIZARDS": "WAS",
}


def empty_odds_frame() -> pl.DataFrame:
    return pl.DataFrame(schema=EMPTY_ODDS_SCHEMA)


def normalize_team_code(code: Optional[str]) -> str:
    normalized = str(code or "").upper().strip()
    return TEAM_CODE_ALIASES.get(normalized, normalized)


def team_name_to_code(name: Optional[str]) -> str:
    normalized = str(name or "").upper().strip()
    return TEAM_NAME_TO_CODE.get(normalized, normalize_team_code(normalized))


class EspnOddsProvider:
    SCOREBOARD_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"

    def __init__(self, priority: int = 20):
        self.priority = priority

    def _scoreboard_json(self, game_date: date) -> dict:
        query = urlencode({"dates": game_date.strftime("%Y%m%d")})
        with urlopen(f"{self.SCOREBOARD_URL}?{query}", timeout=15) as response:
            return json.loads(response.read().decode("utf-8"))

    def fetch(self, game_date: date) -> pl.DataFrame:
        try:
            payload = self._scoreboard_json(game_date)
        except Exception as exc:
            logger.warning("Failed to fetch ESPN odds for %s: %s", game_date, exc)
            return empty_odds_frame()

        records: list[dict] = []
        for event in payload.get("events", []):
            competitions = event.get("competitions", [])
            if not competitions:
                continue

            competition = competitions[0]
            competitors = competition.get("competitors", [])
            home = next((c for c in competitors if c.get("homeAway") == "home"), None)
            away = next((c for c in competitors if c.get("homeAway") == "away"), None)
            odds_list = competition.get("odds") or []
            if home is None or away is None or not odds_list:
                continue

            odds = odds_list[0]
            moneyline = odds.get("moneyline", {})
            spread = odds.get("pointSpread", {})
            total = odds.get("total", {})
            bookmaker = odds.get("provider", {}).get("displayName") or odds.get("provider", {}).get("name") or "Unknown"

            def parse_int(value):
                try:
                    return int(value) if value is not None else None
                except (TypeError, ValueError):
                    return None

            def parse_float(value):
                try:
                    return float(value) if value is not None else None
                except (TypeError, ValueError):
                    return None

            total_line = (((total.get("over") or {}).get("close") or {}).get("line"))
            try:
                total_line = float(str(total_line).lstrip("ou")) if total_line is not None else None
            except (TypeError, ValueError):
                total_line = None

            records.append(
                {
                    "game_date": game_date.isoformat(),
                    "home_team_code": normalize_team_code(home.get("team", {}).get("abbreviation")),
                    "away_team_code": normalize_team_code(away.get("team", {}).get("abbreviation")),
                    "market_odds": parse_int((((moneyline.get("home") or {}).get("close") or {}).get("odds"))),
                    "away_market_odds": parse_int((((moneyline.get("away") or {}).get("close") or {}).get("odds"))),
                    "market_spread": parse_float((((spread.get("home") or {}).get("close") or {}).get("line"))),
                    "market_total": total_line,
                    "market_provider": "ESPN",
                    "market_bookmaker": bookmaker,
                    "market_source": f"ESPN / {bookmaker}",
                    "provider_priority": self.priority,
                }
            )

        return empty_odds_frame() if not records else pl.DataFrame(records)


class TheOddsApiProvider:
    BASE_URL = "https://api.the-odds-api.com/v4/sports/basketball_nba/odds"

    def __init__(self, api_key: Optional[str] = None, priority: int = 10):
        self.api_key = api_key or os.getenv("THE_ODDS_API_KEY")
        self.priority = priority

    def fetch(self, game_date: date) -> pl.DataFrame:
        if not self.api_key:
            return empty_odds_frame()

        query = urlencode(
            {
                "apiKey": self.api_key,
                "regions": "us",
                "markets": "h2h,spreads,totals",
                "oddsFormat": "american",
                "dateFormat": "iso",
            }
        )

        try:
            with urlopen(f"{self.BASE_URL}?{query}", timeout=20) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except Exception as exc:
            logger.warning("Failed to fetch The Odds API odds for %s: %s", game_date, exc)
            return empty_odds_frame()

        records: list[dict] = []
        for game in payload:
            commence_time = game.get("commence_time")
            if not commence_time:
                continue
            try:
                game_day = datetime.fromisoformat(commence_time.replace("Z", "+00:00")).astimezone(timezone.utc).date()
            except ValueError:
                continue
            if game_day != game_date:
                continue

            home_team = team_name_to_code(game.get("home_team"))
            away_team = team_name_to_code(game.get("away_team"))
            bookmakers = game.get("bookmakers") or []
            if not bookmakers:
                continue

            bookmaker = bookmakers[0]
            markets = {market.get("key"): market for market in bookmaker.get("markets", [])}

            home_ml = away_ml = spread = total = None

            h2h = markets.get("h2h", {})
            for outcome in h2h.get("outcomes", []):
                name = outcome.get("name")
                price = outcome.get("price")
                if name == game.get("home_team"):
                    home_ml = price
                elif name == game.get("away_team"):
                    away_ml = price

            spreads = markets.get("spreads", {})
            for outcome in spreads.get("outcomes", []):
                if outcome.get("name") == game.get("home_team"):
                    spread = outcome.get("point")
                    break

            totals = markets.get("totals", {})
            for outcome in totals.get("outcomes", []):
                if outcome.get("name") == "Over":
                    total = outcome.get("point")
                    break

            records.append(
                {
                    "game_date": game_date.isoformat(),
                    "home_team_code": home_team,
                    "away_team_code": away_team,
                    "market_odds": int(home_ml) if home_ml is not None else None,
                    "away_market_odds": int(away_ml) if away_ml is not None else None,
                    "market_spread": float(spread) if spread is not None else None,
                    "market_total": float(total) if total is not None else None,
                    "market_provider": "The Odds API",
                    "market_bookmaker": bookmaker.get("title") or "Unknown",
                    "market_source": f"The Odds API / {bookmaker.get('title') or 'Unknown'}",
                    "provider_priority": self.priority,
                }
            )

        return empty_odds_frame() if not records else pl.DataFrame(records)
