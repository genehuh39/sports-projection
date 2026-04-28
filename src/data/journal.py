from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Optional

DEFAULT_JOURNAL_PATH = Path("data/paper_trades.db")

# String constants used as enum-like values in the SQLite schema.
SIDE_HOME = "home"
SIDE_AWAY = "away"
STATUS_OPEN = "open"
STATUS_SETTLED = "settled"
VENUE_KALSHI = "kalshi"
VENUE_POLYMARKET = "polymarket"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS paper_trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    venue TEXT NOT NULL,
    game_id TEXT NOT NULL,
    game_date TEXT NOT NULL,
    home_team_abbr TEXT NOT NULL,
    away_team_abbr TEXT NOT NULL,
    side TEXT NOT NULL,
    model_prob REAL NOT NULL,
    market_price REAL NOT NULL,
    edge REAL NOT NULL,
    stake REAL NOT NULL DEFAULT 1.0,
    status TEXT NOT NULL DEFAULT 'open',
    home_won INTEGER,
    side_won INTEGER,
    realized_pnl REAL,
    settled_at TEXT,
    notes TEXT,
    UNIQUE(venue, game_id, side)
);
"""


@dataclass
class PaperTrade:
    venue: str
    game_id: str
    game_date: str  # ISO YYYY-MM-DD
    home_team_abbr: str
    away_team_abbr: str
    side: str  # 'home' or 'away'
    model_prob: float
    market_price: float
    edge: float
    stake: float = 1.0
    notes: str = ""
    id: Optional[int] = None
    created_at: Optional[str] = None
    status: str = STATUS_OPEN
    home_won: Optional[int] = None
    side_won: Optional[int] = None
    realized_pnl: Optional[float] = None
    settled_at: Optional[str] = None


def yes_pnl(market_price: float, side_won: bool) -> float:
    """ROI per unit stake on a YES contract at given price.

    Win: payout 1 / cost p - 1 = (1-p)/p
    Lose: -1
    """
    if side_won:
        return (1.0 - market_price) / market_price
    return -1.0


class PaperTradeJournal:
    def __init__(self, path: str | Path = DEFAULT_JOURNAL_PATH):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def append(self, trade: PaperTrade) -> bool:
        """Returns True if inserted, False if (venue, game_id, side) is a duplicate."""
        created_at = trade.created_at or datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT OR IGNORE INTO paper_trades (
                    created_at, venue, game_id, game_date,
                    home_team_abbr, away_team_abbr, side,
                    model_prob, market_price, edge, stake,
                    status, notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    created_at, trade.venue, trade.game_id, trade.game_date,
                    trade.home_team_abbr, trade.away_team_abbr, trade.side,
                    float(trade.model_prob), float(trade.market_price),
                    float(trade.edge), float(trade.stake),
                    trade.status, trade.notes,
                ),
            )
            return cursor.rowcount > 0

    def list_open_past_games(self, today_iso: str) -> list[PaperTrade]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM paper_trades
                WHERE status = 'open' AND game_date < ?
                ORDER BY game_date, id
                """,
                (today_iso,),
            ).fetchall()
        return [self._row_to_trade(r) for r in rows]

    def list_all(self) -> list[PaperTrade]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM paper_trades ORDER BY game_date, id"
            ).fetchall()
        return [self._row_to_trade(r) for r in rows]

    def list_settled(self) -> list[PaperTrade]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM paper_trades
                WHERE status = 'settled'
                ORDER BY game_date, id
                """
            ).fetchall()
        return [self._row_to_trade(r) for r in rows]

    def mark_settled(
        self,
        trade_id: int,
        home_won: bool,
        side_won: bool,
        realized_pnl: float,
    ) -> None:
        settled_at = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE paper_trades
                SET status = 'settled',
                    home_won = ?,
                    side_won = ?,
                    realized_pnl = ?,
                    settled_at = ?
                WHERE id = ?
                """,
                (int(home_won), int(side_won), float(realized_pnl), settled_at, trade_id),
            )

    @staticmethod
    def _row_to_trade(row: sqlite3.Row) -> PaperTrade:
        return PaperTrade(
            id=row["id"],
            created_at=row["created_at"],
            venue=row["venue"],
            game_id=row["game_id"],
            game_date=row["game_date"],
            home_team_abbr=row["home_team_abbr"],
            away_team_abbr=row["away_team_abbr"],
            side=row["side"],
            model_prob=row["model_prob"],
            market_price=row["market_price"],
            edge=row["edge"],
            stake=row["stake"],
            status=row["status"],
            home_won=row["home_won"],
            side_won=row["side_won"],
            realized_pnl=row["realized_pnl"],
            settled_at=row["settled_at"],
            notes=row["notes"] or "",
        )
