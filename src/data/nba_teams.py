from __future__ import annotations

from functools import lru_cache

from nba_api.stats.static import teams as _nba_static_teams


@lru_cache(maxsize=1)
def team_abbr_to_id() -> dict[str, int]:
    """Cached map of NBA team abbreviation (e.g. ``"LAL"``) to its team_id."""
    return {t["abbreviation"]: int(t["id"]) for t in _nba_static_teams.get_teams()}
