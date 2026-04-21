from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime

class Player(BaseModel):
    id: str
    name: str
    sport: str = "nba"

class Team(BaseModel):
    id: str
    name: str
    sport: str = "nba"

class PlayerStats(BaseModel):
    player_id: str
    name: str
    position: str
    games_played: int
    points_per_game: float
    rebounds_per_game: float
    assists_per_game: float
    minutes_per_game: float

class GameMatchup(BaseModel):
    game_id: str
    home_team_id: str
    away_team_id: str
    date: datetime
    home_roster: List[PlayerStats]
    away_roster: List[PlayerStats]

class PlayerProjection(BaseModel):
    player_id: str
    player_name: str
    expected_points: float
    expected_rebounds: float
    expected_assists: float
    projection_type: str  # 'points', 'rebounds', etc.

class TeamProjection(BaseModel):
    team_id: str
    expected_total_score: float
    projected_players: List[PlayerProjection]

class MatchupProjection(BaseModel):
    game_id: str
    home_team_projection: TeamProjection
    away_team_projection: TeamProjection
    win_probability_home: float
