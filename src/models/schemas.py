from pydantic import BaseModel, Field
from datetime import datetime
from typing import List, Optional

class Player(BaseModel):
    id: str
    name: str
    sport: str  # 'nba', 'mlb', 'nfl'
    position: Optional[str] = None

class Team(BaseModel):
    id: str
    name: str
    sport: str

class Game(BaseModel):
    id: str
    date: datetime
    home_team_id: str
    away_team_id: str
    home_score: Optional[float] = None
    away_score: Optional[float] = None
    sport: str  # 'nba', 'mlb', 'nfl'
    status: str  # 'scheduled', 'live', 'final'

class Projection(BaseModel):
    game_id: str
    home_team_expected_score: float
    away_team_expected_score: float
    confidence: float  # 0 to 1
    metadata: dict = Field(default_factory=dict)
