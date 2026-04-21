import polars as pl
from datetime import datetime, timedelta
import random

class MockDataGenerator:
    """Generates fake sports data for development/testing purposes."""

    def __init__(self):
        self.teams = [
            {"id": "team_1", "name": "Lakers", "sport": "nba"},
            {"id": "team_2", "name": "Warriors", "sport": "nba"},
            {"id": "team_3", "name": "Celtics", "sport": "nba"},
            {"id": "team_4", "name": "Bucks", "sport": "nba"},
        ]

    def generate_games(self, num_games: int = 100) -> pl.DataFrame:
        """Generates a polars DataFrame of historical games."""
        data = []
        start_date = datetime.now() - timedelta(days=365)
        
        for i in range(num_games):
            date = start_date + timedelta(days=random.randint(0, 365))
            t1_idx = random.randint(0, len(self.teams) - 1)
            t2_idx = (t1_idx + 1) % len(self.teams)
            
            team1 = self.teams[t1_idx]
            team2 = self.teams[t2_idx]

            # Random scores for 'historical' data
            home_score = random.uniform(80, 130)
            away_score = random.uniform(80, 130)

            data.append({
                "game_id": f"game_{i}",
                "date": date,
                "home_team_id": team1["id"],
                "away_team_id": team2["id"],
                "home_score": home_score,
                "away_score": away_score,
                "sport": team1["sport"],
                "status": "final"
            })

        return pl.DataFrame(data)

if __name__ == "__main__":
    # Quick test
    generator = MockDataGenerator()
    df = generator.generate_games(10)
    print(df)
