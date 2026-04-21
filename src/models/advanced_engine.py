import numpy as np
import polars as pl

from src.models.trained_nba_model import NBAModelManager


class AdvancedModelingEngine:
    """Generates game projections from a trained model when available, otherwise falls back."""

    def __init__(
        self,
        monte_carlo_sims: int = 1000,
        auto_train: bool = False,
        use_trained_model: bool = False,
    ):
        self.sims = monte_carlo_sims
        self.use_trained_model = use_trained_model
        self.model_manager = NBAModelManager(auto_train=auto_train)

    def simulate_player_performance(self, player_stats: dict, minutes_available: float) -> dict:
        base_rate = player_stats.get("ppg", 20.0)
        scale_factor = minutes_available / 36.0
        expected_val = base_rate * scale_factor
        simulated_val = np.random.poisson(expected_val)

        return {
            "player_id": player_stats["id"],
            "name": player_stats["name"],
            "points": simulated_val,
            "rebounds": np.random.poisson(player_stats.get("rpg", 5)) * scale_factor,
            "assists": np.random.poisson(player_stats.get("apg", 5)) * scale_factor,
        }

    def _fallback_projections(self, upcoming_df: pl.DataFrame) -> pl.DataFrame:
        home_fallback = (
            pl.lit(105.0)
            + (pl.col("home_team_id").cast(pl.Utf8).hash().cast(pl.Float64) % 15)
        )
        away_fallback = (
            pl.lit(100.0)
            + (pl.col("away_team_id").cast(pl.Utf8).hash().cast(pl.Float64) % 15)
        )

        required_context = {
            "home_avg_points",
            "away_avg_points",
            "home_avg_allowed",
            "away_avg_allowed",
        }
        has_context = required_context.issubset(set(upcoming_df.columns))

        if has_context:
            context_available = pl.all_horizontal(
                pl.col("home_avg_points").is_not_null(),
                pl.col("away_avg_points").is_not_null(),
                pl.col("home_avg_allowed").is_not_null(),
                pl.col("away_avg_allowed").is_not_null(),
            )
            home_expected = pl.when(context_available).then(
                ((pl.col("home_avg_points") + pl.col("away_avg_allowed")) / 2.0) + 1.5
            ).otherwise(home_fallback)
            away_expected = pl.when(context_available).then(
                (pl.col("away_avg_points") + pl.col("home_avg_allowed")) / 2.0
            ).otherwise(away_fallback)
        else:
            home_expected = home_fallback
            away_expected = away_fallback

        return upcoming_df.with_columns(
            [
                home_expected.alias("home_expected_score"),
                away_expected.alias("away_expected_score"),
                pl.lit("heuristic").alias("model_margin_source"),
                pl.lit(None).cast(pl.Utf8).alias("model_trained_at"),
                pl.lit(None).cast(pl.Float64).alias("model_margin_mae"),
                pl.lit(None).cast(pl.Float64).alias("model_total_mae"),
            ]
        )

    def generate_projections(self, upcoming_df: pl.DataFrame) -> pl.DataFrame:
        """Generates projections for upcoming games."""
        projections = None
        if self.use_trained_model:
            projections = self.model_manager.predict_games(upcoming_df)
        if projections is None or projections.is_empty():
            projections = self._fallback_projections(upcoming_df)

        projections = projections.with_columns(
            [
                (pl.col("home_expected_score") - pl.col("away_expected_score")).alias(
                    "expected_margin"
                )
            ]
        ).with_columns(
            [
                (1.0 / (1.0 + (-pl.col("expected_margin") / 8.0).exp())).alias(
                    "home_win_prob"
                )
            ]
        )

        return projections


AdvancedProjectionEngine = AdvancedModelingEngine


if __name__ == "__main__":
    engine = AdvancedModelingEngine()
    test_df = pl.DataFrame(
        [{"game_id": "G1", "home_team_id": "t1", "away_team_id": "t2"}]
    )
    print(engine.generate_projections(test_df))
