import polars as pl

from src.data.nba_fetcher import NBAFetcher
from src.models.advanced_engine import AdvancedModelingEngine
from src.models.value_engine import ValueEngine


def build_mock_schedule() -> pl.DataFrame:
    return pl.DataFrame(
        [
            {"game_id": "GAME-001", "home_team_id": "team_1", "away_team_id": "team_2", "market_odds": 150, "home_team_code": "T1", "away_team_code": "T2"},
            {"game_id": "GAME-002", "home_team_id": "team_3", "away_team_id": "team_4", "market_odds": -110, "home_team_code": "T3", "away_team_code": "T4"},
            {"game_id": "GAME-003", "home_team_id": "team_2", "away_team_id": "team_1", "market_odds": 200, "home_team_code": "T2", "away_team_code": "T1"},
            {"game_id": "GAME-004", "home_team_id": "team_4", "away_team_id": "team_3", "market_odds": -120, "home_team_code": "T4", "away_team_code": "T3"},
        ]
    )


def run_full_pipeline(auto_train: bool = True, use_live_data: bool = True) -> pl.DataFrame:
    print("--- Starting Full Sports Projection Pipeline ---")

    proj_engine = AdvancedModelingEngine(auto_train=auto_train, use_trained_model=auto_train)
    value_engine = ValueEngine()
    fetcher = NBAFetcher()

    if use_live_data:
        try:
            upcoming_df = fetcher.get_upcoming_games_with_context(days_ahead=7)
        except Exception:
            upcoming_df = pl.DataFrame()
    else:
        upcoming_df = pl.DataFrame()

    data_source = "live nba_api"
    if upcoming_df.is_empty():
        upcoming_df = build_mock_schedule()
        data_source = "mock fallback"
    elif "market_source" in upcoming_df.columns:
        odds_sources = upcoming_df.get_column("market_source").drop_nulls().unique().to_list()
        if odds_sources:
            data_source += f" + odds ({', '.join(str(x) for x in odds_sources)})"

    print(f"Step 1: Loaded {len(upcoming_df)} upcoming games from {data_source}.\n")

    print("Step 2: Running Projection Engine...")
    projections = proj_engine.generate_projections(upcoming_df)
    print("Projections generated.\n")

    print("Step 3: Calculating Expected Value (EV)...")

    def compute_implied_prob(row):
        return value_engine.american_to_implied_prob(row["market_odds"])

    def compute_edge(row):
        implied_prob = value_engine.american_to_implied_prob(row["market_odds"])
        return value_engine.calculate_edge(row["home_win_prob"], implied_prob)

    def compute_ev(row):
        return value_engine.calculate_expected_value(row["home_win_prob"], row["market_odds"])

    results = projections.with_columns(
        [
            pl.struct(["market_odds"])
            .map_elements(compute_implied_prob, return_dtype=pl.Float64)
            .alias("implied_probability"),
            pl.struct(["home_win_prob", "market_odds"])
            .map_elements(compute_edge, return_dtype=pl.Float64)
            .alias("edge"),
            pl.struct(["home_win_prob", "market_odds"])
            .map_elements(compute_ev, return_dtype=pl.Float64)
            .alias("expected_value"),
            pl.when(
                pl.all_horizontal(
                    pl.col("away_team_code").is_not_null(),
                    pl.col("home_team_code").is_not_null(),
                )
            )
            .then(pl.concat_str([pl.col("away_team_code"), pl.lit(" @ "), pl.col("home_team_code")]))
            .otherwise(pl.col("game_id"))
            .alias("matchup"),
        ]
    )

    print("\n--- FINAL VALUE REPORT ---")
    report = results.select(
        [
            "matchup",
            "home_expected_score",
            "away_expected_score",
            "home_win_prob",
            "implied_probability",
            "edge",
            "market_odds",
            "expected_value",
        ]
    ).sort("expected_value", descending=True)

    print(report)

    print("\n--- RECOMMENDATIONS ---")
    for row in report.iter_rows(named=True):
        rec = value_engine.get_bet_recommendation(row["expected_value"])
        print(
            f"[{rec}] {row['matchup']}: Model {row['home_win_prob']:.2%}, "
            f"Implied {row['implied_probability']:.2%}, Edge {row['edge']:.2%}, "
            f"Odds {row['market_odds']}, EV {row['expected_value']:.2%}"
        )

    return report
