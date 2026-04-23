import socket
import unittest

import polars as pl

from src.data.nba_fetcher import NBAFetcher
from src.data.odds_providers import normalize_team_code
from src.pipeline import run_full_pipeline
from src.models.advanced_engine import AdvancedModelingEngine
from src.models.value_engine import ValueEngine
from src.ui.main_ui import SportsApp, resolve_port


class ValueEngineTests(unittest.TestCase):
    def test_american_to_decimal_positive_and_negative(self):
        engine = ValueEngine()
        self.assertAlmostEqual(engine.american_to_decimal(150), 2.5)
        self.assertAlmostEqual(engine.american_to_decimal(-110), 1.9090909090909092)

    def test_american_to_implied_probability(self):
        engine = ValueEngine()
        self.assertAlmostEqual(engine.american_to_implied_prob(150), 0.4)
        self.assertAlmostEqual(engine.american_to_implied_prob(-110), 0.5238095238)

    def test_expected_value_positive_for_good_price(self):
        engine = ValueEngine()
        ev = engine.calculate_expected_value(0.60, 150)
        self.assertGreater(ev, 0)

    def test_edge_is_model_minus_implied(self):
        engine = ValueEngine()
        edge = engine.calculate_edge(0.60, 0.52)
        self.assertAlmostEqual(edge, 0.08)


class ProjectionEngineTests(unittest.TestCase):
    def test_generate_projections_adds_expected_fields(self):
        engine = AdvancedModelingEngine()
        games = pl.DataFrame(
            [
                {
                    "game_id": "G-001",
                    "home_team_id": "team_1",
                    "away_team_id": "team_2",
                    "market_odds": -110,
                }
            ]
        )

        results = engine.generate_projections(games)
        row = results.to_dicts()[0]

        self.assertIn("home_expected_score", row)
        self.assertIn("away_expected_score", row)
        self.assertIn("expected_margin", row)
        self.assertIn("home_win_prob", row)
        self.assertGreaterEqual(row["home_win_prob"], 0.0)
        self.assertLessEqual(row["home_win_prob"], 1.0)

    def test_generate_projections_uses_real_context_when_available(self):
        engine = AdvancedModelingEngine()
        games = pl.DataFrame(
            [
                {
                    "game_id": "G-001",
                    "home_team_id": 1610612747,
                    "away_team_id": 1610612738,
                    "market_odds": -110,
                    "home_avg_points": 118.0,
                    "away_avg_points": 112.0,
                    "home_avg_allowed": 108.0,
                    "away_avg_allowed": 110.0,
                }
            ]
        )

        row = engine.generate_projections(games).to_dicts()[0]
        self.assertAlmostEqual(row["home_expected_score"], 115.5)
        self.assertAlmostEqual(row["away_expected_score"], 110.0)
        self.assertAlmostEqual(row["expected_margin"], 5.5)


class UtilityTests(unittest.TestCase):
    def test_team_code_normalization_handles_espn_aliases(self):
        self.assertEqual(normalize_team_code("SA"), "SAS")
        self.assertEqual(normalize_team_code("GS"), "GSW")
        self.assertEqual(normalize_team_code("PHI"), "PHI")

    def test_resolve_port_skips_busy_port(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            sock.listen(1)
            busy_port = sock.getsockname()[1]
            resolved = resolve_port(busy_port)
            self.assertNotEqual(resolved, busy_port)

    def test_market_odds_prefers_higher_priority_provider(self):
        class ProviderA:
            def fetch(self, _):
                return pl.DataFrame([
                    {
                        "game_date": "2026-04-21",
                        "home_team_code": "SAS",
                        "away_team_code": "POR",
                        "market_odds": -700,
                        "away_market_odds": 500,
                        "market_spread": -11.5,
                        "market_total": 220.5,
                        "market_provider": "Feed A",
                        "market_bookmaker": "Book A",
                        "market_source": "Feed A / Book A",
                        "provider_priority": 20,
                    }
                ])

        class ProviderB:
            def fetch(self, _):
                return pl.DataFrame([
                    {
                        "game_date": "2026-04-21",
                        "home_team_code": "SAS",
                        "away_team_code": "POR",
                        "market_odds": -650,
                        "away_market_odds": 480,
                        "market_spread": -10.5,
                        "market_total": 221.5,
                        "market_provider": "Feed B",
                        "market_bookmaker": "Book B",
                        "market_source": "Feed B / Book B",
                        "provider_priority": 10,
                    }
                ])

        fetcher = NBAFetcher()
        fetcher.odds_providers = [ProviderA(), ProviderB()]
        odds = fetcher.get_market_odds("2026-04-21")
        row = odds.to_dicts()[0]
        self.assertEqual(row["market_source"], "Feed B / Book B")
        self.assertEqual(row["market_provider"], "Feed B")
        self.assertEqual(row["market_bookmaker"], "Book B")
        self.assertEqual(row["market_odds"], -650)

    def test_dashboard_data_contains_market_and_edge_columns(self):
        app = SportsApp(auto_train=False, use_live_data=False)
        app.refresh_data()
        self.assertIn("market_source", app.results_df.columns)
        self.assertIn("market_provider", app.results_df.columns)
        self.assertIn("market_bookmaker", app.results_df.columns)
        self.assertIn("market_spread", app.results_df.columns)
        self.assertIn("market_total", app.results_df.columns)
        self.assertIn("implied_probability", app.results_df.columns)
        self.assertIn("edge", app.results_df.columns)
        self.assertIn("model_margin_source", app.results_df.columns)
        self.assertIn("model_margin_mae", app.results_df.columns)
        self.assertIn("model_total_mae", app.results_df.columns)

    def test_dashboard_filters_positive_ev_and_min_edge(self):
        app = SportsApp(auto_train=False, use_live_data=False)
        app.results_df = pl.DataFrame(
            [
                {"matchup": "A @ B", "expected_value": 0.10, "edge": 0.08, "market_source": "X", "game_date": "2026-04-21"},
                {"matchup": "C @ D", "expected_value": -0.01, "edge": 0.02, "market_source": "X", "game_date": "2026-04-21"},
                {"matchup": "E @ F", "expected_value": 0.03, "edge": 0.01, "market_source": "Y", "game_date": "2026-04-22"},
            ]
        )
        app.positive_ev_only = True
        app.min_edge_pct = 5.0
        app.selected_source = "X"
        app.selected_date = "2026-04-21"

        filtered = app.get_filtered_results()
        self.assertEqual(filtered.height, 1)
        self.assertEqual(filtered["matchup"].to_list(), ["A @ B"])


class PipelineTests(unittest.TestCase):
    def test_full_pipeline_returns_sorted_report(self):
        report = run_full_pipeline(auto_train=False, use_live_data=False)
        self.assertGreater(report.height, 0)
        evs = report["expected_value"].to_list()
        self.assertEqual(evs, sorted(evs, reverse=True))
        self.assertIn("implied_probability", report.columns)
        self.assertIn("edge", report.columns)


class WalkForwardTests(unittest.TestCase):
    def _synthetic_raw_df(self):
        import numpy as np
        import pandas as pd
        from datetime import date, timedelta

        rng = np.random.default_rng(7)
        teams = [1610612737 + i for i in range(10)]
        rows = []
        start = date(2024, 11, 1)
        gid = 0
        for day in range(140):
            d = start + timedelta(days=day)
            order = teams.copy()
            rng.shuffle(order)
            for i in range(0, len(order) - 1, 2):
                h, a = order[i], order[i + 1]
                margin = float(rng.normal(0, 10))
                total_noise = float(rng.normal(0, 12))
                gid += 1
                home_pts = 110 + margin / 2 + total_noise / 2
                away_pts = 110 - margin / 2 + total_noise / 2
                rows.append(
                    {
                        "GAME_ID": f"00{gid:05d}",
                        "GAME_DATE": pd.Timestamp(d),
                        "TEAM_ID": h,
                        "TEAM_ABBREVIATION": f"T{h % 100}",
                        "TEAM_NAME": f"Team {h}",
                        "MATCHUP": f"T{h} vs. T{a}",
                        "WL": "W" if margin > 0 else "L",
                        "PTS": home_pts,
                        "FGA": 85,
                        "FTA": 20,
                        "OREB": 10,
                        "TOV": 14,
                        "PLUS_MINUS": margin,
                    }
                )
                rows.append(
                    {
                        "GAME_ID": f"00{gid:05d}",
                        "GAME_DATE": pd.Timestamp(d),
                        "TEAM_ID": a,
                        "TEAM_ABBREVIATION": f"T{a % 100}",
                        "TEAM_NAME": f"Team {a}",
                        "MATCHUP": f"T{a} @ T{h}",
                        "WL": "L" if margin > 0 else "W",
                        "PTS": away_pts,
                        "FGA": 85,
                        "FTA": 20,
                        "OREB": 10,
                        "TOV": 14,
                        "PLUS_MINUS": -margin,
                    }
                )
        return pd.DataFrame(rows)

    def test_walk_forward_returns_sensible_aggregates(self):
        from src.models.trained_nba_model import NBAModelManager

        manager = NBAModelManager()
        raw = self._synthetic_raw_df()
        manager.fetch_historical_team_games = lambda seasons=None: raw  # type: ignore[assignment]

        results = manager.walk_forward_evaluate(n_folds=3, test_size=40, min_train_size=80)
        self.assertEqual(len(results["folds"]), 3)
        self.assertGreater(results["margin_mae_mean"], 0)
        self.assertGreater(results["total_mae_mean"], 0)
        train_rows = [f["train_rows"] for f in results["folds"]]
        self.assertEqual(train_rows, sorted(train_rows))
        for f in results["folds"]:
            self.assertLessEqual(f["test_start_date"], f["test_end_date"])

    def test_walk_forward_reports_insufficient_data(self):
        from src.models.trained_nba_model import NBAModelManager
        import pandas as pd

        manager = NBAModelManager()
        manager.fetch_historical_team_games = lambda seasons=None: pd.DataFrame()  # type: ignore[assignment]
        results = manager.walk_forward_evaluate(n_folds=3, test_size=40)
        self.assertEqual(results["folds"], [])
        self.assertIn("error", results)


class HistoricalPointsAbsentTests(unittest.TestCase):
    def test_top_n_scorer_missing_counts_their_ppg(self):
        from src.models.trained_nba_model import NBAModelManager
        import pandas as pd

        # Team 1: 3 players. Star averages 30 MPG / 25 PPG; others ~20 MPG.
        # Game A: everyone plays. Game B: star plays 0 mins (injury).
        rows = []
        for gid in range(1, 11):  # 10 games of normal play — builds MPG average
            rows += [
                {"GAME_ID": f"g{gid}", "TEAM_ID": 1, "PLAYER_ID": 100, "GAME_DATE": f"2025-11-{gid:02d}", "MIN": 30, "PTS": 25},
                {"GAME_ID": f"g{gid}", "TEAM_ID": 1, "PLAYER_ID": 101, "GAME_DATE": f"2025-11-{gid:02d}", "MIN": 20, "PTS": 12},
                {"GAME_ID": f"g{gid}", "TEAM_ID": 1, "PLAYER_ID": 102, "GAME_DATE": f"2025-11-{gid:02d}", "MIN": 20, "PTS": 10},
            ]
        # Game B: star skips entirely
        rows += [
            {"GAME_ID": "gB", "TEAM_ID": 1, "PLAYER_ID": 101, "GAME_DATE": "2025-11-11", "MIN": 20, "PTS": 12},
            {"GAME_ID": "gB", "TEAM_ID": 1, "PLAYER_ID": 102, "GAME_DATE": "2025-11-11", "MIN": 20, "PTS": 10},
        ]
        player_log = pd.DataFrame(rows)
        player_log["GAME_DATE"] = pd.to_datetime(player_log["GAME_DATE"])

        absent = NBAModelManager()._historical_points_absent(player_log, top_n=3, min_fraction=0.5)
        by_game = dict(zip(absent["GAME_ID"], absent["POINTS_ABSENT"]))
        # In game g1 nobody is missing → 0
        self.assertAlmostEqual(by_game["g1"], 0.0)
        # In gB star (avg 25 PPG across his 10 games) is absent, others played → ~25
        self.assertAlmostEqual(by_game["gB"], 25.0, places=4)


class InjuryAdjustmentTests(unittest.TestCase):
    def test_adjustment_subtracts_damped_points_absent(self):
        from src.data.injury_provider import (
            IMPACT_DAMPING,
            InjuryReport,
            adjust_projections_for_injuries,
        )

        projections = pl.DataFrame(
            [
                {
                    "home_team_id": 100,
                    "away_team_id": 200,
                    "home_expected_score": 115.0,
                    "away_expected_score": 110.0,
                }
            ]
        )
        report = InjuryReport(
            points_absent_by_team_id={100: 25.0, 200: 5.0},
            source="test",
        )
        adjusted = adjust_projections_for_injuries(projections, report)
        row = adjusted.to_dicts()[0]
        self.assertAlmostEqual(row["home_points_absent"], 25.0)
        self.assertAlmostEqual(row["away_points_absent"], 5.0)
        self.assertAlmostEqual(row["home_expected_score"], 115.0 - 25.0 * IMPACT_DAMPING)
        self.assertAlmostEqual(row["away_expected_score"], 110.0 - 5.0 * IMPACT_DAMPING)

    def test_empty_report_leaves_scores_unchanged(self):
        from src.data.injury_provider import InjuryReport, adjust_projections_for_injuries

        projections = pl.DataFrame(
            [
                {
                    "home_team_id": 1,
                    "away_team_id": 2,
                    "home_expected_score": 110.0,
                    "away_expected_score": 108.0,
                }
            ]
        )
        adjusted = adjust_projections_for_injuries(projections, InjuryReport())
        row = adjusted.to_dicts()[0]
        self.assertAlmostEqual(row["home_expected_score"], 110.0)
        self.assertAlmostEqual(row["away_expected_score"], 108.0)
        self.assertAlmostEqual(row["home_points_absent"], 0.0)
        self.assertAlmostEqual(row["away_points_absent"], 0.0)

    def test_provider_builds_report_from_mocked_inputs(self):
        from src.data import injury_provider as ip

        provider = ip.InjuryProvider()
        provider._ppg_cache = {
            "star player": 28.0,
            "role player": 8.0,
            "long term guy": 22.0,
        }
        provider._recent_active_cache = {"star player", "role player"}
        provider._fetch_espn = lambda: {  # type: ignore[assignment]
            "Atlanta Hawks": [("Star Player", "Out"), ("Role Player", "Questionable")],
            "Nonexistent Team": [("Star Player", "Out")],
            "Boston Celtics": [("Unknown Name", "Out")],
            "Chicago Bulls": [("Long Term Guy", "Out")],
        }
        report = provider.fetch()
        hawks_id = next(
            t["id"] for t in __import__("nba_api.stats.static.teams", fromlist=["x"]).get_teams()
            if t["full_name"] == "Atlanta Hawks"
        )
        bulls_id = next(
            t["id"] for t in __import__("nba_api.stats.static.teams", fromlist=["x"]).get_teams()
            if t["full_name"] == "Chicago Bulls"
        )
        self.assertAlmostEqual(report.points_absent_by_team_id[hawks_id], 28.0 + 8.0 * 0.3)
        self.assertIn("Unknown Name", report.unmatched_players)
        self.assertIn("Long Term Guy", report.filtered_long_term)
        self.assertNotIn(bulls_id, report.points_absent_by_team_id)
        self.assertNotIn("Nonexistent Team", report.points_absent_by_team_id)


if __name__ == "__main__":
    unittest.main()
