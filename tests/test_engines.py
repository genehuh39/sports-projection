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
        app = SportsApp()
        app.refresh_data()
        self.assertIn("market_source", app.results_df.columns)
        self.assertIn("market_provider", app.results_df.columns)
        self.assertIn("market_bookmaker", app.results_df.columns)
        self.assertIn("market_spread", app.results_df.columns)
        self.assertIn("market_total", app.results_df.columns)
        self.assertIn("implied_probability", app.results_df.columns)
        self.assertIn("edge", app.results_df.columns)

    def test_dashboard_filters_positive_ev_and_min_edge(self):
        app = SportsApp()
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
        report = run_full_pipeline()
        self.assertGreater(report.height, 0)
        evs = report["expected_value"].to_list()
        self.assertEqual(evs, sorted(evs, reverse=True))
        self.assertIn("implied_probability", report.columns)
        self.assertIn("edge", report.columns)


if __name__ == "__main__":
    unittest.main()
