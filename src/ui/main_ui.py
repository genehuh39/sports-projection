import os
import socket

from nicegui import ui
import polars as pl

from src.data.nba_fetcher import NBAFetcher
from src.models.advanced_engine import AdvancedModelingEngine
from src.models.value_engine import ValueEngine


def _is_port_available(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.2)
        return sock.connect_ex(("127.0.0.1", port)) != 0


def resolve_port(preferred_port: int) -> int:
    if _is_port_available(preferred_port):
        return preferred_port

    for port in range(preferred_port + 1, preferred_port + 21):
        if _is_port_available(port):
            return port

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class SportsApp:
    def __init__(self):
        self.fetcher = NBAFetcher()
        self.proj_engine = AdvancedModelingEngine()
        self.value_engine = ValueEngine()
        self.games_df = pl.DataFrame()
        self.results_df = pl.DataFrame()
        self.data_source = "mock"
        self.container_column = None

        self.positive_ev_only = False
        self.min_edge_pct = 0.0
        self.selected_source = "All"
        self.selected_date = "All"

    def _load_games(self) -> tuple[pl.DataFrame, str]:
        try:
            real_games = self.fetcher.get_upcoming_games_with_context(days_ahead=7)
            if not real_games.is_empty():
                odds_source = real_games.get_column("market_source").drop_nulls().unique().to_list()
                source_label = "live nba_api"
                if odds_source:
                    source_label += f" + odds ({', '.join(str(x) for x in odds_source)})"
                return real_games, source_label
        except Exception:
            pass

        mock_games = pl.DataFrame(
            [
                {
                    "game_id": f"G-{i:03d}",
                    "home_team_id": f"team_{i % 4 + 1}",
                    "away_team_id": f"team_{(i + 1) % 4 + 1}",
                    "home_team_code": f"T{i % 4 + 1}",
                    "away_team_code": f"T{(i + 1) % 4 + 1}",
                    "home_team_name": f"Team {i % 4 + 1}",
                    "away_team_name": f"Team {(i + 1) % 4 + 1}",
                    "game_date": "mock",
                    "status": "mock",
                    "market_odds": (i * 20 if i % 2 == 0 else -110 + i * 5),
                    "away_market_odds": (-1 * (i * 20) if i % 2 == 0 else 100 + i * 5),
                    "market_spread": round(((i % 5) - 2) * 2.5, 1),
                    "market_total": 218.5 + i,
                    "market_provider": "mock",
                    "market_bookmaker": "mock",
                    "market_source": "mock / mock",
                }
                for i in range(1, 11)
            ]
        )
        return mock_games, "mock"

    def _build_results(self, upcoming_df: pl.DataFrame) -> pl.DataFrame:
        projections = self.proj_engine.generate_projections(upcoming_df)

        def compute_implied_prob(row):
            return self.value_engine.american_to_implied_prob(row["market_odds"])

        def compute_ev(row):
            return self.value_engine.calculate_expected_value(
                row["home_win_prob"], row["market_odds"]
            )

        def compute_edge(row):
            implied_prob = self.value_engine.american_to_implied_prob(row["market_odds"])
            return self.value_engine.calculate_edge(row["home_win_prob"], implied_prob)

        return (
            projections.with_columns(
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
                    pl.concat_str(
                        [pl.col("away_team_code"), pl.lit(" @ "), pl.col("home_team_code")]
                    ).alias("matchup"),
                ]
            )
            .with_columns(
                [
                    pl.col("expected_value")
                    .map_elements(
                        self.value_engine.get_bet_recommendation,
                        return_dtype=pl.String,
                    )
                    .alias("recommendation")
                ]
            )
            .sort("expected_value", descending=True)
        )

    def refresh_data(self):
        upcoming_df, self.data_source = self._load_games()
        self.results_df = self._build_results(upcoming_df)
        self.games_df = upcoming_df

        source_options = self._source_options()
        date_options = self._date_options()
        if self.selected_source not in source_options:
            self.selected_source = "All"
        if self.selected_date not in date_options:
            self.selected_date = "All"

    def _source_options(self) -> list[str]:
        if self.results_df.is_empty() or "market_source" not in self.results_df.columns:
            return ["All"]
        values = [str(v) for v in self.results_df.get_column("market_source").drop_nulls().unique().to_list()]
        return ["All", *sorted(values)]

    def _date_options(self) -> list[str]:
        if self.results_df.is_empty() or "game_date" not in self.results_df.columns:
            return ["All"]
        values = [str(v) for v in self.results_df.get_column("game_date").drop_nulls().unique().to_list()]
        return ["All", *sorted(values)]

    def get_filtered_results(self) -> pl.DataFrame:
        if self.results_df.is_empty():
            return self.results_df

        filtered = self.results_df

        if self.positive_ev_only:
            filtered = filtered.filter(pl.col("expected_value") > 0)

        if self.min_edge_pct > 0:
            filtered = filtered.filter(pl.col("edge") >= (self.min_edge_pct / 100.0))

        if self.selected_source != "All":
            filtered = filtered.filter(pl.col("market_source") == self.selected_source)

        if self.selected_date != "All":
            filtered = filtered.filter(pl.col("game_date") == self.selected_date)

        return filtered.sort("expected_value", descending=True)

    def _set_filter(self, attr: str, value) -> None:
        if attr == "min_edge_pct":
            value = float(value or 0)
        setattr(self, attr, value)
        self.render_ui()

    @staticmethod
    def _format_optional_number(value, digits: int = 1) -> str:
        if value is None:
            return "—"
        try:
            if isinstance(value, float) and value != value:
                return "—"
            return f"{value:.{digits}f}"
        except (TypeError, ValueError):
            return str(value)

    @staticmethod
    def _format_optional_int(value) -> str:
        if value is None:
            return "—"
        try:
            if isinstance(value, float) and value != value:
                return "—"
            value = int(value)
            return f"+{value}" if value > 0 else str(value)
        except (TypeError, ValueError):
            return str(value)

    @staticmethod
    def _format_optional_percent(value) -> str:
        if value is None:
            return "—"
        try:
            if isinstance(value, float) and value != value:
                return "—"
            return f"{value:.1%}"
        except (TypeError, ValueError):
            return str(value)

    def render_ui(self):
        filtered_df = self.get_filtered_results()

        if self.container_column is not None:
            self.container_column.clear()
        else:
            self.container_column = ui.column().classes("w-full items-center p-4")

        with self.container_column:
            ui.label("Sports Projection Dashboard").classes("text-4xl font-bold mb-2")
            ui.label("Real NBA schedule when available, mock fallback otherwise").classes(
                "text-gray-500 mb-2"
            )
            ui.label(f"Data source: {self.data_source}").classes("text-sm text-blue-600 mb-6")

            with ui.card().classes("w-full max-w-[1500px] p-4 mb-6"):
                ui.label("Filters").classes("text-lg font-semibold mb-3")
                with ui.row().classes("w-full items-end gap-4 flex-wrap"):
                    ui.switch(
                        "Positive EV only",
                        value=self.positive_ev_only,
                        on_change=lambda e: self._set_filter("positive_ev_only", e.value),
                    )
                    ui.number(
                        "Min edge %",
                        value=self.min_edge_pct,
                        min=0,
                        step=0.5,
                        format="%.1f",
                        on_change=lambda e: self._set_filter("min_edge_pct", e.value),
                    ).classes("w-32")
                    ui.select(
                        self._source_options(),
                        value=self.selected_source,
                        label="Odds source",
                        on_change=lambda e: self._set_filter("selected_source", e.value),
                    ).classes("w-48")
                    ui.select(
                        self._date_options(),
                        value=self.selected_date,
                        label="Game date",
                        on_change=lambda e: self._set_filter("selected_date", e.value),
                    ).classes("w-40")
                    ui.button("Reset filters", on_click=lambda: self._reset_filters()).props("outline")

            with ui.row().classes("w-full justify-center gap-8 mb-8 flex-wrap"):
                with ui.card().classes("p-4 w-40"):
                    ui.label("Loaded Games").classes("text-xs uppercase text-gray-400")
                    ui.label(f"{len(self.results_df)}").classes("text-2xl font-bold")
                with ui.card().classes("p-4 w-40"):
                    ui.label("Shown Games").classes("text-xs uppercase text-gray-400")
                    ui.label(f"{len(filtered_df)}").classes("text-2xl font-bold")
                with ui.card().classes("p-4 w-40"):
                    ui.label("Best EV").classes("text-xs uppercase text-gray-400")
                    best_ev = filtered_df["expected_value"].max() if not filtered_df.is_empty() else 0
                    ui.label(f"{best_ev:.1%}").classes("text-2xl font-bold text-green-500")
                with ui.card().classes("p-4 w-40"):
                    ui.label("Best Edge").classes("text-xs uppercase text-gray-400")
                    best_edge = filtered_df["edge"].max() if not filtered_df.is_empty() else 0
                    ui.label(f"{best_edge:.1%}").classes("text-2xl font-bold text-blue-500")

            ui.label("Upcoming Market Opportunities").classes("text-xl font-semibold mb-2")

            with ui.card().classes("w-full max-w-[2100px] overflow-x-auto"):
                with ui.element("div").classes(
                    "w-full min-w-[1750px] grid grid-cols-13 gap-3 bg-gray-100 p-3 font-bold border-b items-center"
                ):
                    ui.label("Matchup")
                    ui.label("Date")
                    ui.label("Provider")
                    ui.label("Bookmaker")
                    ui.label("Home Exp")
                    ui.label("Away Exp")
                    ui.label("Model Win %").classes("text-center")
                    ui.label("Implied %").classes("text-center")
                    ui.label("Edge").classes("text-center")
                    ui.label("Home ML")
                    ui.label("Spread")
                    ui.label("Total")
                    ui.label("EV %").classes("text-right")

                for row in filtered_df.iter_rows(named=True):
                    ev_color = "text-green-600" if row["expected_value"] > 0 else "text-red-600"
                    edge_color = "text-green-600" if row["edge"] > 0 else "text-red-600"
                    with ui.element("div").classes(
                        "w-full min-w-[1750px] grid grid-cols-13 gap-3 p-3 border-b hover:bg-gray-50 items-center"
                    ):
                        ui.label(row["matchup"]).classes("font-medium")
                        ui.label(str(row.get("game_date") or "—")).classes("text-sm text-gray-600")
                        ui.label(str(row.get("market_provider") or "—")).classes("text-sm text-gray-600")
                        ui.label(str(row.get("market_bookmaker") or "—")).classes("text-sm text-gray-600")
                        ui.label(f"{row['home_expected_score']:.1f}")
                        ui.label(f"{row['away_expected_score']:.1f}")
                        ui.label(f"{row['home_win_prob']:.1%}").classes("text-center")
                        ui.label(self._format_optional_percent(row.get("implied_probability"))).classes(
                            "text-center"
                        )
                        ui.label(self._format_optional_percent(row.get("edge"))).classes(
                            "text-center font-semibold " + edge_color
                        )
                        ui.label(self._format_optional_int(row.get("market_odds")))
                        ui.label(self._format_optional_number(row.get("market_spread")))
                        ui.label(self._format_optional_number(row.get("market_total")))
                        ui.label(f"{row['expected_value']:.1%}").classes(
                            "font-bold text-right " + ev_color
                        )

            if filtered_df.is_empty():
                ui.label("No games match the current filters.").classes("text-gray-500 mt-4")

    def _reset_filters(self) -> None:
        self.positive_ev_only = False
        self.min_edge_pct = 0.0
        self.selected_source = "All"
        self.selected_date = "All"
        self.render_ui()

    def run(self):
        ui.colors(primary="#2196f3")
        self.refresh_data()
        self.render_ui()

        preferred_port = int(os.getenv("PORT", "8080"))
        resolved_port = os.getenv("SPORTS_DASHBOARD_RESOLVED_PORT")
        if resolved_port is not None:
            port = int(resolved_port)
        else:
            port = resolve_port(preferred_port)
            os.environ["SPORTS_DASHBOARD_RESOLVED_PORT"] = str(port)
            if port != preferred_port:
                print(f"Port {preferred_port} is busy, using {port} instead.")

        ui.run(title="Sports Projection App", port=port)


if __name__ in {"__main__", "__mp_main__"}:
    SportsApp().run()
