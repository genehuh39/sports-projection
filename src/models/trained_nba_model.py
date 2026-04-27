from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import polars as pl
from nba_api.stats.endpoints import leaguegamefinder, leaguegamelog
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, mean_absolute_error
from xgboost import XGBRegressor


class PlattCalibrator:
    """Two-parameter sigmoid calibration on raw probabilities.

    Robust on small calibration sets where non-parametric isotonic
    regression overfits step-function jumps.
    """

    def __init__(self) -> None:
        self._lr: LogisticRegression | None = None

    def fit(self, raw_prob: np.ndarray, y: np.ndarray) -> "PlattCalibrator":
        self._lr = LogisticRegression()
        self._lr.fit(np.asarray(raw_prob).reshape(-1, 1), np.asarray(y))
        return self

    def predict(self, raw_prob: np.ndarray) -> np.ndarray:
        if self._lr is None:
            raise RuntimeError("PlattCalibrator must be fit before calling predict.")
        return self._lr.predict_proba(np.asarray(raw_prob).reshape(-1, 1))[:, 1]


@dataclass
class ModelArtifacts:
    feature_names: list[str]
    feature_defaults: dict[str, float]
    metrics: dict[str, float]
    seasons: list[str]
    trained_at: str
    margin_model: XGBRegressor
    total_model: XGBRegressor
    win_prob_calibrator: PlattCalibrator | None = None


class NBAModelManager:
    def __init__(
        self,
        artifact_path: str | Path = "artifacts/nba_projection_model.joblib",
        auto_train: bool = False,
        seasons: list[str] | None = None,
    ):
        self.artifact_path = Path(artifact_path)
        self.auto_train = auto_train
        self.seasons = seasons or self.default_seasons()
        self.artifacts: ModelArtifacts | None = None

    @staticmethod
    def current_season(today: date | None = None) -> str:
        today = today or datetime.now(timezone.utc).date()
        start_year = today.year if today.month >= 10 else today.year - 1
        end_year_short = str((start_year + 1) % 100).zfill(2)
        return f"{start_year}-{end_year_short}"

    @classmethod
    def previous_season(cls, season: str | None = None) -> str:
        season = season or cls.current_season()
        start_year = int(season.split("-")[0]) - 1
        end_year_short = str((start_year + 1) % 100).zfill(2)
        return f"{start_year}-{end_year_short}"

    @classmethod
    def default_seasons(cls) -> list[str]:
        current = cls.current_season()
        return [cls.previous_season(current), current]

    def load_artifacts(self) -> ModelArtifacts | None:
        if self.artifact_path.exists():
            payload = joblib.load(self.artifact_path)
            payload.setdefault("win_prob_calibrator", None)
            self.artifacts = ModelArtifacts(**payload)
        return self.artifacts

    def ensure_artifacts(self) -> ModelArtifacts | None:
        if self.artifacts is not None:
            return self.artifacts
        artifacts = self.load_artifacts()
        if artifacts is not None:
            return artifacts
        if self.auto_train:
            return self.train_and_save()
        return None

    def fetch_historical_team_games(self, seasons: list[str] | None = None) -> pd.DataFrame:
        seasons = seasons or self.seasons
        frames: list[pd.DataFrame] = []
        for season in seasons:
            frame = leaguegamefinder.LeagueGameFinder(
                season_nullable=season,
                season_type_nullable="Regular Season",
                league_id_nullable="00",
            ).get_data_frames()[0]
            if not frame.empty:
                frame = frame.copy()
                frame["SOURCE_SEASON"] = season
                frames.append(frame)

        if not frames:
            return pd.DataFrame()

        combined = pd.concat(frames, ignore_index=True)
        combined = combined.drop_duplicates(subset=["GAME_ID", "TEAM_ID"])
        return combined

    @staticmethod
    def _shifted_rolling_mean(series: pd.Series, window: int, min_periods: int = 5) -> pd.Series:
        return series.shift(1).rolling(window, min_periods=min_periods).mean()

    @staticmethod
    def _shifted_expanding_mean(series: pd.Series, min_periods: int = 5) -> pd.Series:
        return series.shift(1).expanding(min_periods=min_periods).mean()

    def build_team_history(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        if raw_df.empty:
            return pd.DataFrame()

        cols = [
            "GAME_ID",
            "GAME_DATE",
            "TEAM_ID",
            "TEAM_ABBREVIATION",
            "TEAM_NAME",
            "MATCHUP",
            "WL",
            "PTS",
            "FGA",
            "FTA",
            "OREB",
            "TOV",
            "PLUS_MINUS",
        ]
        df = raw_df[cols].copy()
        df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
        df["IS_HOME"] = df["MATCHUP"].astype(str).str.contains("vs.")
        df["WIN_FLAG"] = (df["WL"] == "W").astype(int)
        df["POSSESSIONS"] = df["FGA"] + (0.44 * df["FTA"]) - df["OREB"] + df["TOV"]

        opponents = df[["GAME_ID", "TEAM_ID", "PTS"]].rename(
            columns={"TEAM_ID": "OPP_TEAM_ID", "PTS": "OPP_PTS"}
        )
        df = df.merge(opponents, on="GAME_ID", how="inner")
        df = df[df["TEAM_ID"] != df["OPP_TEAM_ID"]].copy()
        df = df.sort_values(["TEAM_ID", "GAME_DATE", "GAME_ID"]).reset_index(drop=True)

        group = df.groupby("TEAM_ID", group_keys=False)
        df["GAMES_PLAYED_PRE"] = group.cumcount()
        prev_game_date = group["GAME_DATE"].shift(1)
        df["REST_DAYS_PRE"] = (df["GAME_DATE"] - prev_game_date).dt.days.fillna(3).clip(lower=0, upper=10)
        df["B2B_PRE"] = (df["REST_DAYS_PRE"] <= 1).astype(int)

        for source_col, prefix in [
            ("PTS", "OFFENSE"),
            ("OPP_PTS", "DEFENSE"),
            ("PLUS_MINUS", "MARGIN"),
            ("POSSESSIONS", "PACE"),
            ("WIN_FLAG", "WIN_RATE"),
        ]:
            df[f"RECENT_{prefix}"] = group[source_col].transform(
                lambda s: self._shifted_rolling_mean(s, window=10, min_periods=5)
            )
            df[f"SEASON_{prefix}"] = group[source_col].apply(
                lambda s: self._shifted_expanding_mean(s, min_periods=5)
            ).reset_index(level=0, drop=True)

        df = self._add_srs_pre(df)

        return df

    @staticmethod
    def _solve_srs(
        home_ids: np.ndarray,
        away_ids: np.ndarray,
        home_margins: np.ndarray,
        max_iter: int = 25,
        tol: float = 0.01,
    ) -> dict[int, float]:
        teams = np.unique(np.concatenate([home_ids, away_ids]))
        if teams.size == 0:
            return {}
        idx_of = {int(t): i for i, t in enumerate(teams)}
        n = len(teams)

        games_per_team: list[list[tuple[int, float]]] = [[] for _ in range(n)]
        for h, a, m in zip(home_ids, away_ids, home_margins):
            hi, ai = idx_of[int(h)], idx_of[int(a)]
            games_per_team[hi].append((ai, float(m)))
            games_per_team[ai].append((hi, -float(m)))

        ratings = np.zeros(n)
        for _ in range(max_iter):
            new_ratings = np.zeros(n)
            for i, games in enumerate(games_per_team):
                if not games:
                    continue
                opp_idx = [g[0] for g in games]
                margins = [g[1] for g in games]
                new_ratings[i] = float(np.mean(margins)) + float(np.mean(ratings[opp_idx]))
            new_ratings -= new_ratings.mean()
            if np.abs(new_ratings - ratings).max() < tol:
                ratings = new_ratings
                break
            ratings = new_ratings

        return {int(teams[i]): float(ratings[i]) for i in range(n)}

    @staticmethod
    def _season_start_year(dates: pd.Series) -> pd.Series:
        return dates.dt.year - (dates.dt.month < 10).astype(int)

    def _add_srs_pre(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["_SEASON_START"] = self._season_start_year(df["GAME_DATE"])
        df["SRS_PRE"] = 0.0

        home_pairs = df[df["IS_HOME"]][
            ["GAME_ID", "GAME_DATE", "_SEASON_START", "TEAM_ID", "OPP_TEAM_ID", "PLUS_MINUS"]
        ].rename(
            columns={
                "TEAM_ID": "HOME_ID",
                "OPP_TEAM_ID": "AWAY_ID",
                "PLUS_MINUS": "HOME_MARGIN",
            }
        )

        for season, season_pairs in home_pairs.groupby("_SEASON_START"):
            season_pairs = season_pairs.sort_values("GAME_DATE")
            mask = df["_SEASON_START"] == season
            season_dates = sorted(df.loc[mask, "GAME_DATE"].unique())
            for d in season_dates:
                prior = season_pairs[season_pairs["GAME_DATE"] < d]
                if len(prior) < 30:
                    continue
                ratings = self._solve_srs(
                    prior["HOME_ID"].to_numpy(),
                    prior["AWAY_ID"].to_numpy(),
                    prior["HOME_MARGIN"].to_numpy(),
                )
                if not ratings:
                    continue
                day_mask = mask & (df["GAME_DATE"] == d)
                df.loc[day_mask, "SRS_PRE"] = df.loc[day_mask, "TEAM_ID"].map(ratings).fillna(0.0)

        df = df.drop(columns=["_SEASON_START"])
        return df

    def build_training_frame(self, team_history_df: pd.DataFrame) -> pd.DataFrame:
        if team_history_df.empty:
            return pd.DataFrame()

        base_feature_cols = [
            "RECENT_OFFENSE",
            "RECENT_DEFENSE",
            "RECENT_MARGIN",
            "RECENT_PACE",
            "RECENT_WIN_RATE",
            "SEASON_OFFENSE",
            "SEASON_DEFENSE",
            "SEASON_MARGIN",
            "SEASON_PACE",
            "SEASON_WIN_RATE",
            "REST_DAYS_PRE",
            "B2B_PRE",
            "GAMES_PLAYED_PRE",
            "SRS_PRE",
        ]

        home_df = team_history_df[team_history_df["IS_HOME"]].copy()
        away_df = team_history_df[~team_history_df["IS_HOME"]].copy()

        home_keep = ["GAME_ID", "GAME_DATE", "TEAM_ID", "TEAM_ABBREVIATION", "PTS", *base_feature_cols]
        away_keep = ["GAME_ID", "GAME_DATE", "TEAM_ID", "TEAM_ABBREVIATION", "PTS", *base_feature_cols]

        home_df = home_df[home_keep].rename(
            columns={
                "GAME_DATE": "game_date",
                "TEAM_ID": "home_team_id",
                "TEAM_ABBREVIATION": "home_team_code",
                "PTS": "home_score",
                **{col: f"home_{col.lower()}" for col in base_feature_cols},
            }
        )
        away_df = away_df[away_keep].rename(
            columns={
                "GAME_DATE": "away_game_date",
                "TEAM_ID": "away_team_id",
                "TEAM_ABBREVIATION": "away_team_code",
                "PTS": "away_score",
                **{col: f"away_{col.lower()}" for col in base_feature_cols},
            }
        )

        games = home_df.merge(away_df, on="GAME_ID", how="inner")
        games = games.rename(columns={"GAME_ID": "game_id"})
        games["game_date"] = pd.to_datetime(games["game_date"])
        games["target_margin"] = games["home_score"] - games["away_score"]
        games["target_total"] = games["home_score"] + games["away_score"]

        games["diff_recent_offense"] = games["home_recent_offense"] - games["away_recent_offense"]
        games["diff_recent_defense"] = games["home_recent_defense"] - games["away_recent_defense"]
        games["diff_recent_margin"] = games["home_recent_margin"] - games["away_recent_margin"]
        games["diff_recent_pace"] = games["home_recent_pace"] - games["away_recent_pace"]
        games["diff_season_offense"] = games["home_season_offense"] - games["away_season_offense"]
        games["diff_season_defense"] = games["home_season_defense"] - games["away_season_defense"]
        games["diff_season_margin"] = games["home_season_margin"] - games["away_season_margin"]
        games["diff_rest_days"] = games["home_rest_days_pre"] - games["away_rest_days_pre"]
        games["diff_srs"] = games["home_srs_pre"] - games["away_srs_pre"]

        feature_cols = self.feature_names()
        games = games.dropna(subset=feature_cols)
        games = games[
            (games["home_games_played_pre"] >= 8) & (games["away_games_played_pre"] >= 8)
        ].copy()
        return games.sort_values("game_date").reset_index(drop=True)

    @staticmethod
    def feature_names() -> list[str]:
        return [
            "home_recent_offense",
            "home_recent_defense",
            "home_recent_margin",
            "home_recent_pace",
            "home_recent_win_rate",
            "home_season_offense",
            "home_season_defense",
            "home_season_margin",
            "home_season_pace",
            "home_season_win_rate",
            "home_rest_days_pre",
            "home_b2b_pre",
            "home_games_played_pre",
            "home_srs_pre",
            "away_recent_offense",
            "away_recent_defense",
            "away_recent_margin",
            "away_recent_pace",
            "away_recent_win_rate",
            "away_season_offense",
            "away_season_defense",
            "away_season_margin",
            "away_season_pace",
            "away_season_win_rate",
            "away_rest_days_pre",
            "away_b2b_pre",
            "away_games_played_pre",
            "away_srs_pre",
            "diff_recent_offense",
            "diff_recent_defense",
            "diff_recent_margin",
            "diff_recent_pace",
            "diff_season_offense",
            "diff_season_defense",
            "diff_season_margin",
            "diff_rest_days",
            "diff_srs",
        ]

    def build_team_snapshots(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        if raw_df.empty:
            return pd.DataFrame()

        team_history_df = self.build_team_history(raw_df)
        if team_history_df.empty:
            return pd.DataFrame()

        current_srs = self._current_srs_snapshot(team_history_df)

        records: list[dict[str, Any]] = []
        for team_id, team_games in team_history_df.groupby("TEAM_ID"):
            team_games = team_games.sort_values("GAME_DATE")
            last_10 = team_games.tail(10)
            records.append(
                {
                    "team_id": int(team_id),
                    "team_code": str(team_games["TEAM_ABBREVIATION"].iloc[-1]),
                    "team_name": str(team_games["TEAM_NAME"].iloc[-1]),
                    "recent_offense": float(last_10["PTS"].mean()),
                    "recent_defense": float(last_10["OPP_PTS"].mean()),
                    "recent_margin": float(last_10["PLUS_MINUS"].mean()),
                    "recent_pace": float(last_10["POSSESSIONS"].mean()),
                    "recent_win_rate": float(last_10["WIN_FLAG"].mean()),
                    "season_offense": float(team_games["PTS"].mean()),
                    "season_defense": float(team_games["OPP_PTS"].mean()),
                    "season_margin": float(team_games["PLUS_MINUS"].mean()),
                    "season_pace": float(team_games["POSSESSIONS"].mean()),
                    "season_win_rate": float(team_games["WIN_FLAG"].mean()),
                    "games_played": int(team_games["GAME_ID"].nunique()),
                    "last_game_date": team_games["GAME_DATE"].max().date(),
                    "srs_pre": float(current_srs.get(int(team_id), 0.0)),
                }
            )

        return pd.DataFrame(records)

    def _current_srs_snapshot(self, team_history_df: pd.DataFrame) -> dict[int, float]:
        if team_history_df.empty:
            return {}
        latest_season = int(self._season_start_year(team_history_df["GAME_DATE"]).max())
        season_mask = self._season_start_year(team_history_df["GAME_DATE"]) == latest_season
        home_rows = team_history_df[season_mask & team_history_df["IS_HOME"]]
        if len(home_rows) < 30:
            return {}
        return self._solve_srs(
            home_rows["TEAM_ID"].to_numpy(),
            home_rows["OPP_TEAM_ID"].to_numpy(),
            home_rows["PLUS_MINUS"].to_numpy(),
        )

    def build_upcoming_feature_frame(self, upcoming_df: pl.DataFrame, raw_df: pd.DataFrame | None = None) -> pd.DataFrame:
        raw_df = raw_df if raw_df is not None else self.fetch_historical_team_games()
        snapshots = self.build_team_snapshots(raw_df)
        if snapshots.empty:
            return pd.DataFrame(upcoming_df.to_dicts())

        upcoming = pd.DataFrame(upcoming_df.to_dicts())
        if upcoming.empty:
            return upcoming

        numeric_home_ids = pd.to_numeric(upcoming.get("home_team_id"), errors="coerce")
        numeric_away_ids = pd.to_numeric(upcoming.get("away_team_id"), errors="coerce")
        if numeric_home_ids.isna().any() or numeric_away_ids.isna().any():
            return upcoming
        upcoming["home_team_id"] = numeric_home_ids.astype(int)
        upcoming["away_team_id"] = numeric_away_ids.astype(int)

        upcoming["game_date"] = pd.to_datetime(upcoming.get("game_date"), errors="coerce")
        if upcoming["game_date"].isna().all():
            upcoming["game_date"] = pd.Timestamp(datetime.now(timezone.utc).date())
        else:
            upcoming["game_date"] = upcoming["game_date"].fillna(pd.Timestamp(datetime.now(timezone.utc).date()))

        home_snap = snapshots.rename(
            columns={
                "team_id": "home_team_id",
                "team_code": "home_snapshot_code",
                "team_name": "home_snapshot_name",
                "recent_offense": "home_recent_offense",
                "recent_defense": "home_recent_defense",
                "recent_margin": "home_recent_margin",
                "recent_pace": "home_recent_pace",
                "recent_win_rate": "home_recent_win_rate",
                "season_offense": "home_season_offense",
                "season_defense": "home_season_defense",
                "season_margin": "home_season_margin",
                "season_pace": "home_season_pace",
                "season_win_rate": "home_season_win_rate",
                "games_played": "home_games_played_pre",
                "last_game_date": "home_last_game_date",
                "srs_pre": "home_srs_pre",
            }
        )
        away_snap = snapshots.rename(
            columns={
                "team_id": "away_team_id",
                "team_code": "away_snapshot_code",
                "team_name": "away_snapshot_name",
                "recent_offense": "away_recent_offense",
                "recent_defense": "away_recent_defense",
                "recent_margin": "away_recent_margin",
                "recent_pace": "away_recent_pace",
                "recent_win_rate": "away_recent_win_rate",
                "season_offense": "away_season_offense",
                "season_defense": "away_season_defense",
                "season_margin": "away_season_margin",
                "season_pace": "away_season_pace",
                "season_win_rate": "away_season_win_rate",
                "games_played": "away_games_played_pre",
                "last_game_date": "away_last_game_date",
                "srs_pre": "away_srs_pre",
            }
        )

        upcoming = upcoming.merge(home_snap, on="home_team_id", how="left")
        upcoming = upcoming.merge(away_snap, on="away_team_id", how="left")

        home_last_game = pd.to_datetime(upcoming["home_last_game_date"], errors="coerce")
        away_last_game = pd.to_datetime(upcoming["away_last_game_date"], errors="coerce")
        upcoming["home_rest_days_pre"] = (
            upcoming["game_date"] - home_last_game
        ).dt.days.fillna(3).clip(lower=0, upper=10)
        upcoming["away_rest_days_pre"] = (
            upcoming["game_date"] - away_last_game
        ).dt.days.fillna(3).clip(lower=0, upper=10)
        upcoming["home_b2b_pre"] = (upcoming["home_rest_days_pre"] <= 1).astype(int)
        upcoming["away_b2b_pre"] = (upcoming["away_rest_days_pre"] <= 1).astype(int)

        upcoming["diff_recent_offense"] = upcoming["home_recent_offense"] - upcoming["away_recent_offense"]
        upcoming["diff_recent_defense"] = upcoming["home_recent_defense"] - upcoming["away_recent_defense"]
        upcoming["diff_recent_margin"] = upcoming["home_recent_margin"] - upcoming["away_recent_margin"]
        upcoming["diff_recent_pace"] = upcoming["home_recent_pace"] - upcoming["away_recent_pace"]
        upcoming["diff_season_offense"] = upcoming["home_season_offense"] - upcoming["away_season_offense"]
        upcoming["diff_season_defense"] = upcoming["home_season_defense"] - upcoming["away_season_defense"]
        upcoming["diff_season_margin"] = upcoming["home_season_margin"] - upcoming["away_season_margin"]
        upcoming["diff_rest_days"] = upcoming["home_rest_days_pre"] - upcoming["away_rest_days_pre"]
        upcoming["diff_srs"] = upcoming["home_srs_pre"] - upcoming["away_srs_pre"]

        return upcoming

    @staticmethod
    def _new_xgb_regressor() -> XGBRegressor:
        return XGBRegressor(
            objective="reg:squarederror",
            n_estimators=250,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
        )

    def train_and_save(self, force: bool = False) -> ModelArtifacts | None:
        if self.artifact_path.exists() and not force:
            return self.load_artifacts()

        raw_df = self.fetch_historical_team_games(self.seasons)
        team_history_df = self.build_team_history(raw_df)
        training_df = self.build_training_frame(team_history_df)
        if training_df.empty or len(training_df) < 200:
            return None

        feature_names = self.feature_names()
        feature_defaults = {
            name: float(training_df[name].median()) for name in feature_names
        }

        training_df = training_df.sort_values("game_date").reset_index(drop=True)
        split_idx = max(int(len(training_df) * 0.8), 1)
        train_df = training_df.iloc[:split_idx]
        test_df = training_df.iloc[split_idx:]
        if test_df.empty:
            test_df = training_df.iloc[-min(50, len(training_df)):]
            train_df = training_df.iloc[:-len(test_df)] if len(training_df) > len(test_df) else training_df

        X_train = train_df[feature_names].fillna(feature_defaults)
        X_test = test_df[feature_names].fillna(feature_defaults)

        margin_model = self._new_xgb_regressor()
        total_model = self._new_xgb_regressor()

        margin_model.fit(X_train, train_df["target_margin"])
        total_model.fit(X_train, train_df["target_total"])

        margin_pred = margin_model.predict(X_test)
        total_pred = total_model.predict(X_test)
        metrics = {
            "margin_mae": float(mean_absolute_error(test_df["target_margin"], margin_pred)),
            "total_mae": float(mean_absolute_error(test_df["target_total"], total_pred)),
            "train_rows": float(len(train_df)),
            "test_rows": float(len(test_df)),
        }

        # Win-probability calibration was tested via sports-backtest and did not
        # improve any metric (raw Brier 0.2156 vs calibrated 0.22+ at every
        # holdout size). Calibrator slot is kept as None so the artifact schema
        # can support it later if cross-validated calibration becomes worth
        # building. predict_games falls back to the default logistic mapping.
        self.artifact_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "feature_names": feature_names,
            "feature_defaults": feature_defaults,
            "metrics": metrics,
            "seasons": self.seasons,
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "margin_model": margin_model,
            "total_model": total_model,
            "win_prob_calibrator": None,
        }
        joblib.dump(payload, self.artifact_path)
        self.artifacts = ModelArtifacts(**payload)
        return self.artifacts

    @staticmethod
    def _fit_win_prob_calibrator(
        raw_prob: np.ndarray, actual_home_win: np.ndarray
    ) -> PlattCalibrator:
        return PlattCalibrator().fit(raw_prob, actual_home_win)

    def walk_forward_evaluate(
        self,
        n_folds: int = 6,
        test_size: int = 150,
        min_train_size: int = 400,
    ) -> dict[str, Any]:
        """Run time-series walk-forward CV over the last ``n_folds * test_size`` rows.

        For each fold k, trains on all games earlier than the fold's test window
        and measures MAE on that window. Returns per-fold metrics plus aggregate
        mean and stdev so single-split noise does not mask true signal.
        """
        raw_df = self.fetch_historical_team_games(self.seasons)
        team_history_df = self.build_team_history(raw_df)
        training_df = self.build_training_frame(team_history_df)
        if training_df.empty:
            return {"folds": [], "error": "no training data"}

        training_df = training_df.sort_values("game_date").reset_index(drop=True)
        total = len(training_df)
        needed = min_train_size + n_folds * test_size
        if total < needed:
            return {
                "folds": [],
                "error": f"need {needed} rows, have {total}",
            }

        feature_names = self.feature_names()
        feature_defaults = {
            name: float(training_df[name].median()) for name in feature_names
        }

        fold_results: list[dict[str, Any]] = []
        first_test_start = total - n_folds * test_size
        for k in range(n_folds):
            test_start = first_test_start + k * test_size
            test_end = test_start + test_size
            train_df = training_df.iloc[:test_start]
            test_df = training_df.iloc[test_start:test_end]
            if len(train_df) < min_train_size or test_df.empty:
                continue

            X_train = train_df[feature_names].fillna(feature_defaults)
            X_test = test_df[feature_names].fillna(feature_defaults)

            margin_model = self._new_xgb_regressor()
            total_model = self._new_xgb_regressor()
            margin_model.fit(X_train, train_df["target_margin"])
            total_model.fit(X_train, train_df["target_total"])

            margin_mae = float(
                mean_absolute_error(test_df["target_margin"], margin_model.predict(X_test))
            )
            total_mae = float(
                mean_absolute_error(test_df["target_total"], total_model.predict(X_test))
            )
            fold_results.append(
                {
                    "fold": k,
                    "train_rows": int(len(train_df)),
                    "test_rows": int(len(test_df)),
                    "test_start_date": pd.Timestamp(test_df["game_date"].iloc[0]).date().isoformat(),
                    "test_end_date": pd.Timestamp(test_df["game_date"].iloc[-1]).date().isoformat(),
                    "margin_mae": margin_mae,
                    "total_mae": total_mae,
                }
            )

        if not fold_results:
            return {"folds": [], "error": "no folds produced"}

        margin_maes = [f["margin_mae"] for f in fold_results]
        total_maes = [f["total_mae"] for f in fold_results]
        return {
            "folds": fold_results,
            "n_folds": len(fold_results),
            "margin_mae_mean": float(np.mean(margin_maes)),
            "margin_mae_std": float(np.std(margin_maes, ddof=1)) if len(margin_maes) > 1 else 0.0,
            "total_mae_mean": float(np.mean(total_maes)),
            "total_mae_std": float(np.std(total_maes, ddof=1)) if len(total_maes) > 1 else 0.0,
            "seasons": self.seasons,
            "feature_count": len(feature_names),
        }

    def _fetch_player_game_log(self, seasons: list[str] | None = None) -> pd.DataFrame:
        seasons = seasons or self.seasons
        frames: list[pd.DataFrame] = []
        for season in seasons:
            frame = leaguegamelog.LeagueGameLog(
                season=season,
                player_or_team_abbreviation="P",
                season_type_all_star="Regular Season",
            ).get_data_frames()[0]
            if not frame.empty:
                frames.append(frame)
        if not frames:
            return pd.DataFrame()
        combined = pd.concat(frames, ignore_index=True)
        combined["GAME_DATE"] = pd.to_datetime(combined["GAME_DATE"])
        combined["MIN"] = pd.to_numeric(combined["MIN"], errors="coerce").fillna(0.0)
        combined["PTS"] = pd.to_numeric(combined["PTS"], errors="coerce").fillna(0.0)
        return combined

    def _historical_points_absent(
        self,
        player_log_df: pd.DataFrame,
        top_n: int = 8,
        min_fraction: float = 0.5,
    ) -> pd.DataFrame:
        """Estimate per-team, per-game "points absent" using box scores.

        For each team, the top-N players by season MPG are considered the
        rotation core. For each game, any core player whose minutes were
        below ``min_fraction * their season MPG`` has their season PPG
        counted as absent. Returns a DataFrame keyed by (GAME_ID, TEAM_ID)
        with a ``POINTS_ABSENT`` column.

        NB: "season MPG" here is the full-season average. This is a known
        leakage source for calibration but not for production, since we
        are just tuning a scalar — not training a model. The shape of the
        absence signal is what matters.
        """
        if player_log_df.empty:
            return pd.DataFrame(columns=["GAME_ID", "TEAM_ID", "POINTS_ABSENT"])

        player_season = (
            player_log_df.groupby(["PLAYER_ID", "TEAM_ID"])
            .agg(GP=("GAME_ID", "nunique"), MIN_SUM=("MIN", "sum"), PTS_SUM=("PTS", "sum"))
            .reset_index()
        )
        player_season = player_season[player_season["GP"] > 0].copy()
        player_season["MPG"] = player_season["MIN_SUM"] / player_season["GP"]
        player_season["PPG"] = player_season["PTS_SUM"] / player_season["GP"]
        player_season = player_season.sort_values(
            ["TEAM_ID", "MPG"], ascending=[True, False]
        )
        core = (
            player_season.groupby("TEAM_ID", group_keys=False)
            .head(top_n)
            .reset_index(drop=True)
        )

        team_games = player_log_df[["GAME_ID", "TEAM_ID", "GAME_DATE"]].drop_duplicates()
        cross = team_games.merge(core[["TEAM_ID", "PLAYER_ID", "MPG", "PPG"]], on="TEAM_ID")
        cross = cross.merge(
            player_log_df[["GAME_ID", "PLAYER_ID", "MIN"]],
            on=["GAME_ID", "PLAYER_ID"],
            how="left",
        )
        cross["MIN"] = cross["MIN"].fillna(0.0)
        cross["ABSENT"] = cross["MIN"] < (min_fraction * cross["MPG"])
        cross["POINTS_ABSENT"] = cross["PPG"] * cross["ABSENT"].astype(float)
        return (
            cross.groupby(["GAME_ID", "TEAM_ID"])["POINTS_ABSENT"].sum().reset_index()
        )

    def calibrate_injury_damping(
        self,
        damping_grid: list[float] | None = None,
        n_folds: int = 6,
        test_size: int = 150,
        min_train_size: int = 400,
        top_n: int = 8,
        min_fraction: float = 0.5,
    ) -> dict[str, Any]:
        """Walk-forward CV that sweeps injury damping factors post-hoc.

        For each fold, trains once on the fold's training window, makes
        raw predictions on the test window, then evaluates MAE under each
        damping value by subtracting damping * points_absent from the
        raw scores. The damping that minimizes mean margin MAE across
        folds is the empirical optimum for this harness.
        """
        if damping_grid is None:
            damping_grid = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5]

        raw_df = self.fetch_historical_team_games(self.seasons)
        team_history_df = self.build_team_history(raw_df)
        training_df = self.build_training_frame(team_history_df)
        if training_df.empty:
            return {"folds": [], "error": "no training data"}

        player_log = self._fetch_player_game_log(self.seasons)
        absent_df = self._historical_points_absent(
            player_log, top_n=top_n, min_fraction=min_fraction
        )
        absent_lookup = {
            (str(row["GAME_ID"]), int(row["TEAM_ID"])): float(row["POINTS_ABSENT"])
            for _, row in absent_df.iterrows()
        }

        training_df = training_df.sort_values("game_date").reset_index(drop=True)
        training_df = training_df.copy()
        training_df["home_points_absent"] = [
            absent_lookup.get((str(gid), int(tid)), 0.0)
            for gid, tid in zip(training_df["game_id"], training_df["home_team_id"])
        ]
        training_df["away_points_absent"] = [
            absent_lookup.get((str(gid), int(tid)), 0.0)
            for gid, tid in zip(training_df["game_id"], training_df["away_team_id"])
        ]

        total = len(training_df)
        needed = min_train_size + n_folds * test_size
        if total < needed:
            return {"folds": [], "error": f"need {needed} rows, have {total}"}

        feature_names = self.feature_names()
        feature_defaults = {
            name: float(training_df[name].median()) for name in feature_names
        }

        grid_results: dict[float, dict[str, list[float]]] = {
            d: {"margin_mae": [], "total_mae": []} for d in damping_grid
        }
        first_test_start = total - n_folds * test_size
        for k in range(n_folds):
            test_start = first_test_start + k * test_size
            test_end = test_start + test_size
            train_df = training_df.iloc[:test_start]
            test_df = training_df.iloc[test_start:test_end]
            if len(train_df) < min_train_size or test_df.empty:
                continue

            X_train = train_df[feature_names].fillna(feature_defaults)
            X_test = test_df[feature_names].fillna(feature_defaults)

            margin_model = self._new_xgb_regressor()
            total_model = self._new_xgb_regressor()
            margin_model.fit(X_train, train_df["target_margin"])
            total_model.fit(X_train, train_df["target_total"])

            raw_margin = margin_model.predict(X_test)
            raw_total = total_model.predict(X_test)
            home_abs = test_df["home_points_absent"].to_numpy()
            away_abs = test_df["away_points_absent"].to_numpy()
            y_margin = test_df["target_margin"].to_numpy()
            y_total = test_df["target_total"].to_numpy()

            for damping in damping_grid:
                adj_margin = raw_margin - damping * home_abs + damping * away_abs
                adj_total = raw_total - damping * home_abs - damping * away_abs
                grid_results[damping]["margin_mae"].append(
                    float(mean_absolute_error(y_margin, adj_margin))
                )
                grid_results[damping]["total_mae"].append(
                    float(mean_absolute_error(y_total, adj_total))
                )

        summary: list[dict[str, Any]] = []
        for damping in damping_grid:
            m = grid_results[damping]["margin_mae"]
            t = grid_results[damping]["total_mae"]
            if not m:
                continue
            summary.append(
                {
                    "damping": damping,
                    "margin_mae_mean": float(np.mean(m)),
                    "margin_mae_std": float(np.std(m, ddof=1)) if len(m) > 1 else 0.0,
                    "total_mae_mean": float(np.mean(t)),
                    "total_mae_std": float(np.std(t, ddof=1)) if len(t) > 1 else 0.0,
                }
            )

        if not summary:
            return {"folds": [], "error": "no folds produced"}

        best_margin = min(summary, key=lambda s: s["margin_mae_mean"])
        best_total = min(summary, key=lambda s: s["total_mae_mean"])
        return {
            "grid": summary,
            "n_folds": len(grid_results[damping_grid[0]]["margin_mae"]),
            "seasons": self.seasons,
            "top_n_players": top_n,
            "min_fraction": min_fraction,
            "best_margin_damping": best_margin["damping"],
            "best_total_damping": best_total["damping"],
            "baseline_margin_mae": next(
                s["margin_mae_mean"] for s in summary if s["damping"] == 0.0
            ),
            "baseline_total_mae": next(
                s["total_mae_mean"] for s in summary if s["damping"] == 0.0
            ),
        }

    @staticmethod
    def _margin_to_home_win_prob(margin: np.ndarray, scale: float = 8.0) -> np.ndarray:
        """Match the engine's logistic mapping in advanced_engine.py."""
        return 1.0 / (1.0 + np.exp(-margin / scale))

    def backtest_evaluate(
        self,
        n_folds: int = 6,
        test_size: int = 150,
        min_train_size: int = 400,
        vig: float = 0.0476,
        edge_thresholds: list[float] | None = None,
        n_calibration_buckets: int = 10,
        apply_calibration: bool = True,
        calibration_holdout: int = 100,
    ) -> dict[str, Any]:
        """Walk-forward backtest against a synthetic uniform -110 market.

        Computes calibration (Brier, log loss, accuracy, decile-bucket
        calibration) plus ROI for an "edge > threshold" betting strategy
        assuming a flat ``vig`` line on every game. The synthetic market
        means ROI is a benchmark, not a real-world PnL number — once
        historical odds are wired in, swap the synthetic prices for
        actual ones at the same call site.

        Default vig of 0.0476 corresponds to standard -110 / -110 lines
        (overround = 2 * 110/210 - 1). Breakeven prob = (1 + vig) / 2 =
        0.5238. Win payout per $1 = (1 - vig) / (1 + vig) ≈ 0.9091.
        """
        if edge_thresholds is None:
            edge_thresholds = [0.0, 0.02, 0.05, 0.10]
        # Dedupe while preserving order so duplicate thresholds don't double-count
        edge_thresholds = list(dict.fromkeys(edge_thresholds))

        raw_df = self.fetch_historical_team_games(self.seasons)
        team_history_df = self.build_team_history(raw_df)
        training_df = self.build_training_frame(team_history_df)
        if training_df.empty:
            return {"folds": [], "error": "no training data"}

        training_df = training_df.sort_values("game_date").reset_index(drop=True)
        total = len(training_df)
        needed = min_train_size + n_folds * test_size
        if total < needed:
            return {"folds": [], "error": f"need {needed} rows, have {total}"}

        feature_names = self.feature_names()
        feature_defaults = {
            name: float(training_df[name].median()) for name in feature_names
        }

        breakeven = (1.0 + vig) / 2.0
        win_payout = (1.0 - vig) / (1.0 + vig)

        per_fold: list[dict[str, Any]] = []
        all_probs: list[float] = []
        all_outcomes: list[int] = []
        threshold_pnl: dict[float, list[float]] = {t: [] for t in edge_thresholds}

        first_test_start = total - n_folds * test_size
        for k in range(n_folds):
            test_start = first_test_start + k * test_size
            test_end = test_start + test_size
            train_df = training_df.iloc[:test_start]
            test_df = training_df.iloc[test_start:test_end]
            if len(train_df) < min_train_size or test_df.empty:
                continue

            X_test = test_df[feature_names].fillna(feature_defaults)

            margin_model = self._new_xgb_regressor()
            calibrator: PlattCalibrator | None = None
            if (
                apply_calibration
                and len(train_df) >= min_train_size + calibration_holdout
            ):
                inner_train = train_df.iloc[:-calibration_holdout]
                calib_df = train_df.iloc[-calibration_holdout:]
                X_inner = inner_train[feature_names].fillna(feature_defaults)
                X_calib = calib_df[feature_names].fillna(feature_defaults)
                margin_model.fit(X_inner, inner_train["target_margin"])
                calib_raw_prob = self._margin_to_home_win_prob(
                    margin_model.predict(X_calib)
                )
                calib_actual = (calib_df["target_margin"].to_numpy() > 0).astype(int)
                calibrator = self._fit_win_prob_calibrator(calib_raw_prob, calib_actual)
            else:
                X_train = train_df[feature_names].fillna(feature_defaults)
                margin_model.fit(X_train, train_df["target_margin"])

            raw_prob = self._margin_to_home_win_prob(margin_model.predict(X_test))
            if calibrator is not None:
                predicted_home_prob = calibrator.predict(raw_prob)
            else:
                predicted_home_prob = raw_prob
            predicted_home_prob = np.clip(predicted_home_prob, 1e-6, 1 - 1e-6)
            actual_home_win = (test_df["target_margin"].to_numpy() > 0).astype(int)

            for prob, outcome in zip(predicted_home_prob, actual_home_win):
                all_probs.append(float(prob))
                all_outcomes.append(int(outcome))

            picks = (predicted_home_prob >= 0.5).astype(int)
            accuracy = float((picks == actual_home_win).mean())
            brier = float(brier_score_loss(actual_home_win, predicted_home_prob))
            ll = float(log_loss(actual_home_win, predicted_home_prob, labels=[0, 1]))

            fold_thresholds: list[dict[str, Any]] = []
            for threshold in edge_thresholds:
                bet_home = predicted_home_prob >= breakeven + threshold
                bet_away = (1.0 - predicted_home_prob) >= breakeven + threshold
                pnls: list[float] = []
                for bh, ba, win in zip(bet_home, bet_away, actual_home_win):
                    if bh:
                        pnls.append(win_payout if win == 1 else -1.0)
                    elif ba:
                        pnls.append(win_payout if win == 0 else -1.0)
                threshold_pnl[threshold].extend(pnls)
                fold_thresholds.append(
                    {
                        "edge_threshold": threshold,
                        "n_bets": int(len(pnls)),
                        "roi": float(np.mean(pnls)) if pnls else 0.0,
                    }
                )
            per_fold.append(
                {
                    "fold": k,
                    "n_games": int(len(test_df)),
                    "accuracy": accuracy,
                    "brier": brier,
                    "log_loss": ll,
                    "thresholds": fold_thresholds,
                }
            )

        if not per_fold:
            return {"folds": [], "error": "no folds produced"}

        probs_arr = np.asarray(all_probs)
        outcomes_arr = np.asarray(all_outcomes)
        bucket_edges = np.linspace(0.0, 1.0, n_calibration_buckets + 1)
        calibration: list[dict[str, float]] = []
        for i in range(n_calibration_buckets):
            lo, hi = bucket_edges[i], bucket_edges[i + 1]
            mask = (probs_arr >= lo) & (probs_arr < hi if i < n_calibration_buckets - 1 else probs_arr <= hi)
            if not mask.any():
                continue
            calibration.append(
                {
                    "bucket_lo": float(lo),
                    "bucket_hi": float(hi),
                    "n": int(mask.sum()),
                    "mean_predicted": float(probs_arr[mask].mean()),
                    "actual_rate": float(outcomes_arr[mask].mean()),
                }
            )

        threshold_summary: list[dict[str, Any]] = []
        for threshold in edge_thresholds:
            pnls = threshold_pnl[threshold]
            threshold_summary.append(
                {
                    "edge_threshold": threshold,
                    "n_bets": int(len(pnls)),
                    "roi_mean": float(np.mean(pnls)) if pnls else 0.0,
                    "roi_std": float(np.std(pnls, ddof=1)) if len(pnls) > 1 else 0.0,
                    "win_rate": float(
                        sum(1 for p in pnls if p > 0) / len(pnls)
                    ) if pnls else 0.0,
                }
            )

        return {
            "folds": per_fold,
            "n_folds": len(per_fold),
            "seasons": self.seasons,
            "vig": vig,
            "breakeven_prob": breakeven,
            "win_payout": win_payout,
            "applied_calibration": apply_calibration,
            "accuracy_mean": float(np.mean([f["accuracy"] for f in per_fold])),
            "accuracy_std": float(np.std([f["accuracy"] for f in per_fold], ddof=1))
            if len(per_fold) > 1 else 0.0,
            "brier_mean": float(np.mean([f["brier"] for f in per_fold])),
            "log_loss_mean": float(np.mean([f["log_loss"] for f in per_fold])),
            "calibration": calibration,
            "thresholds": threshold_summary,
        }

    def predict_games(self, upcoming_df: pl.DataFrame) -> pl.DataFrame | None:
        artifacts = self.ensure_artifacts()
        if artifacts is None or upcoming_df.is_empty():
            return None

        raw_df = self.fetch_historical_team_games(self.seasons)
        feature_df = self.build_upcoming_feature_frame(upcoming_df, raw_df=raw_df)
        if feature_df.empty:
            return None

        if any(feature not in feature_df.columns for feature in artifacts.feature_names):
            return None

        X = feature_df[artifacts.feature_names].fillna(artifacts.feature_defaults)
        margin_pred = artifacts.margin_model.predict(X)
        total_pred = artifacts.total_model.predict(X)

        home_score = np.clip((total_pred + margin_pred) / 2.0, 80, 145)
        away_score = np.clip((total_pred - margin_pred) / 2.0, 80, 145)

        feature_df["home_expected_score"] = home_score
        feature_df["away_expected_score"] = away_score
        feature_df["expected_margin"] = feature_df["home_expected_score"] - feature_df["away_expected_score"]
        feature_df["model_margin_source"] = "xgboost"
        feature_df["model_trained_at"] = artifacts.trained_at
        feature_df["model_margin_mae"] = artifacts.metrics.get("margin_mae")
        feature_df["model_total_mae"] = artifacts.metrics.get("total_mae")

        if artifacts.win_prob_calibrator is not None:
            raw_prob = self._margin_to_home_win_prob(margin_pred)
            feature_df["home_win_prob_calibrated"] = artifacts.win_prob_calibrator.predict(
                raw_prob
            )

        return pl.DataFrame(feature_df.to_dict(orient="records"))
