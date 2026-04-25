import runpy
import sys
import unittest
from pathlib import Path

from src.models.trained_nba_model import NBAModelManager
from src.pipeline import run_full_pipeline


def main() -> None:
    """Run the terminal pipeline."""
    run_full_pipeline()


def dashboard() -> None:
    """Run the NiceGUI dashboard."""
    runpy.run_path(str(Path(__file__).resolve().parents[1] / "run_final.py"), run_name="__main__")


def train() -> None:
    """Train and persist the NBA projection model."""
    artifacts = NBAModelManager(auto_train=True).train_and_save(force=True)
    if artifacts is None:
        raise SystemExit("Training did not produce a model artifact.")
    print("Model trained.")
    print(f"Seasons: {', '.join(artifacts.seasons)}")
    print(f"Margin MAE: {artifacts.metrics.get('margin_mae', 0):.2f}")
    print(f"Total MAE: {artifacts.metrics.get('total_mae', 0):.2f}")


def evaluate() -> None:
    """Walk-forward cross-validation of the NBA projection model."""
    n_folds = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    test_size = int(sys.argv[3]) if len(sys.argv) > 3 else 150
    results = NBAModelManager().walk_forward_evaluate(n_folds=n_folds, test_size=test_size)

    if "error" in results and not results.get("folds"):
        raise SystemExit(f"Evaluation failed: {results['error']}")

    print(f"Seasons: {', '.join(results['seasons'])}")
    print(f"Features: {results['feature_count']}")
    print(f"Folds: {results['n_folds']}")
    print()
    print(f"{'fold':>4}  {'dates':<25}  {'train':>6}  {'test':>5}  {'m_mae':>6}  {'t_mae':>6}")
    for f in results["folds"]:
        dates = f"{f['test_start_date']}..{f['test_end_date']}"
        print(
            f"{f['fold']:>4}  {dates:<25}  {f['train_rows']:>6}  "
            f"{f['test_rows']:>5}  {f['margin_mae']:>6.2f}  {f['total_mae']:>6.2f}"
        )
    print()
    print(
        f"Margin MAE: {results['margin_mae_mean']:.2f} ± {results['margin_mae_std']:.2f}"
    )
    print(
        f"Total  MAE: {results['total_mae_mean']:.2f} ± {results['total_mae_std']:.2f}"
    )


def calibrate() -> None:
    """Sweep injury damping factors against walk-forward CV."""
    results = NBAModelManager().calibrate_injury_damping()
    if "error" in results and not results.get("grid"):
        raise SystemExit(f"Calibration failed: {results['error']}")

    print(f"Seasons: {', '.join(results['seasons'])}")
    print(f"Folds: {results['n_folds']}")
    print(f"Top-N rotation: {results['top_n_players']} (min fraction {results['min_fraction']})")
    print()
    print(f"{'damping':>8}  {'margin mae':>16}  {'total mae':>16}")
    for row in results["grid"]:
        print(
            f"{row['damping']:>8.2f}  "
            f"{row['margin_mae_mean']:>6.3f} ± {row['margin_mae_std']:>4.2f}   "
            f"{row['total_mae_mean']:>6.3f} ± {row['total_mae_std']:>4.2f}"
        )
    print()
    print(
        f"Best damping for margin: {results['best_margin_damping']:.2f} "
        f"(baseline 0.0 = {results['baseline_margin_mae']:.3f})"
    )
    print(
        f"Best damping for total:  {results['best_total_damping']:.2f} "
        f"(baseline 0.0 = {results['baseline_total_mae']:.3f})"
    )


def backtest() -> None:
    """Walk-forward backtest against a synthetic uniform -110 market."""
    results = NBAModelManager().backtest_evaluate()
    if "error" in results and not results.get("folds"):
        raise SystemExit(f"Backtest failed: {results['error']}")

    print(f"Seasons: {', '.join(results['seasons'])}")
    print(
        f"Synthetic market: vig={results['vig']:.4f}  "
        f"breakeven_prob={results['breakeven_prob']:.4f}  "
        f"win_payout={results['win_payout']:+.4f}"
    )
    print(f"Folds: {results['n_folds']}")
    print()
    print(f"Accuracy: {results['accuracy_mean']:.3f} ± {results['accuracy_std']:.3f}")
    print(f"Brier:    {results['brier_mean']:.4f}")
    print(f"Log loss: {results['log_loss_mean']:.4f}")
    print()
    print("Calibration table (predicted probability -> actual home-win rate)")
    print(f"  {'bucket':<14}  {'n':>5}  {'pred':>7}  {'actual':>7}")
    for row in results["calibration"]:
        bucket = f"{row['bucket_lo']:.1f}-{row['bucket_hi']:.1f}"
        print(
            f"  {bucket:<14}  {row['n']:>5}  "
            f"{row['mean_predicted']:>7.3f}  {row['actual_rate']:>7.3f}"
        )
    print()
    print("Synthetic-market ROI by edge threshold")
    print(f"  {'edge>':>6}  {'n_bets':>7}  {'roi':>9}  {'win_rate':>9}")
    for row in results["thresholds"]:
        roi_pct = row["roi_mean"] * 100.0
        print(
            f"  {row['edge_threshold']:>6.2f}  {row['n_bets']:>7}  "
            f"{roi_pct:>+8.2f}%  {row['win_rate']:>9.3f}"
        )
    print()
    print(
        "Note: ROI is against a uniform -110 market — a benchmark, not a "
        "real-world PnL number. Real bookmaker prices will differ."
    )


def test() -> None:
    """Run the unittest suite."""
    suite = unittest.defaultTestLoader.discover("tests")
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(0 if result.wasSuccessful() else 1)


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else "main"
    if command == "main":
        main()
    elif command == "dashboard":
        dashboard()
    elif command == "train":
        train()
    elif command == "evaluate":
        evaluate()
    elif command == "calibrate":
        calibrate()
    elif command == "backtest":
        backtest()
    elif command == "test":
        test()
    else:
        raise SystemExit(f"Unknown command: {command}")
