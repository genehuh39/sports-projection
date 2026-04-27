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
    """Walk-forward backtest against a synthetic uniform -110 market.

    Runs both raw and calibrated configurations side-by-side so the
    calibration impact is visible.
    """
    manager = NBAModelManager()
    raw = manager.backtest_evaluate(apply_calibration=False)
    if "error" in raw and not raw.get("folds"):
        raise SystemExit(f"Backtest failed: {raw['error']}")
    calibrated = manager.backtest_evaluate(apply_calibration=True)

    print(f"Seasons: {', '.join(raw['seasons'])}")
    print(
        f"Synthetic market: vig={raw['vig']:.4f}  "
        f"breakeven_prob={raw['breakeven_prob']:.4f}  "
        f"win_payout={raw['win_payout']:+.4f}"
    )
    print(f"Folds: {raw['n_folds']}")
    print()
    print(f"{'metric':<10}  {'raw':>10}  {'calibrated':>12}")
    print(
        f"{'Accuracy':<10}  {raw['accuracy_mean']:>10.3f}  "
        f"{calibrated['accuracy_mean']:>12.3f}"
    )
    print(
        f"{'Brier':<10}  {raw['brier_mean']:>10.4f}  "
        f"{calibrated['brier_mean']:>12.4f}"
    )
    print(
        f"{'Log loss':<10}  {raw['log_loss_mean']:>10.4f}  "
        f"{calibrated['log_loss_mean']:>12.4f}"
    )
    print()
    print("Calibration table (raw vs calibrated)")
    print(
        f"  {'bucket':<14}  {'n_raw':>5}  {'pred_r':>7}  {'act_r':>7}  "
        f"{'n_cal':>5}  {'pred_c':>7}  {'act_c':>7}"
    )
    raw_by_bucket = {(r["bucket_lo"], r["bucket_hi"]): r for r in raw["calibration"]}
    cal_by_bucket = {(r["bucket_lo"], r["bucket_hi"]): r for r in calibrated["calibration"]}
    for key in sorted(set(raw_by_bucket) | set(cal_by_bucket)):
        r = raw_by_bucket.get(key)
        c = cal_by_bucket.get(key)
        bucket = f"{key[0]:.1f}-{key[1]:.1f}"
        r_n = r["n"] if r else 0
        r_pred = f"{r['mean_predicted']:.3f}" if r else "    -  "
        r_act = f"{r['actual_rate']:.3f}" if r else "    -  "
        c_n = c["n"] if c else 0
        c_pred = f"{c['mean_predicted']:.3f}" if c else "    -  "
        c_act = f"{c['actual_rate']:.3f}" if c else "    -  "
        print(
            f"  {bucket:<14}  {r_n:>5}  {r_pred:>7}  {r_act:>7}  "
            f"{c_n:>5}  {c_pred:>7}  {c_act:>7}"
        )
    print()
    print("Synthetic-market ROI by edge threshold (raw vs calibrated)")
    print(
        f"  {'edge>':>6}  {'n_raw':>6}  {'roi_r':>9}  {'wr_r':>6}  "
        f"{'n_cal':>6}  {'roi_c':>9}  {'wr_c':>6}"
    )
    raw_by_t = {row["edge_threshold"]: row for row in raw["thresholds"]}
    cal_by_t = {row["edge_threshold"]: row for row in calibrated["thresholds"]}
    for t in sorted(set(raw_by_t) | set(cal_by_t)):
        r = raw_by_t.get(t)
        c = cal_by_t.get(t)
        r_n = r["n_bets"] if r else 0
        r_roi = f"{r['roi_mean'] * 100.0:+.2f}%" if r and r_n else "    -  "
        r_wr = f"{r['win_rate']:.3f}" if r and r_n else "   -  "
        c_n = c["n_bets"] if c else 0
        c_roi = f"{c['roi_mean'] * 100.0:+.2f}%" if c and c_n else "    -  "
        c_wr = f"{c['win_rate']:.3f}" if c and c_n else "   -  "
        print(
            f"  {t:>6.2f}  {r_n:>6}  {r_roi:>9}  {r_wr:>6}  "
            f"{c_n:>6}  {c_roi:>9}  {c_wr:>6}"
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
