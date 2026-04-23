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
    elif command == "test":
        test()
    else:
        raise SystemExit(f"Unknown command: {command}")
