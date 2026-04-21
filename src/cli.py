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
    elif command == "test":
        test()
    else:
        raise SystemExit(f"Unknown command: {command}")
