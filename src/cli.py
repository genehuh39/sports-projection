import runpy
import sys
import unittest
from pathlib import Path

from src.pipeline import run_full_pipeline


def main() -> None:
    """Run the terminal pipeline."""
    run_full_pipeline()


def dashboard() -> None:
    """Run the NiceGUI dashboard."""
    runpy.run_path(str(Path(__file__).resolve().parents[1] / "run_final.py"), run_name="__main__")


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
    elif command == "test":
        test()
    else:
        raise SystemExit(f"Unknown command: {command}")
