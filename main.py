"""Main launcher for the Facial Recognition project."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

SCRIPTS = {
    "run": PROJECT_ROOT / "src" / "face_recognition_webcam.py",
    "train": PROJECT_ROOT / "src" / "train_face_recognizer.py",
    "calibrate": PROJECT_ROOT / "src" / "calibrate_phone.py",
}


def run_script(name: str) -> int:
    script = SCRIPTS[name]

    if not script.exists():
        print(f"Error: required file was not found: {script}")
        return 1

    print(f"\nStarting: {script.name}\n")
    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=PROJECT_ROOT,
    )
    return result.returncode


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Face Recognition ML Project launcher"
    )

    parser.add_argument(
        "mode",
        nargs="?",
        default="run",
        choices=("run", "train", "calibrate", "all"),
        help="run webcam (default), train model, collect calibration, or train then run",
    )

    args = parser.parse_args()

    print("=" * 58)
    print(" FACIAL RECOGNITION ML PROJECT")
    print("=" * 58)

    if args.mode == "all":
        if run_script("train") != 0:
            return 1
        return run_script("run")

    return run_script(args.mode)


if __name__ == "__main__":
    raise SystemExit(main())