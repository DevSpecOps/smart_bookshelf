import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

STEPS = [
    ("download_data", [sys.executable, "scripts/download_data.py"]),
    ("eda", [sys.executable, "scripts/run_eda.py"]),
    ("evaluate", [sys.executable, "src/evaluate.py"]),
    ("build_notebook", [sys.executable, "scripts/build_notebook.py"]),
    (
        "execute_notebook",
        [
            "jupyter",
            "nbconvert",
            "--to",
            "notebook",
            "--execute",
            "--inplace",
            "notebooks/report.ipynb",
        ],
    ),
]


def main():
    for name, cmd in STEPS:
        print(f"\n=== {name} ===")
        result = subprocess.run(cmd, cwd=ROOT)
        if result.returncode != 0:
            print(f"step '{name}' failed with exit code {result.returncode}", file=sys.stderr)
            return result.returncode
    print("\nall steps completed")
    return 0


if __name__ == "__main__":
    sys.exit(main())