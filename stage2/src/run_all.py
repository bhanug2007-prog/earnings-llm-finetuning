"""Rebuild every Stage 2 output from the Stage 1 dataset (CPU only, about 2-4 minutes).

    python stage2/src/run_all.py
"""
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RES = HERE.parent / "results"
STEPS = ["data_extraction.py", "verify_extraction.py", "train_logistic_regression.py", "train_random_forest.py",
         "compare.py", "visualization.py", "final_checks.py"]


def main():
    for s in STEPS:
        print(f"\n=== {s}")
        subprocess.run([sys.executable, "-W", "ignore", str(HERE / s)], cwd=HERE, check=True)
    cfg = {"description": "Stage 2 numerical-feature models; all configs and test/val metrics",
           "experiments": {p.stem.replace("config_", ""): json.load(open(p))
                           for p in sorted((RES / "metrics").glob("config_*.json"))}}
    json.dump(cfg, open(RES / "experiment_config.json", "w"), indent=2)
    print("\nwrote", RES / "experiment_config.json")


if __name__ == "__main__":
    main()
