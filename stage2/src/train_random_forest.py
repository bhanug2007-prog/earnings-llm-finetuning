"""Stage 2 - Phase E (optional): Random Forest, Base and Full, same preprocessing and same selection rule
(max validation ROC-AUC; test never used for selection). Run only after the LR pipeline works."""
import sys
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_logistic_regression import SEED, run_experiment  # noqa: E402

RF_GRID = [{"max_depth": d, "class_weight": w} for w in [None, "balanced"] for d in [5, 10, None]]


def make_rf(max_depth, class_weight):
    return RandomForestClassifier(n_estimators=200, max_depth=max_depth, min_samples_leaf=5,
                                  class_weight=class_weight, random_state=SEED, n_jobs=-1)


def main():
    for ds in ["base", "full"]:
        run_experiment("RandomForestClassifier", "rf", make_rf, RF_GRID, ds)


if __name__ == "__main__":
    main()
