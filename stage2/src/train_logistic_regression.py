"""Stage 2 - Phase D: Logistic Regression (main model), Base and Full.

Model selection: every grid setting is fitted on TRAIN and scored by ROC-AUC on VALIDATION.
The best setting (highest validation AUC; ties -> first in grid order, i.e. stronger regularisation)
is the final model. The TEST split is scored once, after the choice is fixed, and is never used to choose.
Decision threshold: 0.5 (as in Stage 1).
"""
import json
import platform
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from preprocessing import ROOT, build_preprocessor, clean_name, data_checks, feature_columns, load  # noqa: E402
from evaluate import metrics  # noqa: E402

RES = ROOT / "stage2" / "results"
SEED = 42
LR_GRID = [{"C": c, "class_weight": w} for w in [None, "balanced"] for c in [0.001, 0.01, 0.1, 1.0, 10.0]]


def make_lr(C, class_weight):
    # L2 penalty is the sklearn default (l1_ratio=0); passing penalty="l2" is deprecated in sklearn 1.8+
    return LogisticRegression(C=C, class_weight=class_weight, solver="lbfgs", max_iter=2000,
                              random_state=SEED)


def save_predictions(split_df, p, tag, split):
    out = pd.DataFrame({"event_id": split_df.event_id, "date": split_df.date, "label": split_df.label,
                        "pred": np.where(p >= 0.5, "Long", "Short"), "p_long": p, "p_short": 1 - p})
    path = RES / "predictions" / f"pred_{split}_{tag}.csv"
    out.to_csv(path, index=False)
    return path


def run_experiment(model_name, tag_prefix, make_model, grid, dataset):
    """Generic train -> validate -> select -> test runner (used by LR and RF)."""
    for d in ["predictions", "metrics", "models"]:
        (RES / d).mkdir(parents=True, exist_ok=True)
    splits = load(dataset)
    cols = feature_columns(dataset)
    Xtr, ytr = splits["train"][cols], splits["train"].y.values
    Xva, yva = splits["val"][cols], splits["val"].y.values

    search = []
    for params in grid:
        pipe = Pipeline([("pre", build_preprocessor(dataset)), ("model", make_model(**params))]).fit(Xtr, ytr)
        pva = pipe.predict_proba(Xva)[:, 1]
        search.append({**{k: (v if v is not None else "None") for k, v in params.items()},
                       "val_auc": float(roc_auc_score(yva, pva)),
                       "val_accuracy": float(((pva >= 0.5) == yva).mean()),
                       "val_share_pred_long": float((pva >= 0.5).mean())})
    best_i = int(np.argmax([s["val_auc"] for s in search]))       # argmax -> first of ties
    best = grid[best_i]
    tag = f"{tag_prefix}_{dataset}"
    pipe = Pipeline([("pre", build_preprocessor(dataset)), ("model", make_model(**best))]).fit(Xtr, ytr)
    joblib.dump(pipe, RES / "models" / f"{tag}.joblib")

    probs = {s: pipe.predict_proba(splits[s][cols])[:, 1] for s in ["train", "val", "test"]}
    for s in ["val", "test"]:
        save_predictions(splits[s], probs[s], tag, s)
    m = {s: metrics(splits[s].y.values, probs[s]) for s in ["train", "val", "test"]}
    pd.DataFrame(search).to_csv(RES / "metrics" / f"search_{tag}.csv", index=False)
    feat_names = [clean_name(n) for n in pipe.named_steps["pre"].get_feature_names_out()]
    cfg = {
        "experiment": tag, "model": model_name, "dataset": dataset,
        "input_features": cols, "model_features_after_preprocessing": feat_names,
        "n_train": len(ytr), "n_val": len(yva), "n_test": len(splits["test"]),
        "grid": [{k: (v if v is not None else "None") for k, v in g.items()} for g in grid],
        "selection": "max validation ROC-AUC (test never used)", "selected_params":
            {k: (v if v is not None else "None") for k, v in best.items()},
        "fixed_params": {k: (v if not callable(v) else str(v)) for k, v in pipe.named_steps["model"].get_params().items()
                         if isinstance(v, (int, float, str, type(None), bool))},
        "threshold": 0.5, "seed": SEED, "search": search,
        "metrics": {s: {k: (float(v) if isinstance(v, (float, np.floating)) else v) for k, v in mm.items()}
                    for s, mm in m.items()},
        "preprocessing": "signed log on growth + eps_surprise; train-median imputation + missing flags on growth; "
                         "one-hot analyst_grade; StandardScaler; all fitted on train only",
        "versions": {"python": platform.python_version(), "sklearn": sklearn.__version__,
                     "pandas": pd.__version__, "numpy": np.__version__},
    }
    json.dump(cfg, open(RES / "metrics" / f"config_{tag}.json", "w"), indent=2, default=str)
    print(f"[{tag}] best {best} | val AUC {search[best_i]['val_auc']:.4f} | "
          f"test acc {m['test']['accuracy']:.4f} AUC {m['test']['roc_auc']:.4f} | "
          f"test share Long {m['test']['share_pred_long']:.2f}")
    return cfg, pipe


def main():
    (RES / "metrics").mkdir(parents=True, exist_ok=True)
    checks = {}
    for ds in ["base", "full"]:
        rep, corr = data_checks(load(ds), ds)
        checks[ds] = rep
        corr.to_csv(RES / "metrics" / f"train_feature_correlation_{ds}.csv")
        assert rep["duplicate_event_ids"] == 0 and rep["infinite_values"] == 0
        assert not rep["label_in_features"] and not rep["identifier_in_features"]
    json.dump(checks, open(RES / "metrics" / "data_checks.json", "w"), indent=2)
    for ds in ["base", "full"]:
        print(ds, "features after preprocessing:", checks[ds]["n_model_features"],
              "| constant:", checks[ds]["constant_columns_train"],
              "| |r|>0.9 pairs:", checks[ds]["high_corr_pairs_train_abs_gt_0.9"])
        run_experiment("LogisticRegression", "lr", make_lr, LR_GRID, ds)


if __name__ == "__main__":
    main()
