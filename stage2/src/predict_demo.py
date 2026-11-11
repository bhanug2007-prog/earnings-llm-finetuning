"""Stage 2 demo: Long/Short prediction from numerical features with the trained Logistic Regression.

    python predict_demo.py --dataset full --event NVDA_2024-02-21
    python predict_demo.py --dataset base --event AAPL_2024-05-02

Or from Python:  from predict_demo import predict_event, predict_values
The explanation shows each feature's contribution = coefficient x standardized value (log-odds toward Long).
These are associations learned by a weak model, not causes, and the model is close to chance on test data.
"""
import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from preprocessing import DATA, ROOT, clean_name, feature_columns  # noqa: E402

MODELS = ROOT / "stage2" / "results" / "models"


def _explain(pipe, X):
    pre, lr = pipe.named_steps["pre"], pipe.named_steps["model"]
    z = pre.transform(X)[0]
    names = [clean_name(n) for n in pre.get_feature_names_out()]
    contrib = pd.DataFrame({"feature": names, "standardized_value": z, "coefficient": lr.coef_[0],
                            "contribution": z * lr.coef_[0]})
    return contrib.reindex(contrib.contribution.abs().sort_values(ascending=False).index)


def predict_values(values: dict, dataset="base", model="lr", top=8, verbose=True):
    """values: raw features as in stage2_<dataset>_numeric.csv (growth/returns/eps in %, analyst_grade text).
    Missing growth metrics can be left out (they are imputed like in training)."""
    pipe = joblib.load(MODELS / f"{model}_{dataset}.joblib")
    cols = feature_columns(dataset)
    X = pd.DataFrame([{c: values.get(c, np.nan) for c in cols}])
    if "analyst_grade" in cols and pd.isna(X.loc[0, "analyst_grade"]):
        X.loc[0, "analyst_grade"] = "none"
    p = float(pipe.predict_proba(X)[0, 1])
    out = {"p_long": p, "p_short": 1 - p, "prediction": "Long" if p >= 0.5 else "Short"}
    if verbose:
        print(f"Prediction: {out['prediction'].upper()}   P(Long) = {p:.4f}   P(Short) = {1 - p:.4f}")
    if model == "lr":
        ex = _explain(pipe, X)
        out["explanation"] = ex
        if verbose:
            print(f"Intercept: {pipe.named_steps['model'].intercept_[0]:+.4f} (log-odds)")
            print("Top contributions (log-odds; + pushes toward Long, - toward Short):")
            print(ex.head(top).round(4).to_string(index=False))
    return out


def predict_event(event_id, dataset="base", model="lr", top=8):
    df = pd.read_csv(DATA / f"stage2_{dataset}_numeric.csv").set_index("event_id")
    if event_id not in df.index:
        raise KeyError(f"{event_id} not in dataset; examples: {list(df.index[-5:])}")
    row = df.loc[event_id]
    print(f"{event_id} | split: {row.split} | true label: {row.label}")
    vals = {c: (row[c] if isinstance(row[c], str) else float(row[c])) for c in feature_columns(dataset)}
    print("Inputs:", {k: v for k, v in vals.items() if not (isinstance(v, float) and np.isnan(v))})
    return predict_values(vals, dataset, model, top)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["base", "full"], default="full")
    ap.add_argument("--model", choices=["lr", "rf"], default="lr")
    ap.add_argument("--event", default="NVDA_2024-02-21")
    a = ap.parse_args()
    predict_event(a.event, a.dataset, a.model)
