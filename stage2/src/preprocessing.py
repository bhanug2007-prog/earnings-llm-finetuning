"""Stage 2 - Phase C: leakage-safe preprocessing.

Everything that is *fitted* (median imputer, one-hot encoder, StandardScaler) lives in one scikit-learn
ColumnTransformer that is fitted on the TRAINING split only and then applied to validation and test.

Per feature group:
  growth metrics (12)   signed log  ->  median impute (train) + 0/1 missing flags  ->  StandardScaler (train)
  eps_surprise          signed log  ->  StandardScaler (train)          (no missing values)
  past-week returns (4) StandardScaler (train)                          (no missing values, mild range)
  analyst_grade         one-hot {buy, hold, none, sell}, categories learnt on train, unknown -> all zeros

signed log: x -> sign(x) * log(1 + |x| / 100). x is in percent, so |x| = 100% maps to log 2 = 0.69 and
1e9 % to about 16. It is a fixed formula (nothing is estimated), keeps sign and order, and stops a few
extreme growth values (from tiny prior-year bases) from dominating a linear model.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "stage2" / "data"

GROWTH = ["g_revenue", "g_gross_profit", "g_operating_income", "g_net_income", "g_eps_diluted",
          "g_operating_cash_flow", "g_capital_expenditure", "g_total_assets", "g_total_liabilities",
          "g_shareholders_equity", "g_cash_and_equivalents", "g_free_cash_flow"]
RETURNS = ["ret_stock_1w", "ret_spy_1w", "ret_qqq_1w", "ret_dow_1w"]
EPS = ["eps_surprise"]
GRADE = ["analyst_grade"]
META = ["event_id", "ticker", "date", "split", "label", "y"]


def feature_columns(dataset):
    return GROWTH if dataset == "base" else GROWTH + RETURNS + EPS + GRADE


def load(dataset):
    """Returns {split: DataFrame} keeping the original chronological Stage 1 split."""
    df = pd.read_csv(DATA / f"stage2_{dataset}_numeric.csv")
    assert df.event_id.is_unique
    return {s: df[df.split == s].reset_index(drop=True) for s in ["train", "val", "test"]}


def signed_log(x):
    x = np.asarray(x, dtype=float)
    return np.sign(x) * np.log1p(np.abs(x) / 100.0)


def _signed_log_names(transformer, names):
    return names


def build_preprocessor(dataset):
    slog = FunctionTransformer(signed_log, feature_names_out=_signed_log_names)
    parts = [("growth", Pipeline([("slog", slog),
                                  ("impute", SimpleImputer(strategy="median", add_indicator=True)),
                                  ("scale", StandardScaler())]), GROWTH)]
    if dataset == "full":
        parts += [
            ("eps", Pipeline([("slog", FunctionTransformer(signed_log, feature_names_out=_signed_log_names)),
                              ("scale", StandardScaler())]), EPS),
            ("returns", StandardScaler(), RETURNS),
            ("grade", OneHotEncoder(handle_unknown="ignore", sparse_output=False), GRADE),
        ]
    return ColumnTransformer(parts, remainder="drop", verbose_feature_names_out=True)


def clean_name(n):
    """'growth__missingindicator_g_revenue' -> 'missing: g_revenue', 'grade__analyst_grade_buy' -> 'analyst_grade=buy'"""
    n = n.split("__", 1)[1]
    if n.startswith("missingindicator_"):
        return "missing: " + n[len("missingindicator_"):]
    if n.startswith("analyst_grade_"):
        return "analyst_grade=" + n[len("analyst_grade_"):]
    return n


def data_checks(splits, dataset):
    """Checks on raw features (all splits) and on the transformed TRAIN matrix only (correlations)."""
    cols = feature_columns(dataset)
    num = [c for c in cols if c not in GRADE]
    rep = {"dataset": dataset}
    allx = pd.concat(splits.values())
    rep["duplicate_event_ids"] = int(allx.event_id.duplicated().sum())
    rep["infinite_values"] = int(np.isinf(allx[num].to_numpy(dtype=float)).sum())
    rep["missing_pct"] = {s: {c: round(100 * d[c].isna().mean(), 2) for c in cols if d[c].isna().any()}
                          for s, d in splits.items()}
    rep["constant_columns_train"] = [c for c in cols if splits["train"][c].nunique(dropna=True) <= 1]
    rep["label_in_features"] = any(c in cols for c in ["y", "label"])
    rep["identifier_in_features"] = any(c in cols for c in ["event_id", "ticker", "date", "split"])
    pre = build_preprocessor(dataset).fit(splits["train"][cols])
    Xt = pd.DataFrame(pre.transform(splits["train"][cols]), columns=[clean_name(n) for n in pre.get_feature_names_out()])
    corr = Xt.corr()
    pairs = [(a, b, round(float(corr.loc[a, b]), 3)) for i, a in enumerate(corr.columns) for b in corr.columns[i + 1:]
             if abs(corr.loc[a, b]) > 0.9]
    rep["high_corr_pairs_train_abs_gt_0.9"] = pairs
    rep["n_model_features"] = Xt.shape[1]
    rep["model_features"] = list(Xt.columns)
    return rep, corr
