"""Stage 2 - Phase F: metrics, bootstrap confidence intervals and paired bootstrap comparisons."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, brier_score_loss, confusion_matrix, f1_score, log_loss,
                             matthews_corrcoef, precision_score, recall_score, roc_auc_score)

ROOT = Path(__file__).resolve().parents[2]
S1 = ROOT / "stage1" / "4_results" / "final_summarized" if (ROOT / "stage1" / "4_results" / "final_summarized").exists() else ROOT / "stage1" / "4_results"
N_BOOT, SEED = 2000, 42

# Stage 1 runs with per-event test predictions (read only, never modified)
STAGE1_RUNS = {
    "Our QLoRA (Base)": "pred_test_base_time_3072_train_eval_sum.csv",
    "Our QLoRA (Full)": "pred_test_full_time_3072_train_eval_sum.csv",
    "Our zero-shot (Base)": "pred_test_base_time_3072_zeroshot_sum.csv",
    "Our zero-shot (Full)": "pred_test_full_time_3072_zeroshot_sum.csv",
}


def metrics(y, p, thr=0.5):
    y, p = np.asarray(y), np.asarray(p, dtype=float)
    pred = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    pc = np.clip(p, 1e-6, 1 - 1e-6)
    return {
        "accuracy": accuracy_score(y, pred),
        "precision_long": precision_score(y, pred, zero_division=0),
        "recall_long": recall_score(y, pred, zero_division=0),
        "f1_weighted": f1_score(y, pred, average="weighted", zero_division=0),
        "mcc": matthews_corrcoef(y, pred),
        "roc_auc": roc_auc_score(y, p) if len(set(y)) > 1 else np.nan,
        "log_loss": log_loss(y, pc, labels=[0, 1]),
        "brier": brier_score_loss(y, p),
        "share_pred_long": float(pred.mean()),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


CI_KEYS = ["accuracy", "f1_weighted", "mcc", "roc_auc", "precision_long", "recall_long", "log_loss", "brier"]


def bootstrap_ci(y, p, thr=0.5, n=N_BOOT, seed=SEED):
    """Percentile 95% CI, resampling events with replacement."""
    y, p = np.asarray(y), np.asarray(p, dtype=float)
    rng = np.random.default_rng(seed)
    out = {k: [] for k in CI_KEYS}
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        m = metrics(y[i], p[i], thr)
        for k in CI_KEYS:
            out[k].append(m[k])
    return {k: [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))] for k, v in out.items()}


def paired_auc_diff(y, p_a, p_b, n=N_BOOT, seed=SEED):
    """AUC(a) - AUC(b) on the SAME events; the same resample is used for both models.
    Two-sided p-value = 2 * min(share of diffs <= 0, share >= 0)."""
    y, p_a, p_b = map(np.asarray, (y, p_a, p_b))
    obs = roc_auc_score(y, p_a) - roc_auc_score(y, p_b)
    rng = np.random.default_rng(seed)
    d = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if len(set(y[i])) < 2:
            continue
        d.append(roc_auc_score(y[i], p_a[i]) - roc_auc_score(y[i], p_b[i]))
    d = np.array(d)
    pval = min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))
    return {"auc_diff": float(obs), "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
            "p_value_two_sided": float(pval), "n_boot": int(len(d))}


def load_stage1(test_ids, test_y):
    """Stage 1 per-event test predictions aligned to Stage 2 test order. Asserts identical IDs and labels."""
    out = {}
    lab = dict(zip(test_ids, test_y))
    for name, f in STAGE1_RUNS.items():
        df = pd.read_csv(S1 / f)
        assert len(df) == 946 and df.event_id.is_unique, f
        assert set(df.event_id) == set(test_ids), f"{f}: event IDs differ from Stage 2 test set"
        df = df.set_index("event_id").loc[list(test_ids)]
        assert ((df.label == "Long").astype(int).values == np.asarray(test_y)).all(), f"{f}: labels differ"
        out[name] = df.p_long.astype(float).values
    out["Always Short"] = np.zeros(len(test_ids))   # P(Long) = 0 for every event -> AUC 0.5
    return out
