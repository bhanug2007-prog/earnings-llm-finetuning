"""Stage 2 - Phase F/G: final test evaluation, comparison with Stage 1, paired bootstrap AUC tests and
feature analysis. Reads the saved Stage 2 test predictions and the Stage 1 per-event predictions (read only)."""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import bootstrap_ci, load_stage1, metrics, paired_auc_diff  # noqa: E402
from preprocessing import ROOT, clean_name, feature_columns, load  # noqa: E402

RES = ROOT / "stage2" / "results"
M = RES / "metrics"
S2 = {"Stage 2 LR (Base)": "lr_base", "Stage 2 LR (Full)": "lr_full",
      "Stage 2 RF (Base)": "rf_base", "Stage 2 RF (Full)": "rf_full"}
PAPER = [("Paper QLoRA", "Base", 0.550, 0.533, 0.104), ("Paper QLoRA", "Full", 0.573, 0.565, 0.154)]


def main():
    test = load("base")["test"]
    ids, y = test.event_id.values, test.y.values
    preds = {}
    for name, tag in S2.items():
        df = pd.read_csv(RES / "predictions" / f"pred_test_{tag}.csv").set_index("event_id").loc[ids]
        assert len(df) == 946 and (df.label.eq("Long").astype(int).values == y).all()
        preds[name] = df.p_long.values
    s1 = load_stage1(ids, y)          # asserts identical 946 IDs and labels
    preds.update(s1)

    rows, cis = [], {}
    for name, p in preds.items():
        m = metrics(y, p)
        ci = bootstrap_ci(y, p)
        cis[name] = ci
        kind = "Stage 2" if name.startswith("Stage 2") else ("Baseline" if name == "Always Short" else "Stage 1")
        ds = "Base" if "Base" in name else ("Full" if "Full" in name else "-")
        rows.append({"model": name, "stage": kind, "dataset": ds, **m,
                     "accuracy_ci": ci["accuracy"], "roc_auc_ci": ci["roc_auc"], "mcc_ci": ci["mcc"],
                     "f1_weighted_ci": ci["f1_weighted"]})
    table = pd.DataFrame(rows)
    paper = pd.DataFrame([{"model": f"{a} ({b})", "stage": "Paper", "dataset": b, "accuracy": c, "f1_weighted": d,
                           "mcc": e} for a, b, c, d, e in PAPER])
    full_table = pd.concat([paper, table], ignore_index=True)
    full_table.to_csv(M / "comparison_test.csv", index=False)

    # paired bootstrap: each Stage 2 model vs every Stage 1 model / baseline, same 946 events
    paired = []
    for a in S2:
        for b in s1:
            r = paired_auc_diff(y, preds[a], preds[b])
            paired.append({"stage2_model": a, "vs": b, **r})
    for a, b in [("Stage 2 LR (Full)", "Stage 2 LR (Base)"), ("Stage 2 RF (Full)", "Stage 2 RF (Base)"),
                 ("Stage 2 LR (Base)", "Stage 2 RF (Base)"), ("Stage 2 LR (Full)", "Stage 2 RF (Full)")]:
        paired.append({"stage2_model": a, "vs": b, **paired_auc_diff(y, preds[a], preds[b])})
    pdf = pd.DataFrame(paired)
    pdf.to_csv(M / "paired_bootstrap_auc.csv", index=False)

    # validation metrics of the selected Stage 2 models
    val = {}
    for name, tag in S2.items():
        cfg = json.load(open(M / f"config_{tag}.json"))
        val[name] = {"selected_params": cfg["selected_params"], "val": cfg["metrics"]["val"]}

    # ---- feature analysis
    fa = {}
    for ds in ["base", "full"]:
        pipe = joblib.load(RES / "models" / f"lr_{ds}.joblib")
        names = [clean_name(n) for n in pipe.named_steps["pre"].get_feature_names_out()]
        coef = pipe.named_steps["model"].coef_[0]
        c = pd.DataFrame({"feature": names, "coefficient": coef, "abs": np.abs(coef)}).sort_values("abs", ascending=False)
        c.to_csv(M / f"lr_coefficients_{ds}.csv", index=False)
        fa[f"lr_{ds}"] = {"intercept": float(pipe.named_steps["model"].intercept_[0]),
                          "top_positive": c[c.coefficient > 0].head(5)[["feature", "coefficient"]].values.tolist(),
                          "top_negative": c[c.coefficient < 0].head(5)[["feature", "coefficient"]].values.tolist()}
        # RF: impurity importance + permutation importance on VALIDATION (input columns)
        rf = joblib.load(RES / "models" / f"rf_{ds}.joblib")
        imp = pd.DataFrame({"feature": [clean_name(n) for n in rf.named_steps["pre"].get_feature_names_out()],
                            "impurity_importance": rf.named_steps["model"].feature_importances_})
        imp.sort_values("impurity_importance", ascending=False).to_csv(M / f"rf_impurity_importance_{ds}.csv", index=False)
        v = load(ds)["val"]
        cols = feature_columns(ds)
        pi = permutation_importance(rf, v[cols], v.y.values, scoring="roc_auc", n_repeats=20, random_state=42, n_jobs=-1)
        pim = pd.DataFrame({"input_feature": cols, "val_auc_drop_mean": pi.importances_mean,
                            "val_auc_drop_std": pi.importances_std}).sort_values("val_auc_drop_mean", ascending=False)
        pim.to_csv(M / f"rf_permutation_importance_val_{ds}.csv", index=False)
        fa[f"rf_{ds}"] = {"top_permutation_val": pim.head(5).values.tolist()}

    summary = {"test_events": 946, "same_ids_as_stage1": True,
               "test_metrics": {r["model"]: {k: r[k] for k in ["accuracy", "f1_weighted", "mcc", "roc_auc",
                                                               "precision_long", "recall_long", "log_loss", "brier",
                                                               "share_pred_long", "tn", "fp", "fn", "tp"]}
                                for r in rows},
               "test_ci95": cis, "validation_selected": val, "paired_bootstrap_auc": paired,
               "feature_analysis": fa, "paper_reported": [dict(zip(["model", "dataset", "accuracy", "f1_weighted", "mcc"], p))
                                                          for p in PAPER]}
    json.dump(summary, open(M / "stage2_summary.json", "w"), indent=2, default=float)
    pd.set_option("display.width", 220)
    print(full_table[["model", "accuracy", "f1_weighted", "mcc", "roc_auc", "roc_auc_ci", "share_pred_long"]]
          .round(3).to_string(index=False))
    print(pdf.round(3).to_string(index=False))
    print(json.dumps(fa, indent=1, default=float)[:3000])


if __name__ == "__main__":
    main()
