"""Stage 2 - Phase H: figures. Reads only saved results (metrics, predictions, models); no numbers are typed in
except the paper's published values, which are labelled as such."""
import json
import sys
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_curve  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import load_stage1  # noqa: E402
from preprocessing import ROOT, load  # noqa: E402

RES = ROOT / "stage2" / "results"
P = RES / "plots"
M = RES / "metrics"

# reference palette (dataviz skill, light mode), fixed order by model family
C = {"Stage 2 LR": "#2a78d6", "Stage 2 RF": "#eb6834", "Our QLoRA": "#1baf7a", "Our zero-shot": "#6250d6",
     "Paper QLoRA": "#8a8984", "Always Short": "#c3c2b7"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.titleweight": "bold", "axes.titlesize": 11, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 150, "savefig.bbox": "tight",
                     "figure.facecolor": "white"})


def family(name):
    for k in C:
        if name.startswith(k):
            return k
    return "Always Short"


def metric_comparison(tab):
    order = ["Paper QLoRA (Base)", "Paper QLoRA (Full)", "Our QLoRA (Base)", "Our QLoRA (Full)",
             "Our zero-shot (Base)", "Our zero-shot (Full)", "Always Short",
             "Stage 2 LR (Base)", "Stage 2 LR (Full)", "Stage 2 RF (Base)", "Stage 2 RF (Full)"]
    tab = tab.set_index("model").loc[order]
    mets = [("accuracy", "Accuracy", 0.5), ("f1_weighted", "Weighted F1", None), ("mcc", "MCC", 0.0),
            ("roc_auc", "ROC-AUC", 0.5)]
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.8), sharey=True)
    ypos = np.arange(len(order))[::-1]
    for ax, (k, title, ref) in zip(axes, mets):
        vals = tab[k].astype(float).values
        cis = tab[f"{k}_ci"] if f"{k}_ci" in tab else None
        for i, (name, v) in enumerate(zip(order, vals)):
            if np.isnan(v):
                ax.text(0.01, ypos[i], "not reported", va="center", fontsize=8, color=INK2,
                        transform=ax.get_yaxis_transform())
                continue
            ax.barh(ypos[i], v, height=0.62, color=C[family(name)],
                    hatch="//" if name == "Always Short" else None, edgecolor="white", linewidth=0.5)
            ci = cis.iloc[i] if cis is not None else None
            xt = max(v, 0)
            if isinstance(ci, (list, tuple, str)) and not (isinstance(ci, float)):
                lo, hi = json.loads(ci) if isinstance(ci, str) else ci
                ax.plot([lo, hi], [ypos[i]] * 2, color=INK, lw=1)
                xt = max(hi, xt)
            ax.text(xt, ypos[i], f" {v:.3f}", va="center", fontsize=7.5, color=INK)
        if ref is not None:
            ax.axvline(ref, color=INK2, lw=0.8, ls="--")
        ax.set_title(title)
        ax.grid(axis="x", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
    axes[0].set_yticks(ypos, order)
    axes[0].set_xlim(0.3, 0.65); axes[1].set_xlim(0.3, 0.65); axes[2].set_xlim(-0.08, 0.22); axes[3].set_xlim(0.4, 0.64)
    fig.suptitle("Test set (946 events, Jan-Jun 2024): Stage 1 vs Stage 2.  Bars = point estimate, lines = 95% bootstrap CI",
                 fontsize=11, x=0.01, ha="left")
    fig.text(0.01, -0.02, "Paper values are as published (no AUC, no CI). Dashed line = chance level. "
             "Always Short has P(Long)=0 for every event.", fontsize=8, color=INK2)
    fig.savefig(P / "01_metric_comparison_stage1_vs_stage2.png"); plt.close(fig)
    # single-metric charts as separately named files
    for k, title, ref in mets[:3]:
        fig, ax = plt.subplots(figsize=(7, 4.6))
        sub = tab.drop(index=[n for n in order if np.isnan(tab.loc[n, k])])
        yp = np.arange(len(sub))[::-1]
        ax.barh(yp, sub[k], color=[C[family(n)] for n in sub.index], height=0.62,
                hatch=None, edgecolor="white")
        for y_, n in zip(yp, sub.index):
            ci = sub.loc[n, f"{k}_ci"] if f"{k}_ci" in sub else None
            xt = max(sub.loc[n, k], 0)
            if isinstance(ci, (list, tuple)):
                ax.plot(ci, [y_] * 2, color=INK, lw=1)
                xt = max(xt, ci[1])
            ax.text(xt, y_, f" {sub.loc[n, k]:.3f}", va="center", fontsize=8)
        ax.set_yticks(yp, sub.index)
        if ref is not None:
            ax.axvline(ref, color=INK2, lw=0.8, ls="--")
        ax.set_title(f"{title} on the 946 test events (95% bootstrap CI where available)", loc="left")
        ax.grid(axis="x", color=GRID, lw=0.6); ax.set_axisbelow(True)
        if k == "mcc":
            ax.set_xlim(-0.08, 0.2)
        else:
            ax.set_xlim(0.3, 0.65)
        fig.savefig(P / f"0{2 + mets.index((k, title, ref))}_{k}_comparison.png"); plt.close(fig)


def roc_plot(y, preds):
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 5))
    for ax, ds in zip(axes, ["Base", "Full"]):
        for name in [f"Stage 2 LR ({ds})", f"Stage 2 RF ({ds})", f"Our QLoRA ({ds})", f"Our zero-shot ({ds})"]:
            fpr, tpr, _ = roc_curve(y, preds[name])
            from sklearn.metrics import roc_auc_score
            ax.plot(fpr, tpr, lw=2, color=C[family(name)], label=f"{name.split(' (')[0]}  AUC {roc_auc_score(y, preds[name]):.3f}")
        ax.plot([0, 1], [0, 1], ls="--", color=INK2, lw=0.8, label="Chance (AUC 0.5)")
        ax.set_title(f"{ds} features: ROC curves on the test set", loc="left")
        ax.set_xlabel("False positive rate (Short called Long)"); ax.set_ylabel("True positive rate (Long called Long)")
        ax.set_aspect("equal"); ax.grid(color=GRID, lw=0.6); ax.legend(fontsize=8, frameon=False, loc="lower right")
    fig.savefig(P / "05_roc_curves.png"); plt.close(fig)


def confusion_plot(summary):
    names = ["Stage 2 LR (Base)", "Stage 2 LR (Full)", "Stage 2 RF (Base)", "Stage 2 RF (Full)"]
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.6))
    for ax, n in zip(axes, names):
        m = summary["test_metrics"][n]
        cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
        ax.imshow(cm, cmap="Blues", vmin=0, vmax=cm.sum())
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]}\n({cm[i, j] / cm.sum():.0%})", ha="center", va="center",
                        color="white" if cm[i, j] > cm.sum() * 0.45 else INK, fontsize=10)
        ax.set_xticks([0, 1], ["Short", "Long"]); ax.set_yticks([0, 1], ["Short", "Long"])
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title(n, fontsize=10)
        for s in ax.spines.values():
            s.set_visible(False)
    fig.suptitle("Confusion matrices on the 946 test events (threshold 0.5)", x=0.01, ha="left", fontsize=11)
    fig.savefig(P / "06_confusion_matrices.png"); plt.close(fig)


def coef_plot():
    fig, axes = plt.subplots(1, 2, figsize=(14, 7.5), gridspec_kw={"wspace": 0.75})
    for ax, ds in zip(axes, ["base", "full"]):
        c = pd.read_csv(M / f"lr_coefficients_{ds}.csv").sort_values("coefficient")
        ax.barh(c.feature, c.coefficient, color=np.where(c.coefficient > 0, "#2a78d6", "#eb6834"), height=0.7)
        ax.axvline(0, color=INK2, lw=0.8)
        ax.set_title(f"LR {ds.capitalize()}: standardized coefficients", loc="left")
        ax.set_xlabel("Coefficient: log-odds of Long per 1 SD\n(blue = toward Long, orange = toward Short)")
        ax.grid(axis="x", color=GRID, lw=0.6); ax.set_axisbelow(True); ax.tick_params(axis="y", labelsize=8)
    fig.text(0.01, -0.01, "Associations learned by a strongly regularized model (C = 0.001); not causal effects.",
             fontsize=8, color=INK2)
    fig.savefig(P / "07_lr_coefficients.png"); plt.close(fig)


def corr_plot():
    c = pd.read_csv(M / "train_feature_correlation_full.csv", index_col=0)
    fig, ax = plt.subplots(figsize=(11, 9.5))
    im = ax.imshow(c.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(c)), c.columns, rotation=90, fontsize=7); ax.set_yticks(range(len(c)), c.index, fontsize=7)
    fig.colorbar(im, ax=ax, shrink=0.7, label="Pearson r (training split, after preprocessing)")
    ax.set_title("Feature correlations, Full feature set, training data only", loc="left")
    fig.savefig(P / "08_feature_correlation_heatmap.png"); plt.close(fig)


def base_vs_full(summary):
    rows = [("LR", "Stage 2 LR"), ("RF", "Stage 2 RF"), ("QLoRA", "Our QLoRA"), ("Zero-shot", "Our zero-shot")]
    fig, ax = plt.subplots(figsize=(8, 4.3))
    x = np.arange(len(rows))
    for off, ds, col in [(-0.18, "Base", "#2a78d6"), (0.18, "Full", "#eb6834")]:
        v = [summary["test_metrics"][f"{r[1]} ({ds})"]["roc_auc"] for r in rows]
        lo = [summary["test_ci95"][f"{r[1]} ({ds})"]["roc_auc"][0] for r in rows]
        hi = [summary["test_ci95"][f"{r[1]} ({ds})"]["roc_auc"][1] for r in rows]
        ax.bar(x + off, v, width=0.34, color=col, label=ds, edgecolor="white")
        ax.errorbar(x + off, v, yerr=[np.array(v) - lo, np.array(hi) - v], fmt="none", ecolor=INK, lw=1, capsize=3)
    ax.axhline(0.5, ls="--", color=INK2, lw=0.8)
    ax.set_xticks(x, [r[0] for r in rows]); ax.set_ylim(0.4, 0.6); ax.set_ylabel("Test ROC-AUC (95% CI)")
    ax.set_title("Base vs Full features: test ROC-AUC by model", loc="left"); ax.legend(frameon=False)
    ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
    fig.savefig(P / "09_base_vs_full.png"); plt.close(fig)


def paired_plot():
    d = pd.read_csv(M / "paired_bootstrap_auc.csv")
    d["label"] = d.stage2_model + "  vs  " + d.vs
    d = d.iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 8))
    y = np.arange(len(d))
    lo = d.ci95.apply(lambda s: json.loads(s)[0] if isinstance(s, str) else s[0])
    hi = d.ci95.apply(lambda s: json.loads(s)[1] if isinstance(s, str) else s[1])
    ax.hlines(y, lo, hi, color=INK2, lw=1.5)
    ax.plot(d.auc_diff, y, "o", color="#2a78d6", ms=7)
    ax.axvline(0, color=INK, lw=0.8)
    ax.set_yticks(y, d.label, fontsize=8)
    ax.set_xlabel("Difference in test ROC-AUC (first minus second), paired bootstrap 95% CI")
    ax.set_title("Paired AUC comparisons on the same 946 test events", loc="left")
    ax.grid(axis="x", color=GRID, lw=0.6); ax.set_axisbelow(True)
    fig.savefig(P / "10_paired_auc_differences.png"); plt.close(fig)


def main():
    P.mkdir(parents=True, exist_ok=True)
    summary = json.load(open(M / "stage2_summary.json"))
    tab = pd.read_csv(M / "comparison_test.csv")
    for k in ["accuracy", "f1_weighted", "mcc", "roc_auc"]:
        tab[f"{k}_ci"] = tab[f"{k}_ci"].apply(lambda s: json.loads(s) if isinstance(s, str) else np.nan)
    metric_comparison(tab)
    test = load("base")["test"]
    y = test.y.values
    preds = load_stage1(test.event_id.values, y)
    for n, tag in [("Stage 2 LR (Base)", "lr_base"), ("Stage 2 LR (Full)", "lr_full"),
                   ("Stage 2 RF (Base)", "rf_base"), ("Stage 2 RF (Full)", "rf_full")]:
        preds[n] = pd.read_csv(RES / "predictions" / f"pred_test_{tag}.csv").set_index("event_id").loc[test.event_id].p_long.values
    roc_plot(y, preds)
    confusion_plot(summary)
    coef_plot()
    corr_plot()
    base_vs_full(summary)
    paired_plot()
    print("plots:", sorted(p.name for p in P.glob("*.png")))


if __name__ == "__main__":
    main()
