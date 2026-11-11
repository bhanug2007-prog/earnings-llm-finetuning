"""Stage 2 - Phase B: extract numerical features from the Stage 1 prompts.

The Stage 1 dataset stores the numbers only inside the prompt text (base_head / full_head), written by
stage1/2_code/dataset_builder/step05_build_base_full.py. This script parses every line back into numbers
with fixed patterns that mirror step05, and STOPS if any line of a head is not recognised.

Outputs (stage2/data/):
  stage2_base_numeric.csv   event_id, ticker, date, split, label, y + 12 growth metrics (%, NaN = not reported)
  stage2_full_numeric.csv   same + 4 past-week returns (%), eps_surprise (%), analyst_grade (buy/hold/sell/none)
  feature_dictionary.csv    one row per feature
  extraction_checks.json    counts and checks
ticker/date are kept for identification only; they are never model inputs.
"""
import gzip
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "stage1" / "1_data" / "llm_data.jsonl.gz"
OUT = ROOT / "stage2" / "data"

# name in text -> column name (same order as step02 growth_columns)
GROWTH = {
    "Revenue": "g_revenue",
    "Gross Profit": "g_gross_profit",
    "Operating Income": "g_operating_income",
    "Net Income": "g_net_income",
    "EPS (diluted)": "g_eps_diluted",
    "Operating Cash Flow": "g_operating_cash_flow",
    "Capital Expenditure": "g_capital_expenditure",
    "Total Assets": "g_total_assets",
    "Total Liabilities": "g_total_liabilities",
    "Shareholders' Equity": "g_shareholders_equity",
    "Cash and Cash Equivalents": "g_cash_and_equivalents",
    "Free Cash Flow": "g_free_cash_flow",
}
MARKET = {"stock": "ret_stock_1w", "SPY": "ret_spy_1w", "QQQ": "ret_qqq_1w", "DOW": "ret_dow_1w"}
NUM = r"([0-9][0-9,]*(?:\.[0-9]+)?)%"

RE_GROWTH = re.compile(r"^Compared to the same quarter last year, (.+?) (grew by " + NUM + r"|declined by " + NUM
                       + r"|remained unchanged)\.$")
RE_STOCK = re.compile(r"^In the past week, the company's stock \((.+?)\) (went up by " + NUM + r"|went down by "
                      + NUM + r"|was unchanged)\.$")
RE_INDEX = re.compile(r"^In the past week, (SPY|QQQ|DOW) (went up by " + NUM + r"|went down by " + NUM
                      + r"|was unchanged)\.$")
RE_GRADE = re.compile(r"^In the past (\d+) days, (most grading companies suggest (buying|holding|selling) this stock"
                      r"|no analyst grades were issued for this stock)\.$")
RE_EPS = re.compile(r"^The company's reported earnings per share were (" + NUM + r" (higher|lower) than"
                    r"|in line with) analysts' consensus estimates\.$")
SKIP = re.compile(r"^(Company: .+|Earnings call date: \d{4}-\d{2}-\d{2}|Financial metrics growth:|Market performance:"
                  r"|Analyst grades:|Earnings surprise:|Earnings call transcript:|)$")


def _num(s):
    return float(s.replace(",", ""))


def _signed(groups, up_word):
    """groups = (whole phrase, up value, down value) -> signed float."""
    phrase, up, down = groups
    if up is not None:
        return _num(up)
    if down is not None:
        return -_num(down)
    return 0.0  # "remained unchanged" / "was unchanged" (|value| < 0.05% in step05)


def parse_head(text, ticker, full):
    feats, unparsed = {}, []
    for line in text.splitlines():
        line = line.strip()
        if SKIP.match(line):
            continue
        m = RE_GROWTH.match(line)
        if m:
            name = m.group(1)
            if name not in GROWTH:
                unparsed.append(line); continue
            col = GROWTH[name]
            if col in feats:
                raise ValueError(f"duplicate metric {name} for {ticker}")
            feats[col] = _signed((m.group(2), m.group(3), m.group(4)), "grew")
            continue
        if full:
            m = RE_STOCK.match(line)
            if m:
                if m.group(1) != ticker:
                    raise ValueError(f"stock line names {m.group(1)}, expected {ticker}")
                feats["ret_stock_1w"] = _signed((m.group(2), m.group(3), m.group(4)), "up"); continue
            m = RE_INDEX.match(line)
            if m:
                feats[MARKET[m.group(1)]] = _signed((m.group(2), m.group(3), m.group(4)), "up"); continue
            m = RE_GRADE.match(line)
            if m:
                feats["analyst_grade"] = {"buying": "buy", "holding": "hold", "selling": "sell"}.get(m.group(3), "none")
                feats["_grade_window_days"] = int(m.group(1)); continue
            m = RE_EPS.match(line)
            if m:
                feats["eps_surprise"] = 0.0 if m.group(1) == "in line with" else \
                    _num(m.group(2)) * (1 if m.group(3) == "higher" else -1)
                continue
        unparsed.append(line)
    return feats, unparsed


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(l) for l in gzip.open(SRC, "rt", encoding="utf-8")]
    base, full, unparsed = [], [], []
    for r in rows:
        meta = {"event_id": r["event_id"], "ticker": r["ticker"], "date": r["date"], "split": r["split"],
                "label": r["label"], "y": int(r["label"] == "Long")}
        fb, ub = parse_head(r["base_head"], r["ticker"], full=False)
        ff, uf = parse_head(r["full_head"], r["ticker"], full=True)
        unparsed += [(r["event_id"], "base", l) for l in ub] + [(r["event_id"], "full", l) for l in uf]
        # the growth sentences must be identical in both heads
        if {k: v for k, v in ff.items() if k.startswith("g_")} != fb:
            raise ValueError(f"base/full growth mismatch for {r['event_id']}")
        base.append({**meta, **fb})
        full.append({**meta, **ff})
    if unparsed:
        for u in unparsed[:20]:
            print("UNPARSED:", u)
        raise SystemExit(f"{len(unparsed)} unparsed lines - stop")

    gcols = list(GROWTH.values())
    full_cols = gcols + list(MARKET.values()) + ["eps_surprise", "analyst_grade"]
    meta_cols = ["event_id", "ticker", "date", "split", "label", "y"]
    dfb = pd.DataFrame(base).reindex(columns=meta_cols + gcols)
    dff = pd.DataFrame(full)
    windows = set(dff.pop("_grade_window_days"))
    dff = dff.reindex(columns=meta_cols + full_cols)

    # ---- checks
    checks = {"source": str(SRC.relative_to(ROOT)), "events": len(rows),
              "unique_event_ids": int(dfb["event_id"].nunique()), "unparsed_lines": 0,
              "analyst_window_days": sorted(windows)}
    assert len(rows) == 8271 and checks["unique_event_ids"] == 8271
    for c in list(MARKET.values()) + ["eps_surprise", "analyst_grade"]:
        assert dff[c].notna().all(), f"{c} missing in some Full rows"
    assert np.isfinite(dff[gcols + list(MARKET.values()) + ["eps_surprise"]].fillna(0).values).all()
    checks["split_counts"] = dfb.groupby(["split", "label"]).size().unstack().to_dict(orient="index")
    for s in ["train", "val", "test"]:
        d = dfb.loc[dfb.split == s, "date"]
        checks[f"{s}_dates"] = [d.min(), d.max()]
    assert checks["train_dates"][1] < checks["val_dates"][0] and checks["val_dates"][1] < checks["test_dates"][0]
    checks["n_growth_metrics_per_event"] = dfb[gcols].notna().sum(axis=1).describe().round(2).to_dict()

    dfb.to_csv(OUT / "stage2_base_numeric.csv", index=False)
    dff.to_csv(OUT / "stage2_full_numeric.csv", index=False)

    # ---- feature dictionary
    desc = []
    for name, col in GROWTH.items():
        desc.append((col, f"base_head/full_head: 'Compared to the same quarter last year, {name} grew by / declined by x%'",
                     f"Year-over-year growth of {name} vs the same quarter last year, in % (declined = negative, "
                     "unchanged = 0). Sentence omitted = not reported (NaN).", "float (%)", "Base+Full"))
    desc += [
        ("ret_stock_1w", "full_head: 'In the past week, the company's stock (T) went up/down by x%'",
         "Stock return over the 5 trading days ending the last trading day before the label day, in %", "float (%)", "Full"),
        ("ret_spy_1w", "full_head: 'In the past week, SPY went up/down by x%'", "SPY 5-day return, same window, in %",
         "float (%)", "Full"),
        ("ret_qqq_1w", "full_head: 'In the past week, QQQ went up/down by x%'", "QQQ 5-day return, same window, in %",
         "float (%)", "Full"),
        ("ret_dow_1w", "full_head: 'In the past week, DOW went up/down by x%'", "Dow 5-day return, same window, in %",
         "float (%)", "Full"),
        ("eps_surprise", "full_head: 'reported earnings per share were x% higher/lower than / in line with consensus'",
         "EPS surprise vs analysts' consensus, in % (lower = negative, in line = 0)", "float (%)", "Full"),
        ("analyst_grade", "full_head: 'In the past 30 days, most grading companies suggest buying/holding/selling' "
         "or 'no analyst grades were issued'", "Most frequent analyst grade category in the 30 days before the call",
         "category {buy, hold, sell, none}", "Full"),
    ]
    fd = pd.DataFrame(desc, columns=["feature", "source", "description", "dtype", "dataset"])
    fd["missing_pct_all"] = [round(100 * dff[c].isna().mean(), 2) for c in fd.feature]
    for s in ["train", "val", "test"]:
        fd[f"missing_pct_{s}"] = [round(100 * dff.loc[dff.split == s, c].isna().mean(), 2) for c in fd.feature]
    fd.to_csv(OUT / "feature_dictionary.csv", index=False)
    checks["analyst_grade_share"] = dff["analyst_grade"].value_counts(normalize=True).round(4).to_dict()
    json.dump(checks, open(OUT / "extraction_checks.json", "w"), indent=2, default=str)
    print(json.dumps(checks, indent=2, default=str))
    print(fd[["feature", "dataset", "missing_pct_all"]].to_string(index=False))


if __name__ == "__main__":
    main()
