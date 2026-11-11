"""
STEP 2 - Financial-metric growth (year-over-year) for every earnings event

Paper: "Financial Metrics Growth: ... contrasting the quarter ... with the same quarter from the
        previous year" - income statement, balance sheet and cash-flow metrics (FMP financial-growth).
        Textualised as "Compared to the same quarter last year, Net Income grew by 4.6%".
Here : FMP's growth endpoint is not on the free plan, so we compute the same YoY growth from the
       official SEC EDGAR XBRL "companyfacts" API (free, the source FMP itself builds on).

How a quarter value is obtained from XBRL:
  * reported 3-month value for the quarter            -> used directly
  * otherwise (Q4 only in the 10-K, cash-flow items only year-to-date in 10-Qs):
    quarter = YTD value ending at the quarter - YTD value (same start) ending one quarter earlier
  * balance-sheet items are point-in-time values at the quarter end
  * the earliest-filed value is used (= the originally reported number, not a later restatement)

Output: data/processed/financials.parquet  (event_id + growth_* columns)
"""
import json
import os
import time

import numpy as np
import pandas as pd
import requests

import config
from utils import cli_args, load_df, log, save_df, with_retries

SEC_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"
FORMS = {"10-Q", "10-K", "10-Q/A", "10-K/A", "10-KT", "20-F", "40-F"}
ALL_TAGS = sorted({t for tags in list(config.DURATION_METRICS.values()) + list(config.INSTANT_METRICS.values())
                   for t in tags})


# --------------------------------------------------------------- download --
def compact_companyfacts(raw: dict) -> dict:
    """Keep only the tags we need -> {tag: [{start,end,val,filed,form}, ...]} (pure, unit-tested)."""
    gaap = raw.get("facts", {}).get("us-gaap", {})
    out = {}
    for tag in ALL_TAGS:
        if tag not in gaap:
            continue
        units = gaap[tag].get("units", {})
        unit = "USD" if "USD" in units else ("USD/shares" if "USD/shares" in units else next(iter(units), None))
        if unit is None:
            continue
        rows = []
        for f in units[unit]:
            if f.get("form") not in FORMS or f.get("val") is None or not f.get("end"):
                continue
            rows.append({"start": f.get("start"), "end": f["end"], "val": f["val"],
                         "filed": f.get("filed"), "form": f.get("form")})
        if rows:
            out[tag] = rows
    return out


def get_companyfacts(cik: int, refresh=False) -> dict:
    cache_dir = os.path.join(config.RAW_DIR, "sec_companyfacts")
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"{int(cik):010d}.json")
    if os.path.exists(path) and not refresh:
        with open(path) as fh:
            return json.load(fh)

    def _get():
        r = requests.get(SEC_URL.format(cik=int(cik)), headers={"User-Agent": config.SEC_USER_AGENT,
                                                                 "Accept-Encoding": "gzip, deflate"}, timeout=60)
        if r.status_code == 404:
            return {}
        r.raise_for_status()
        return r.json()

    raw = with_retries(_get, what=f"SEC companyfacts CIK {cik}")
    time.sleep(config.SEC_SLEEP_SECONDS)
    if raw is None:
        return None                          # network failure -> not cached, retried next run
    facts = compact_companyfacts(raw)
    with open(path, "w") as fh:
        json.dump(facts, fh)
    return facts


# ------------------------------------------------------------- processing --
def facts_frame(rows) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["end"] = pd.to_datetime(df["end"], errors="coerce")
    df["start"] = pd.to_datetime(df["start"], errors="coerce") if "start" in df else pd.NaT
    df["filed"] = pd.to_datetime(df["filed"], errors="coerce")
    df["val"] = pd.to_numeric(df["val"], errors="coerce")
    df = df.dropna(subset=["end", "val"])
    df["days"] = (df["end"] - df["start"]).dt.days
    return df.sort_values("filed")


def _first(df):
    return None if df.empty else float(df.iloc[0]["val"])     # earliest filed


def quarter_value(df: pd.DataFrame, end: pd.Timestamp, instant: bool, additive: bool = True):
    if df is None or df.empty:
        return None
    same = df[df["end"] == end]
    if same.empty:
        return None
    if instant:
        return _first(same)
    direct = same[(same["days"] >= 80) & (same["days"] <= 100)]
    if not direct.empty:
        return _first(direct)
    if not additive:
        return None
    # derive: YTD(end) - YTD(previous quarter end) with the same start date
    for _, row in same[same["days"] > 100].iterrows():
        prev = df[(df["start"] == row["start"]) &
                  (df["end"] >= end - pd.Timedelta(days=100)) & (df["end"] <= end - pd.Timedelta(days=75))]
        if not prev.empty:
            return float(row["val"]) - float(prev.iloc[0]["val"])
    return None


def prior_year_end(df: pd.DataFrame, end: pd.Timestamp):
    """The period end ~1 year before `end` (52/53-week fiscal years shift by a few days)."""
    if df is None or df.empty:
        return None
    target = end - pd.Timedelta(days=364)
    ends = pd.Series(df["end"].unique())
    diff = (ends - target).abs()
    if diff.empty or diff.min() > pd.Timedelta(days=10):
        return None
    return ends[diff.idxmin()]


def metric_values(frames: dict, tags: list, end, instant: bool, additive: bool):
    """(current, prior-year) quarter values; prefer one tag for both, else combine tags."""
    pairs = []
    for tag in tags:
        df = frames.get(tag)
        if df is None or df.empty:
            continue
        cur = quarter_value(df, end, instant, additive)
        pend = prior_year_end(df, end)
        prev = quarter_value(df, pend, instant, additive) if pend is not None else None
        if cur is not None and prev is not None:
            return cur, prev
        pairs.append((cur, prev))
    cur = next((c for c, _ in pairs if c is not None), None)
    prev = next((p for _, p in pairs if p is not None), None)
    return cur, prev


def growth(cur, prev):
    if cur is None or prev is None or prev == 0 or np.isnan(cur) or np.isnan(prev):
        return None
    return (cur - prev) / abs(prev)


def reported_period_end(frames: dict, call_date: pd.Timestamp):
    """Latest fiscal period end in the 120 days before the call = the quarter discussed on the call."""
    ends = []
    for tag in config.DURATION_METRICS["Revenue"] + config.DURATION_METRICS["Net Income"]:
        df = frames.get(tag)
        if df is not None and not df.empty:
            ends.extend(df.loc[(df["end"] < call_date) &
                               (df["end"] >= call_date - pd.Timedelta(days=120)), "end"].tolist())
    return max(ends) if ends else None


def event_growth(frames: dict, call_date: pd.Timestamp) -> dict:
    end = reported_period_end(frames, call_date)
    out = {"fin_period_end": end}
    if end is None:
        return out
    vals = {}
    for name, tags in config.DURATION_METRICS.items():
        vals[name] = metric_values(frames, tags, end, instant=False, additive=name not in config.NON_ADDITIVE)
    for name, tags in config.INSTANT_METRICS.items():
        vals[name] = metric_values(frames, tags, end, instant=True, additive=False)
    ocf, capex = vals.get("Operating Cash Flow", (None, None)), vals.get("Capital Expenditure", (None, None))
    if None not in (ocf[0], ocf[1], capex[0], capex[1]):
        vals["Free Cash Flow"] = (ocf[0] - capex[0], ocf[1] - capex[1])
    for name, (cur, prev) in vals.items():
        out[f"growth_{name}"] = growth(cur, prev)
    return out


def growth_columns():
    return [f"growth_{m}" for m in list(config.DURATION_METRICS) + list(config.INSTANT_METRICS) + config.DERIVED_METRICS]


def build_financials(events: pd.DataFrame, companies: pd.DataFrame, fetch=get_companyfacts, refresh=False):
    cik_of = companies.set_index("ticker")["cik"].to_dict()
    rows = []
    tickers = sorted(events["ticker"].unique())
    for i, t in enumerate(tickers, 1):
        ev = events[events["ticker"] == t]
        cik = cik_of.get(t)
        facts = fetch(int(cik), refresh) if cik is not None and not pd.isna(cik) else None
        frames = {tag: facts_frame(r) for tag, r in (facts or {}).items()}
        for _, e in ev.iterrows():
            rows.append({"event_id": e["event_id"], **event_growth(frames, e["call_date"])})
        if i % 25 == 0 or i == len(tickers):
            log.info(f"  financials: {i}/{len(tickers)} companies")
    df = pd.DataFrame(rows)
    for c in growth_columns():
        if c not in df:
            df[c] = None
    df[growth_columns()] = df[growth_columns()].astype(float)
    df["n_fin_metrics"] = df[growth_columns()].notna().sum(axis=1)
    return df


def main():
    args = cli_args("Step 2: financial-metric growth from SEC EDGAR")
    if "your_email" in config.SEC_USER_AGENT:
        log.warning("Set SEC_USER_AGENT in config.py to your name + e-mail (SEC may block the default).")
    events = load_df("events")
    companies = pd.read_csv(f"{config.PROCESSED_DIR}/companies.csv")
    df = build_financials(events, companies, refresh=args.refresh)
    save_df(df, "financials")
    ok = (df["n_fin_metrics"] >= config.MIN_FIN_METRICS).mean() if len(df) else 0
    log.info(f"events with >= {config.MIN_FIN_METRICS} growth metrics: {ok:.1%}")
    log.info("coverage per metric:\n" + df[growth_columns()].notna().mean().map("{:.1%}".format).to_string())


if __name__ == "__main__":
    main()
