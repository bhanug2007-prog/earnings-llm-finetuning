"""
STEP 3 - Earnings surprise (EPS) + analyst grades for every earnings event

Paper (Full dataset):
  * Earnings Surprises: actual vs. estimated EPS
      "The company's reported earnings per share were 7.69% higher than analysts' consensus estimates"
  * Analyst Grades: upgrades/downgrades before the earnings date, most frequent grade
      "In the past 30 days, most grading companies suggest buying this stock"
Here : FMP's earnings-surprises and grades endpoints are paid-only, so we use Yahoo Finance via
       yfinance:  Ticker.get_earnings_dates(limit=100)  (EPS estimate / reported EPS, back to ~2000)
                  Ticker.upgrades_downgrades           (dated grade changes by each firm)

Output: data/processed/eps_analyst.parquet
"""
import os
import time

import numpy as np
import pandas as pd

import config
from utils import cli_args, load_df, log, save_df, with_retries, yahoo_ticker

# ----------------------------------------------------------- grade mapping --
BUY = {"buy", "strong buy", "outperform", "overweight", "market outperform", "sector outperform",
       "positive", "accumulate", "add", "top pick", "conviction buy", "long-term buy", "long term buy",
       "speculative buy", "moderate buy", "outperformer", "above average", "sector overweight",
       "market overweight", "buy (high risk)", "strong-buy", "industry outperform", "trading buy"}
HOLD = {"hold", "neutral", "equal-weight", "equal weight", "market perform", "sector perform", "peer perform",
        "in-line", "in line", "sector weight", "perform", "mixed", "market weight", "fair value",
        "industry perform", "average", "equal-weighted", "sector neutral", "neutral weight"}
SELL = {"sell", "strong sell", "underperform", "underweight", "reduce", "negative", "market underperform",
        "sector underperform", "industry underperform", "below average", "cautious", "sector underweight",
        "strong-sell", "moderate sell", "underperformer"}


def grade_category(g):
    if g is None or (isinstance(g, float) and np.isnan(g)):
        return None
    g = str(g).strip().lower()
    if g in BUY:
        return "buy"
    if g in HOLD:
        return "hold"
    if g in SELL:
        return "sell"
    return None


# ------------------------------------------------------------ yahoo fetch --
def _cache(kind, ticker):
    d = os.path.join(config.RAW_DIR, f"yahoo_{kind}")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f"{ticker}.csv")


def _num(s):
    return pd.to_numeric(s.astype(str).str.replace(r"[+,%]", "", regex=True).str.strip(), errors="coerce")


def parse_earnings_dates(df: pd.DataFrame) -> pd.DataFrame:
    """yfinance get_earnings_dates() frame -> earnings_date, eps_estimate, eps_actual, yahoo_surprise_pct."""
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=["earnings_date", "eps_estimate", "eps_actual", "yahoo_surprise_pct"])
    d = df.copy()
    idx = pd.DatetimeIndex(pd.to_datetime(d.index))
    if idx.tz is not None:
        idx = idx.tz_convert("America/New_York").tz_localize(None)
    out = pd.DataFrame({
        "earnings_date": idx.normalize(),
        "eps_estimate": _num(d["EPS Estimate"]).values if "EPS Estimate" in d else np.nan,
        "eps_actual": _num(d["Reported EPS"]).values if "Reported EPS" in d else np.nan,
        "yahoo_surprise_pct": _num(d["Surprise(%)"]).values if "Surprise(%)" in d else np.nan,
    })
    return out.dropna(subset=["earnings_date"]).reset_index(drop=True)


def parse_upgrades(df: pd.DataFrame) -> pd.DataFrame:
    """yfinance upgrades_downgrades frame -> grade_date, firm, to_grade, action."""
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=["grade_date", "firm", "to_grade", "action"])
    d = df.reset_index()
    date_col = "GradeDate" if "GradeDate" in d else d.columns[0]
    out = pd.DataFrame({
        "grade_date": pd.to_datetime(d[date_col], errors="coerce"),
        "firm": d.get("Firm"),
        "to_grade": d.get("ToGrade"),
        "action": d.get("Action"),
    })
    if getattr(out["grade_date"].dt, "tz", None) is not None:
        out["grade_date"] = out["grade_date"].dt.tz_convert("America/New_York").dt.tz_localize(None)
    return out.dropna(subset=["grade_date"]).reset_index(drop=True)


def fetch_yahoo(ticker: str, refresh=False):
    """Returns (earnings_df, upgrades_df). Cached per ticker in data/raw/."""
    import yfinance as yf
    p_eps, p_ud = _cache("eps", ticker), _cache("grades", ticker)
    if os.path.exists(p_eps) and os.path.exists(p_ud) and not refresh:
        eps = pd.read_csv(p_eps, parse_dates=["earnings_date"])
        ud = pd.read_csv(p_ud, parse_dates=["grade_date"])
        return eps, ud

    tk = yf.Ticker(yahoo_ticker(ticker))
    raw_eps = with_retries(lambda: tk.get_earnings_dates(limit=100), what=f"{ticker} earnings dates")
    time.sleep(config.YAHOO_SLEEP_SECONDS)
    raw_ud = with_retries(lambda: tk.upgrades_downgrades, what=f"{ticker} upgrades/downgrades", retries=2)
    time.sleep(config.YAHOO_SLEEP_SECONDS)

    eps, ud = parse_earnings_dates(raw_eps), parse_upgrades(raw_ud)
    if raw_eps is not None:                 # only cache successful downloads
        eps.to_csv(p_eps, index=False)
        ud.to_csv(p_ud, index=False)
    return eps, ud


# ------------------------------------------------------------- per event --
def match_eps(eps: pd.DataFrame, call_date: pd.Timestamp) -> dict:
    out = {"eps_actual": np.nan, "eps_estimate": np.nan, "eps_surprise": np.nan, "eps_date": pd.NaT}
    if eps is None or eps.empty:
        return out
    e = eps.dropna(subset=["eps_actual"]).copy()
    e["gap"] = (e["earnings_date"] - call_date).abs()
    e = e[e["gap"] <= pd.Timedelta(days=config.EPS_DATE_TOLERANCE_DAYS)].sort_values("gap")
    if e.empty:
        return out
    r = e.iloc[0]
    act, est = r["eps_actual"], r["eps_estimate"]
    if pd.notna(est) and est != 0:
        surprise = (act - est) / abs(est)
    elif pd.notna(r["yahoo_surprise_pct"]):
        surprise = r["yahoo_surprise_pct"] / 100.0
    else:
        surprise = np.nan
    out.update(eps_actual=act, eps_estimate=est, eps_surprise=surprise, eps_date=r["earnings_date"])
    return out


def majority_grade(ud: pd.DataFrame, call_date: pd.Timestamp) -> dict:
    out = {"analyst_grade": None, "n_analyst_grades": 0}
    if ud is None or ud.empty:
        return out
    w = ud[(ud["grade_date"] < call_date) &
           (ud["grade_date"] >= call_date - pd.Timedelta(days=config.ANALYST_WINDOW_DAYS))].copy()
    w["cat"] = w["to_grade"].map(grade_category)
    w = w.dropna(subset=["cat"])
    if w.empty:
        return out
    counts = w["cat"].value_counts()
    top = counts[counts == counts.max()].index.tolist()
    if len(top) == 1:
        grade = top[0]
    else:                                       # tie -> most recent grade among the tied ones
        grade = w[w["cat"].isin(top)].sort_values("grade_date").iloc[-1]["cat"]
    out.update(analyst_grade=grade, n_analyst_grades=int(len(w)))
    return out


def build_eps_analyst(events: pd.DataFrame, fetch=fetch_yahoo, refresh=False) -> pd.DataFrame:
    rows, tickers = [], sorted(events["ticker"].unique())
    unknown_grades = {}
    for i, t in enumerate(tickers, 1):
        eps, ud = fetch(t, refresh)
        if ud is not None and len(ud):
            for g in ud["to_grade"].dropna().unique():
                if grade_category(g) is None:
                    unknown_grades[g] = unknown_grades.get(g, 0) + 1
        for _, e in events[events["ticker"] == t].iterrows():
            rows.append({"event_id": e["event_id"], **match_eps(eps, e["call_date"]),
                         **majority_grade(ud, e["call_date"])})
        if i % 25 == 0 or i == len(tickers):
            log.info(f"  yahoo: {i}/{len(tickers)} companies")
    if unknown_grades:
        log.info(f"grade labels not mapped (ignored): {sorted(unknown_grades)[:30]}")
    return pd.DataFrame(rows)


def main():
    args = cli_args("Step 3: EPS surprise + analyst grades (Yahoo Finance)")
    events = load_df("events")
    df = build_eps_analyst(events, refresh=args.refresh)
    save_df(df, "eps_analyst")
    log.info(f"EPS surprise found: {df['eps_surprise'].notna().mean():.1%} | "
             f"analyst grade found: {df['analyst_grade'].notna().mean():.1%}")


if __name__ == "__main__":
    main()
