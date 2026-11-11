"""
Offline tests: every processing function is run on data in the exact formats returned by
SEC EDGAR / Yahoo Finance (yfinance) / Hugging Face, with the network calls replaced by fakes.
Run:  python tests/test_offline.py
"""
import json
import os
import sys
import tempfile

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
os.environ["STAGE1_DATA_DIR"] = tempfile.mkdtemp(prefix="stage1_test_")

import config  # noqa: E402
import step01_companies_and_transcripts as s1  # noqa: E402
import step02_financial_growth as s2  # noqa: E402
import step03_eps_and_analyst as s3  # noqa: E402
import step04_market_and_labels as s4  # noqa: E402
import step05_build_base_full as s5  # noqa: E402
from utils import save_df, load_df  # noqa: E402

PASS = []


def check(cond, msg):
    assert cond, "FAILED: " + msg
    PASS.append(msg)


# ---------------------------------------------------------------- SEC --
# Real Apple (CIK 320193) facts copied from data.sec.gov/api/xbrl/companyfacts/CIK0000320193.json
def F(start, end, val, filed, form="10-Q", fy=2023, fp="Q3"):
    d = {"end": end, "val": val, "accn": "x", "fy": fy, "fp": fp, "form": form, "filed": filed}
    if start:
        d["start"] = start
    return d


AAPL_RAW = {"cik": 320193, "entityName": "Apple Inc.", "facts": {"dei": {}, "us-gaap": {
    "RevenueFromContractWithCustomerExcludingAssessedTax": {"units": {"USD": [
        F("2021-09-26", "2022-06-25", 304182000000, "2022-07-29"),
        F("2022-03-27", "2022-06-25", 82959000000, "2022-07-29"),
        F("2021-09-26", "2022-09-24", 394328000000, "2022-10-28", "10-K"),
        F("2022-09-25", "2023-07-01", 293787000000, "2023-08-04"),
        F("2022-04-02", "2023-07-01", 1, "2023-08-04", "8-K"),                 # wrong form -> ignored
        F("2023-04-02", "2023-07-01", 81797000000, "2023-08-04"),
        F("2023-04-02", "2023-07-01", 81797000000, "2024-08-02"),
        F("2022-09-25", "2023-09-30", 383285000000, "2023-11-03", "10-K"),
    ]}},
    "NetIncomeLoss": {"units": {"USD": [
        F("2022-03-27", "2022-06-25", 19442000000, "2022-07-29"),
        F("2021-09-26", "2022-06-25", 79082000000, "2022-07-29"),
        F("2021-09-26", "2022-09-24", 99803000000, "2022-10-28", "10-K"),
        F("2023-04-02", "2023-07-01", 19881000000, "2023-08-04"),
        F("2022-09-25", "2023-07-01", 74039000000, "2023-08-04"),
        F("2022-09-25", "2023-09-30", 96995000000, "2023-11-03", "10-K"),
    ]}},
    "EarningsPerShareDiluted": {"units": {"USD/shares": [
        F("2022-03-27", "2022-06-25", 1.20, "2022-07-29"),
        F("2023-04-02", "2023-07-01", 1.26, "2023-08-04"),
        F("2022-09-25", "2023-09-30", 6.13, "2023-11-03", "10-K"),
    ]}},
    "NetCashProvidedByUsedInOperatingActivities": {"units": {"USD": [   # cash flow: YTD only
        F("2021-09-26", "2022-03-26", 75000000000, "2022-04-29"),
        F("2021-09-26", "2022-06-25", 98000000000, "2022-07-29"),
        F("2022-09-25", "2023-04-01", 62565000000, "2023-05-05"),
        F("2022-09-25", "2023-07-01", 88945000000, "2023-08-04"),
    ]}},
    "Assets": {"units": {"USD": [
        F(None, "2022-06-25", 336309000000, "2022-07-29"),
        F(None, "2023-07-01", 335038000000, "2023-08-04"),
    ]}},
}}}


def test_sec():
    facts = s2.compact_companyfacts(AAPL_RAW)
    check(set(facts) == {"RevenueFromContractWithCustomerExcludingAssessedTax", "NetIncomeLoss",
                         "EarningsPerShareDiluted", "NetCashProvidedByUsedInOperatingActivities", "Assets"},
          "SEC: compact keeps only needed tags")
    check(all(r["form"] != "8-K" for r in facts["RevenueFromContractWithCustomerExcludingAssessedTax"]),
          "SEC: non 10-Q/10-K forms dropped")
    frames = {t: s2.facts_frame(r) for t, r in facts.items()}
    rev = frames["RevenueFromContractWithCustomerExcludingAssessedTax"]

    check(s2.quarter_value(rev, pd.Timestamp("2023-07-01"), instant=False) == 81797000000,
          "SEC: direct 3-month quarter value (AAPL FQ3-2023 revenue 81.797B)")
    q4 = s2.quarter_value(rev, pd.Timestamp("2023-09-30"), instant=False)
    check(q4 == 383285000000 - 293787000000, "SEC: Q4 derived as FY - 9M YTD (AAPL FQ4-2023 = 89.498B)")
    ocf = frames["NetCashProvidedByUsedInOperatingActivities"]
    check(s2.quarter_value(ocf, pd.Timestamp("2023-07-01"), instant=False) == 88945000000 - 62565000000,
          "SEC: cash-flow quarter derived from YTD difference")
    check(s2.quarter_value(frames["EarningsPerShareDiluted"], pd.Timestamp("2023-09-30"), False, additive=False)
          is None, "SEC: EPS never derived by subtraction")

    # Apple reported FQ3-2023 on 2023-08-03
    g = s2.event_growth(frames, pd.Timestamp("2023-08-03"))
    check(g["fin_period_end"] == pd.Timestamp("2023-07-01"), "SEC: reported quarter = latest period end before call")
    check(abs(g["growth_Revenue"] - (81797 / 82959 - 1)) < 1e-9, "SEC: revenue YoY growth (-1.4%)")
    check(abs(g["growth_Net Income"] - (19881 / 19442 - 1)) < 1e-9, "SEC: net income YoY growth (+2.3%)")
    check(abs(g["growth_EPS (diluted)"] - (1.26 / 1.20 - 1)) < 1e-9, "SEC: EPS YoY growth")
    check(abs(g["growth_Total Assets"] - (335038 / 336309 - 1)) < 1e-9, "SEC: balance-sheet YoY growth")
    exp_ocf = (88945000000 - 62565000000) / (98000000000 - 75000000000) - 1
    check(abs(g["growth_Operating Cash Flow"] - exp_ocf) < 1e-9, "SEC: cash-flow YoY growth from YTD values")
    check(pd.isna(g.get("growth_Gross Profit")) or g.get("growth_Gross Profit") is None,
          "SEC: missing tag -> missing metric, no crash")
    check(s2.event_growth({}, pd.Timestamp("2023-08-03")) == {"fin_period_end": None}, "SEC: no facts -> empty")
    return facts


# --------------------------------------------------------------- Yahoo --
def yahoo_earnings_frame():
    """Same shape as yfinance Ticker.get_earnings_dates(): tz-aware index, string-ish numbers."""
    idx = pd.DatetimeIndex(["2023-11-02 16:00", "2023-08-03 16:00", "2023-05-04 16:00", "2023-02-02 16:00"],
                           name="Earnings Date").tz_localize("America/New_York")
    return pd.DataFrame({"EPS Estimate": [1.39, 1.19, 1.43, 1.94],
                         "Reported EPS": [1.46, 1.26, 1.52, 1.88],
                         "Surprise(%)": ["+5.04", "+5.88", "+6.29", "-3.09"]}, index=idx)


def yahoo_upgrades_frame():
    """Same shape as yfinance Ticker.upgrades_downgrades."""
    idx = pd.DatetimeIndex(["2023-07-05", "2023-07-12", "2023-07-20", "2023-07-28", "2023-08-03", "2023-05-01"],
                           name="GradeDate")
    return pd.DataFrame({"Firm": ["A", "B", "C", "D", "E", "F"],
                         "ToGrade": ["Buy", "Overweight", "Neutral", "Some Odd Grade", "Sell", "Sell"],
                         "FromGrade": [""] * 6, "Action": ["main"] * 6}, index=idx)


def test_yahoo():
    eps = s3.parse_earnings_dates(yahoo_earnings_frame())
    check(list(eps.columns) == ["earnings_date", "eps_estimate", "eps_actual", "yahoo_surprise_pct"],
          "Yahoo: earnings frame parsed")
    check(eps["yahoo_surprise_pct"].iloc[0] == 5.04, "Yahoo: '+5.04' string parsed to number")
    m = s3.match_eps(eps, pd.Timestamp("2023-08-03"))
    check(abs(m["eps_surprise"] - (1.26 - 1.19) / 1.19) < 1e-9, "Yahoo: EPS surprise = (actual-est)/|est|")
    check(pd.isna(s3.match_eps(eps, pd.Timestamp("2023-09-15"))["eps_surprise"]),
          "Yahoo: no earnings date within tolerance -> missing")
    check(s3.match_eps(eps, pd.Timestamp("2023-08-05"))["eps_actual"] == 1.26, "Yahoo: +-3 day date tolerance")

    ud = s3.parse_upgrades(yahoo_upgrades_frame())
    g = s3.majority_grade(ud, pd.Timestamp("2023-08-03"))
    # window = [Jul 4, Aug 3): Buy, Overweight, Neutral, (odd ignored)  -> buy (2 vs 1); Aug-3 Sell excluded
    check(g == {"analyst_grade": "buy", "n_analyst_grades": 3}, "Yahoo: 30-day majority grade, no look-ahead")
    check(s3.majority_grade(ud, pd.Timestamp("2022-01-01"))["analyst_grade"] is None, "Yahoo: no grades -> None")
    check(s3.parse_earnings_dates(None).empty and s3.parse_upgrades(None).empty, "Yahoo: None responses handled")


# -------------------------------------------------------------- prices --
def fake_prices(symbols, start="2019-11-01", end="2024-08-01", seed=0):
    rng = np.random.default_rng(seed)
    days = pd.bdate_range(start, end)
    rows = []
    for s in symbols:
        p = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, len(days))))
        o = p * (1 + rng.normal(0, 0.005, len(days)))
        rows.append(pd.DataFrame({"symbol": s, "date": days, "open": o, "close": p, "adj_close": p}))
    return pd.concat(rows, ignore_index=True)


def test_market_rules():
    d = pd.DatetimeIndex(pd.bdate_range("2023-07-24", "2023-08-11"))
    # after-close call (Thu 16:30) -> label Fri, info Thu
    lab, info = s4.label_and_info_day(d, pd.Timestamp("2023-08-03 16:30"))
    check((lab, info) == (pd.Timestamp("2023-08-04"), pd.Timestamp("2023-08-03")), "Label: after-close call -> next day")
    lab, info = s4.label_and_info_day(d, pd.Timestamp("2023-08-03 08:30"))
    check((lab, info) == (pd.Timestamp("2023-08-03"), pd.Timestamp("2023-08-02")), "Label: pre-market call -> same day")
    lab, info = s4.label_and_info_day(d, pd.Timestamp("2023-08-04 17:00"))   # Friday -> Monday
    check(lab == pd.Timestamp("2023-08-07"), "Label: Friday call -> Monday")
    lab, _ = s4.label_and_info_day(d, pd.Timestamp("2023-08-03 00:00"))
    check(lab == pd.Timestamp("2023-08-04"), "Label: unknown time -> next trading day")

    # yfinance.download(group_by='ticker') column layout
    idx = pd.DatetimeIndex(pd.bdate_range("2023-01-02", periods=3), name="Date")
    cols = pd.MultiIndex.from_product([["AAPL", "^DJI"], ["Open", "High", "Low", "Close", "Adj Close", "Volume"]],
                                      names=["Ticker", "Price"])
    data = pd.DataFrame(np.arange(36, dtype=float).reshape(3, 12) + 1, index=idx, columns=cols)
    sub = s4._extract(data, "^DJI")
    check(list(sub.columns) == ["Open", "Close", "Adj Close"] and len(sub) == 3, "Prices: yf.download MultiIndex parsed")
    flat = data["AAPL"]
    check(s4._extract(flat, "AAPL") is not None, "Prices: single-ticker flat columns parsed")
    check(s4._extract(data, "MSFT") is None, "Prices: missing ticker -> None")


# ----------------------------------------------------------- transcripts --
def test_transcripts():
    batch = pd.DataFrame({
        "symbol": ["AAPL", "AAPL", "AAPL", "BRK-B", "XYZ", "AAPL"],
        "company_name": ["Apple Inc."] * 3 + ["Berkshire", "X", "Apple Inc."],
        "year": [2023, 2023, 2023, 2023, 2023, 2018], "quarter": [3, 3, 2, 2, 1, 1],
        "date": ["2023-08-03 17:00:00", "2023-08-03 17:00:00", "2023-05-04 17:00:00",
                 "2023-08-05 10:00:00", "2023-08-01 10:00:00", "2018-02-01 17:00:00"],
        "content": ["Operator: " + "a" * 5000, "Operator: " + "b" * 3000, json.dumps("Operator:\n" + "c" * 4000),
                    "Operator: " + "d" * 4000, "Operator: " + "e" * 4000, "Operator: " + "f" * 4000],
    })
    kept = s1.filter_transcripts(batch, {"AAPL", "BRK.B"})
    check(len(kept) == 4, "Transcripts: filter by company + period")
    check(kept["transcript"].iloc[2].startswith("Operator:\n"), "Transcripts: JSON-encoded text decoded")
    ev = s1.finalize_events(kept)
    check(len(ev) == 3 and ev[ev["ticker"] == "AAPL"]["transcript"].str.len().max() == 5010,
          "Transcripts: duplicate call -> keep longest")
    check(set(ev["ticker"]) == {"AAPL", "BRK.B"}, "Transcripts: BRK-B normalised to BRK.B")


# ------------------------------------------------------------ end-to-end --
def test_end_to_end(aapl_facts):
    rng = np.random.default_rng(1)
    tickers = ["AAPL", "MSFT", "JPM"]
    companies = pd.DataFrame({"ticker": tickers, "name": ["Apple", "Microsoft", "JPMorgan"],
                              "sector": ["IT", "IT", "Fin"], "cik": [320193, 789019, 19617]})
    companies.to_csv(f"{config.PROCESSED_DIR}/companies.csv", index=False)

    # 4 calls per year per company, 2020-2024H1
    rows = []
    for t in tickers:
        for d in pd.date_range("2020-02-01", "2024-06-01", freq="QS-FEB"):
            dt = d + pd.Timedelta(days=int(rng.integers(0, 5)), hours=16, minutes=30)
            rows.append({"ticker": t, "company_name": t + " Inc.", "year": dt.year, "quarter": 1,
                         "call_datetime": dt, "transcript": f"Operator: welcome to {t}. " + "x " * 2000})
    events = s1.finalize_events(pd.DataFrame(rows))
    save_df(events, "events")

    # synthetic but realistic SEC facts: quarterly 3M revenue & net income, instants, YTD cash flow
    def synth_facts(seed):
        r = np.random.default_rng(seed)
        out = {"Revenues": [], "NetIncomeLoss": [], "Assets": [], "EarningsPerShareDiluted": []}
        for qe in pd.date_range("2018-12-31", "2024-06-30", freq="QE"):
            st = (qe - pd.offsets.QuarterBegin(startingMonth=1)).strftime("%Y-%m-%d")
            e = qe.strftime("%Y-%m-%d")
            filed = (qe + pd.Timedelta(days=35)).strftime("%Y-%m-%d")
            out["Revenues"].append({"start": st, "end": e, "val": float(r.uniform(1e9, 2e9)), "filed": filed, "form": "10-Q"})
            out["NetIncomeLoss"].append({"start": st, "end": e, "val": float(r.uniform(-1e8, 3e8)), "filed": filed, "form": "10-Q"})
            out["EarningsPerShareDiluted"].append({"start": st, "end": e, "val": float(r.uniform(0.5, 2)), "filed": filed, "form": "10-Q"})
            out["Assets"].append({"start": None, "end": e, "val": float(r.uniform(5e9, 6e9)), "filed": filed, "form": "10-Q"})
        return out

    fake_sec = {320193: synth_facts(1), 789019: synth_facts(2), 19617: synth_facts(3)}
    fin = s2.build_financials(events, companies, fetch=lambda cik, refresh=False: fake_sec.get(cik))
    save_df(fin, "financials")
    check((fin["n_fin_metrics"] >= 3).mean() > 0.9, "E2E: financial growth computed for most events")

    def fake_yahoo(t, refresh=False):
        ev = events[events["ticker"] == t]
        idx = pd.DatetimeIndex(ev["call_date"] + pd.Timedelta(hours=16)).tz_localize("America/New_York")
        est = rng.uniform(0.5, 2, len(idx))
        eps_raw = pd.DataFrame({"EPS Estimate": est, "Reported EPS": est * rng.uniform(0.9, 1.15, len(idx)),
                                "Surprise(%)": ["+1.0"] * len(idx)}, index=idx.rename("Earnings Date"))
        gd = pd.DatetimeIndex(ev["call_date"] - pd.Timedelta(days=10)).rename("GradeDate")
        ud_raw = pd.DataFrame({"Firm": "X", "ToGrade": rng.choice(["Buy", "Hold", "Sell", "Outperform"], len(gd)),
                               "FromGrade": "", "Action": "main"}, index=gd)
        return s3.parse_earnings_dates(eps_raw), s3.parse_upgrades(ud_raw)

    ea = s3.build_eps_analyst(events, fetch=fake_yahoo)
    save_df(ea, "eps_analyst")
    check(ea["eps_surprise"].notna().all() and ea["analyst_grade"].notna().all(), "E2E: EPS + grades matched")

    prices = fake_prices(tickers + list(config.MARKET_INDEX_TICKERS))
    save_df(prices, "prices", config.RAW_DIR)
    mkt = s4.build_market(events, prices)
    save_df(mkt, "market")
    check(mkt["label"].isin(["Long", "Short"]).all(), "E2E: every event labelled")
    r = mkt.iloc[5]
    px = prices[prices["symbol"] == events.iloc[5]["ticker"]].set_index("date")
    check(r["label"] == ("Long" if px.at[r["label_day"], "open"] < px.at[r["label_day"], "close"] else "Short"),
          "E2E: label = Long iff next-day open < close")
    check(r["label_day"] > events.iloc[5]["call_date"] and r["info_day"] < r["label_day"], "E2E: no look-ahead in dates")

    s5.main()
    base = [json.loads(line) for line in open(f"{config.FINAL_DIR}/base.jsonl")]
    full = [json.loads(line) for line in open(f"{config.FINAL_DIR}/full.jsonl")]
    check(len(base) == len(full) > 0, "E2E: Base and Full have the same rows")
    check([b["event_id"] for b in base] == [f["event_id"] for f in full], "E2E: same events in same order")
    check("Market performance" not in base[0]["input"] and "Market performance" in full[0]["input"],
          "E2E: Base excludes / Full includes market info")
    check("Compared to the same quarter last year" in base[0]["input"], "E2E: financial growth textualised")
    check("In the past week, SPY went" in full[0]["input"], "E2E: market textualised (paper wording)")
    check("most grading companies suggest" in full[0]["input"] or "no analyst grades" in full[0]["input"],
          "E2E: analyst grade textualised (paper wording)")
    check("consensus estimates" in full[0]["input"], "E2E: EPS surprise textualised (paper wording)")
    check({b["output"] for b in base} <= {"Long", "Short"}, "E2E: outputs are Long/Short")
    check({b["split"] for b in base} == {"train", "val", "test"}, "E2E: chronological split present")
    stats = json.load(open(f"{config.FINAL_DIR}/stats.json"))
    check(stats["final_rows"] == len(base), "E2E: stats.json consistent")
    print("\nSample FULL input (first 1200 chars):\n" + full[0]["input"][:1200])


if __name__ == "__main__":
    facts = test_sec()
    test_yahoo()
    test_market_rules()
    test_transcripts()
    test_end_to_end(facts)
    print(f"\nALL {len(PASS)} CHECKS PASSED  (temp data dir: {config.DATA_DIR})")
