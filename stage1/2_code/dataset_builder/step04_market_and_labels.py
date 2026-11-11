"""
STEP 4 - Market performance (past week) + the target label

Paper (Full dataset): "the performance data for the past week from major market indices (SPY, QQQ,
       and DOW), along with the company's own stock performance"  ->  "In the past week, SPY went down by 1.5%"
Paper (label):        "If the opening price is less than the closing price, we label this as 'Long';
                       otherwise, it is labeled as 'Short'"  (next-day performance)
Here : daily prices from Yahoo Finance (yfinance).  FMP was the paper's source; prices are identical.

Timing rules (no look-ahead):
  label_day = first trading session after the call (pre-market call -> same day)
  info_day  = last trading day BEFORE label_day; past-week return = AdjClose(info_day) / AdjClose(5 trading days earlier) - 1

Output: data/processed/market.parquet
        data/raw/prices.parquet (cached daily prices)
"""
import numpy as np
import pandas as pd

import config
from utils import cli_args, load_df, log, save_df, with_retries, yahoo_ticker


# --------------------------------------------------------------- prices --
def _extract(data: pd.DataFrame, yt: str):
    if data is None or data.empty:
        return None
    if isinstance(data.columns, pd.MultiIndex):
        if yt in data.columns.get_level_values(0):
            sub = data[yt]
        elif yt in data.columns.get_level_values(1):
            sub = data.xs(yt, axis=1, level=1)
        else:
            return None
    else:
        sub = data
    sub = sub.copy()
    need = {"Open", "Close"}
    if not need.issubset(sub.columns):
        return None
    if "Adj Close" not in sub.columns:
        sub["Adj Close"] = sub["Close"]
    sub = sub[["Open", "Close", "Adj Close"]].dropna(subset=["Open", "Close"])
    sub.index = pd.to_datetime(sub.index)
    if sub.index.tz is not None:
        sub.index = sub.index.tz_localize(None)
    return sub


def download_prices(symbols, refresh=False) -> pd.DataFrame:
    """Long table: symbol, date, open, close, adj_close (symbol = our canonical name)."""
    try:
        if not refresh:
            cached = load_df("prices", config.RAW_DIR)
            if set(symbols).issubset(set(cached["symbol"])):
                log.info("using cached prices")
                return cached[cached["symbol"].isin(symbols)]
    except FileNotFoundError:
        pass

    import yfinance as yf
    start = (pd.Timestamp(config.START_DATE) - pd.Timedelta(days=30)).strftime("%Y-%m-%d")
    end = (pd.Timestamp(config.END_DATE) + pd.Timedelta(days=20)).strftime("%Y-%m-%d")
    ymap = {s: (config.MARKET_INDEX_TICKERS.get(s) or yahoo_ticker(s)) for s in symbols}
    frames, failed = [], []
    syms = list(symbols)
    for k in range(0, len(syms), 50):
        chunk = syms[k:k + 50]
        ych = [ymap[s] for s in chunk]
        data = with_retries(lambda: yf.download(ych, start=start, end=end, auto_adjust=False, actions=False,
                                                group_by="ticker", progress=False, threads=True),
                            what=f"prices chunk {k // 50 + 1}")
        for s in chunk:
            sub = _extract(data, ymap[s])
            if sub is None or sub.empty:
                # one retry on its own (batch downloads occasionally drop a ticker)
                single = with_retries(lambda: yf.download(ymap[s], start=start, end=end, auto_adjust=False,
                                                          actions=False, progress=False), what=f"prices {s}")
                sub = _extract(single, ymap[s])
            if sub is None or sub.empty:
                failed.append(s)
                continue
            f = sub.reset_index()
            f.columns = ["date", "open", "close", "adj_close"]
            f["symbol"] = s
            frames.append(f[["symbol", "date", "open", "close", "adj_close"]])
        log.info(f"  prices: {min(k + 50, len(syms))}/{len(syms)} symbols")
    if failed:
        log.warning(f"no price data for {len(failed)} symbols: {failed[:20]}")
    prices = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=["symbol", "date", "open", "close", "adj_close"])
    save_df(prices, "prices", config.RAW_DIR)
    return prices


# ------------------------------------------------------------ per event --
def label_and_info_day(dates: pd.DatetimeIndex, call_dt: pd.Timestamp):
    """dates: sorted trading days. Returns (label_day, info_day) or (None, None)."""
    call_day = call_dt.normalize()
    t = (call_dt.hour, call_dt.minute)
    premarket = t != (0, 0) and t < config.MARKET_OPEN_HHMM
    pos = dates.searchsorted(call_day, side="left" if premarket else "right")
    if pos >= len(dates) or pos == 0:
        return None, None
    return dates[pos], dates[pos - 1]


def past_week_return(px: pd.DataFrame, info_day: pd.Timestamp):
    """px indexed by date (sorted). Return over the last PAST_WEEK_TRADING_DAYS ending at info_day."""
    if px is None or px.empty:
        return np.nan
    upto = px.loc[:info_day]
    n = config.PAST_WEEK_TRADING_DAYS
    if len(upto) < n + 1 or upto.index[-1] < info_day - pd.Timedelta(days=4):
        return np.nan
    a, b = upto["adj_close"].iloc[-1], upto["adj_close"].iloc[-1 - n]
    return np.nan if b == 0 or pd.isna(a) or pd.isna(b) else a / b - 1


def build_market(events: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    by_sym = {}
    for s, g in prices.groupby("symbol"):
        g = g.assign(date=pd.to_datetime(g["date"])).set_index("date").sort_index()
        by_sym[s] = g[~g.index.duplicated(keep="last")]
    rows = []
    for _, e in events.iterrows():
        row = {"event_id": e["event_id"]}
        px = by_sym.get(e["ticker"])
        if px is None or px.empty:
            rows.append(row)
            continue
        label_day, info_day = label_and_info_day(px.index, pd.Timestamp(e["call_datetime"]))
        if label_day is None:
            rows.append(row)
            continue
        o, c = px.at[label_day, "open"], px.at[label_day, "close"]
        row.update(label_day=label_day, info_day=info_day, next_open=o, next_close=c,
                   prev_close=px.at[info_day, "close"],
                   label="Long" if o < c else "Short",
                   ret_stock_1w=past_week_return(px, info_day))
        for name in config.MARKET_INDEX_TICKERS:
            row[f"ret_{name}_1w"] = past_week_return(by_sym.get(name), info_day)
        rows.append(row)
    df = pd.DataFrame(rows)
    for c in ["label_day", "info_day", "next_open", "next_close", "prev_close", "label", "ret_stock_1w"] + \
             [f"ret_{n}_1w" for n in config.MARKET_INDEX_TICKERS]:
        if c not in df:
            df[c] = np.nan
    return df


def main():
    args = cli_args("Step 4: market performance + labels")
    events = load_df("events")
    symbols = sorted(events["ticker"].unique()) + list(config.MARKET_INDEX_TICKERS)
    prices = download_prices(symbols, refresh=args.refresh)
    df = build_market(events, prices)
    save_df(df, "market")
    log.info(f"labelled events: {df['label'].notna().sum():,}/{len(df):,} | "
             f"Long share: {(df['label'] == 'Long').sum() / max(df['label'].notna().sum(), 1):.1%}")


if __name__ == "__main__":
    main()
