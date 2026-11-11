"""
STEP 1 - S&P 500 companies + earnings-call transcripts  ->  events table

Paper: "Full earnings call transcripts were collected for each company" (501 S&P 500 companies, via FMP).
Here : FMP's transcript endpoint is paid-only, so we use the free Hugging Face dataset
       kurry/sp500_earnings_transcripts (33k transcripts, 2005 - May 2025, MIT licence).

Output: data/processed/companies.csv
        data/processed/events.parquet   (one row = one earnings call)
"""
import io
import json

import pandas as pd
import requests

import config
from utils import cli_args, log, norm_ticker, save_df, with_retries

TRANSCRIPT_COLUMNS = ["symbol", "company_name", "year", "quarter", "date", "content"]


# --------------------------------------------------------------- companies --
def load_sp500(tickers_arg=None, limit=None) -> pd.DataFrame:
    def _get():
        r = requests.get(config.SP500_CSV_URL, timeout=30)
        r.raise_for_status()
        return r.text

    text = with_retries(_get, what="S&P 500 list")
    if text is None:
        raise RuntimeError("Could not download the S&P 500 list.")
    df = pd.read_csv(io.StringIO(text))
    df = df.rename(columns={"Symbol": "ticker", "Security": "name", "GICS Sector": "sector", "CIK": "cik"})
    df["ticker"] = df["ticker"].map(norm_ticker)
    df = df[["ticker", "name", "sector", "cik"]].drop_duplicates("ticker").reset_index(drop=True)

    if tickers_arg:
        wanted = [norm_ticker(t) for t in tickers_arg.split(",") if t.strip()]
        missing = sorted(set(wanted) - set(df["ticker"]))
        if missing:
            log.warning(f"not in current S&P 500 list (ignored): {missing}")
        df = df[df["ticker"].isin(wanted)].reset_index(drop=True)
    if limit:
        df = df.head(limit).reset_index(drop=True)
    log.info(f"companies selected: {len(df)}")
    return df


# ------------------------------------------------------------- transcripts --
def clean_transcript(text) -> str:
    """Some rows are stored JSON-encoded ("...\\n...") - decode those; strip whitespace."""
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""
    s = str(text).strip()
    if len(s) >= 2 and s[0] == '"' and s[-1] == '"':
        try:
            s = json.loads(s)
        except (json.JSONDecodeError, TypeError):
            s = s[1:-1].replace("\\n", "\n").replace('\\"', '"')
    return s.strip()


def filter_transcripts(batch: pd.DataFrame, tickers: set) -> pd.DataFrame:
    """Keep calls of our companies inside [START_DATE, END_DATE]. Pure pandas (unit-tested)."""
    b = batch.copy()
    b["ticker"] = b["symbol"].map(norm_ticker)
    b = b[b["ticker"].isin(tickers)]
    if b.empty:
        return b
    b["call_datetime"] = pd.to_datetime(b["date"], errors="coerce")
    start, end = pd.Timestamp(config.START_DATE), pd.Timestamp(config.END_DATE) + pd.Timedelta(days=1)
    b = b[(b["call_datetime"] >= start) & (b["call_datetime"] < end)]
    if b.empty:
        return b
    b["transcript"] = b["content"].map(clean_transcript)
    b = b[b["transcript"].str.len() >= config.MIN_TRANSCRIPT_CHARS]
    return b[["ticker", "company_name", "year", "quarter", "call_datetime", "transcript"]]


def finalize_events(df: pd.DataFrame) -> pd.DataFrame:
    """De-duplicate (same company, same day -> keep the longest transcript) and add ids."""
    if df.empty:
        return df
    df = df.copy()
    df["call_date"] = df["call_datetime"].dt.normalize()
    df["_len"] = df["transcript"].str.len()
    df = (df.sort_values(["ticker", "call_date", "_len"], ascending=[True, True, False])
            .drop_duplicates(["ticker", "call_date"])
            .drop(columns="_len"))
    df = df.rename(columns={"year": "fiscal_year", "quarter": "fiscal_quarter"})
    df["event_id"] = df["ticker"] + "_" + df["call_date"].dt.strftime("%Y-%m-%d")
    cols = ["event_id", "ticker", "company_name", "fiscal_year", "fiscal_quarter",
            "call_datetime", "call_date", "transcript"]
    return df[cols].sort_values(["call_date", "ticker"]).reset_index(drop=True)


def load_transcripts(tickers: set) -> pd.DataFrame:
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq

    log.info("downloading transcripts parquet from Hugging Face (~1.8 GB, first time only) ...")
    path = hf_hub_download(repo_id=config.HF_TRANSCRIPT_REPO, filename=config.HF_TRANSCRIPT_FILE,
                           repo_type="dataset")
    pf = pq.ParquetFile(path)
    available = set(pf.schema_arrow.names)
    cols = [c for c in TRANSCRIPT_COLUMNS if c in available]
    missing = set(TRANSCRIPT_COLUMNS) - set(cols)
    if {"symbol", "date", "content"} - available:
        raise RuntimeError(f"Unexpected transcript schema: {sorted(available)}")
    if missing:
        log.warning(f"transcript file lacks columns {missing}; continuing without them")

    parts = []
    for i, batch in enumerate(pf.iter_batches(batch_size=2000, columns=cols)):
        b = batch.to_pandas()
        for c in missing:
            b[c] = None
        kept = filter_transcripts(b, tickers)
        if len(kept):
            parts.append(kept)
        if i % 5 == 0:
            log.info(f"  scanned batch {i}, kept so far: {sum(len(p) for p in parts):,}")
    if not parts:
        return pd.DataFrame(columns=["ticker", "company_name", "year", "quarter", "call_datetime", "transcript"])
    return pd.concat(parts, ignore_index=True)


def main():
    args = cli_args("Step 1: companies + transcripts")
    companies = load_sp500(args.tickers, args.limit)
    companies.to_csv(f"{config.PROCESSED_DIR}/companies.csv", index=False)

    raw = load_transcripts(set(companies["ticker"]))
    events = finalize_events(raw)
    # use the S&P 500 list's company name when the transcript lacks one
    names = companies.set_index("ticker")["name"]
    events["company_name"] = events["company_name"].fillna(events["ticker"].map(names))
    save_df(events, "events")

    log.info(f"events: {len(events):,} calls from {events['ticker'].nunique()} companies "
             f"({events['call_date'].min().date() if len(events) else '-'} .. "
             f"{events['call_date'].max().date() if len(events) else '-'})")
    no_calls = sorted(set(companies["ticker"]) - set(events["ticker"]))
    if no_calls:
        log.warning(f"{len(no_calls)} companies have no transcript in the period: {no_calls[:20]}"
                    f"{' ...' if len(no_calls) > 20 else ''}")


if __name__ == "__main__":
    main()
