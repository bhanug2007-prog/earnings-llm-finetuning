"""Small helpers shared by all steps."""
import argparse
import logging
import os
import time

import pandas as pd

import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("stage1")


# --------------------------------------------------------------------- I/O --
def _has_pyarrow():
    try:
        import pyarrow  # noqa: F401
        return True
    except ImportError:
        return False


def save_df(df: pd.DataFrame, name: str, folder: str = None) -> str:
    """Save as parquet (or pickle if pyarrow is missing). Returns the path."""
    folder = folder or config.PROCESSED_DIR
    os.makedirs(folder, exist_ok=True)
    if _has_pyarrow():
        path = os.path.join(folder, f"{name}.parquet")
        df.to_parquet(path, index=False)
    else:
        path = os.path.join(folder, f"{name}.pkl")
        df.to_pickle(path)
    log.info(f"saved {len(df):,} rows -> {path}")
    return path


def load_df(name: str, folder: str = None) -> pd.DataFrame:
    folder = folder or config.PROCESSED_DIR
    pq = os.path.join(folder, f"{name}.parquet")
    pk = os.path.join(folder, f"{name}.pkl")
    if os.path.exists(pq):
        return pd.read_parquet(pq)
    if os.path.exists(pk):
        return pd.read_pickle(pk)
    raise FileNotFoundError(f"{name} not found in {folder}. Run the previous step first.")


# ----------------------------------------------------------------- tickers --
def norm_ticker(t: str) -> str:
    """BRK.B / BRK-B / brk.b -> BRK.B (canonical form used inside the pipeline)."""
    return str(t).strip().upper().replace("-", ".").replace("/", ".")


def yahoo_ticker(t: str) -> str:
    """Yahoo uses dashes: BRK.B -> BRK-B."""
    return norm_ticker(t).replace(".", "-")


# ------------------------------------------------------------------ retry --
def with_retries(fn, *args, what="request", retries=None, base_sleep=2.0, **kwargs):
    """Call fn(*args, **kwargs) with exponential back-off. Returns None on final failure."""
    retries = retries or config.MAX_RETRIES
    for attempt in range(1, retries + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as e:  # noqa: BLE001  (network libraries raise many types)
            if attempt == retries:
                log.warning(f"{what} failed after {retries} attempts: {type(e).__name__}: {e}")
                return None
            wait = base_sleep * (2 ** (attempt - 1))
            log.info(f"{what} failed ({type(e).__name__}); retry {attempt}/{retries - 1} in {wait:.0f}s")
            time.sleep(wait)
    return None


# -------------------------------------------------------------------- CLI --
def cli_args(description: str):
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--tickers", type=str, default=None,
                   help="comma-separated subset for a quick test, e.g. AAPL,MSFT,JPM")
    p.add_argument("--limit", type=int, default=None,
                   help="only use the first N companies (quick test)")
    p.add_argument("--refresh", action="store_true",
                   help="ignore cached downloads and fetch again")
    return p.parse_args()


def pct(x: float) -> str:
    return f"{abs(x) * 100:.1f}%"
