"""
Stage-1 dataset configuration.

Paper being reproduced:
  "Harnessing Earnings Reports for Stock Predictions: A QLoRA-Enhanced LLM Approach"
  (Ni et al., 2024, arXiv:2408.06634)

Everything that defines the dataset lives here so it can be changed in ONE place
and documented in the report.
"""
import os

# ---------------------------------------------------------------------------
# Where to store data.  On Colab, point this at Google Drive so nothing is lost
# when the runtime disconnects, e.g. "/content/drive/MyDrive/llm_project/data"
# ---------------------------------------------------------------------------
DATA_DIR = os.environ.get("STAGE1_DATA_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"))
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
FINAL_DIR = os.path.join(DATA_DIR, "final")

# ---------------------------------------------------------------------------
# SEC EDGAR requires a User-Agent with a real contact e-mail.
# PUT YOUR OWN NAME + E-MAIL HERE (SEC blocks requests without it).
# ---------------------------------------------------------------------------
SEC_USER_AGENT = os.environ.get("SEC_USER_AGENT", "Earnings research project your_email@example.com")

# ---------------------------------------------------------------------------
# Universe and period
#   Paper: 501 S&P 500 companies, 8,556 rows after dropping missing values.
#   The paper does not state its period. It was written in mid-2024, so we use
#   2020-01-01 .. 2024-06-30 (~4.5 years x 4 quarters x 500 companies ~ 9k
#   events before filtering), which lands close to the paper's 8,556 rows.
# ---------------------------------------------------------------------------
START_DATE = "2020-01-01"
END_DATE = "2024-06-30"

SP500_CSV_URL = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv"

# Transcripts (free substitute for FMP's earnings-call-transcript endpoint)
HF_TRANSCRIPT_REPO = "kurry/sp500_earnings_transcripts"
HF_TRANSCRIPT_FILE = "parquet_files/part-0.parquet"
MIN_TRANSCRIPT_CHARS = 2000          # drop empty / broken transcripts

# ---------------------------------------------------------------------------
# Paper feature definitions
# ---------------------------------------------------------------------------
MARKET_INDEX_TICKERS = {             # paper: "SPY, QQQ, and DOW"
    "SPY": "SPY",
    "QQQ": "QQQ",
    "DOW": "^DJI",                   # Dow Jones Industrial Average
}
PAST_WEEK_TRADING_DAYS = 5           # paper: "performance ... for the past week"
ANALYST_WINDOW_DAYS = 30             # paper: "In the past 30 days, most grading companies ..."
EPS_DATE_TOLERANCE_DAYS = 3          # match Yahoo earnings date to transcript date

# Financial-metric growth (paper: income statement, balance sheet, cash flow,
# "contrasting the quarter ... with the same quarter from the previous year").
# Each metric -> list of US-GAAP XBRL tags tried in order.
DURATION_METRICS = {
    "Revenue": ["RevenueFromContractWithCustomerExcludingAssessedTax", "Revenues",
                "SalesRevenueNet", "RevenueFromContractWithCustomerIncludingAssessedTax",
                "RevenuesNetOfInterestExpense"],
    "Gross Profit": ["GrossProfit"],
    "Operating Income": ["OperatingIncomeLoss"],
    "Net Income": ["NetIncomeLoss", "ProfitLoss", "NetIncomeLossAvailableToCommonStockholdersBasic"],
    "EPS (diluted)": ["EarningsPerShareDiluted", "EarningsPerShareBasicAndDiluted"],
    "Operating Cash Flow": ["NetCashProvidedByUsedInOperatingActivities",
                            "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations"],
    "Capital Expenditure": ["PaymentsToAcquirePropertyPlantAndEquipment",
                            "PaymentsToAcquireProductiveAssets"],
}
INSTANT_METRICS = {
    "Total Assets": ["Assets"],
    "Total Liabilities": ["Liabilities"],
    "Shareholders' Equity": ["StockholdersEquity",
                             "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"],
    "Cash and Cash Equivalents": ["CashAndCashEquivalentsAtCarryingValue",
                                  "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents"],
}
DERIVED_METRICS = ["Free Cash Flow"]   # = Operating Cash Flow - Capital Expenditure
NON_ADDITIVE = {"EPS (diluted)"}       # cannot derive a quarter by subtracting YTD values
MIN_FIN_METRICS = 3                    # need at least this many growth metrics per event

# ---------------------------------------------------------------------------
# Label (paper): next trading day, Long if Open < Close else Short.
# "Next trading day" = first full trading session AFTER the earnings call:
#   call before 09:30 ET  -> that same day's session
#   call at/after 09:30   -> the following trading day
#   time unknown (00:00)  -> the following trading day
# ---------------------------------------------------------------------------
MARKET_OPEN_HHMM = (9, 30)

# Rows with missing values are dropped, as in the paper.  Analyst grades are
# the one field that is legitimately empty when no firm issued a grade in the
# 30-day window; set True to drop those rows as well.
REQUIRE_ANALYST_GRADE = False

# Chronological split (paper does not report its split; a time split avoids look-ahead).
TRAIN_END = "2023-06-30"     # train: START_DATE .. TRAIN_END
VAL_END = "2023-12-31"       # val:   TRAIN_END .. VAL_END ;  test: VAL_END .. END_DATE

INSTRUCTION = (
    "You are a financial analyst. Based on the company's earnings information below, predict the "
    "stock's price movement on the next trading day after the earnings call. Answer 'Long' if you "
    "expect the closing price to be higher than the opening price, otherwise answer 'Short'. "
    "Answer with one word: Long or Short."
)

# Network politeness
YAHOO_SLEEP_SECONDS = 1.0
SEC_SLEEP_SECONDS = 0.15              # SEC limit is 10 requests/second
MAX_RETRIES = 4

for _d in (RAW_DIR, PROCESSED_DIR, FINAL_DIR):
    os.makedirs(_d, exist_ok=True)
