# Stage 1 – Dataset for "Harnessing Earnings Reports for Stock Predictions" (Ni et al., 2024)

This builds the paper's **Base** and **Full** datasets, one row per earnings event, for S&P 500 companies.

## How to run (Google Colab, CPU is enough)
1. Upload this `stage1_dataset` folder to `MyDrive/LLM_Project/`.
2. Open `Stage1_Dataset_Colab.ipynb` in Colab and put your e-mail in cell 2 (SEC rule).
3. Run the cells in this order: offline self-test, then the smoke test (5 companies), then the full run.

Each step caches what it downloads, so if Colab disconnects, just run the step again.

| Step | Script | Produces |
|---|---|---|
| 1 | `step01_companies_and_transcripts.py` | S&P 500 list + one event per earnings call (with transcript) |
| 2 | `step02_financial_growth.py` | YoY growth of 12 financial metrics (SEC EDGAR) |
| 3 | `step03_eps_and_analyst.py` | EPS surprise + 30-day majority analyst grade (Yahoo) |
| 4 | `step04_market_and_labels.py` | Past-week return of the stock/SPY/QQQ/DOW + Long/Short label |
| 5 | `step05_build_base_full.py` | `final/base.jsonl`, `final/full.jsonl`, `merged_events.csv`, `stats.json` |

`python tests/test_offline.py` checks all of the processing logic. It runs offline using data in the exact formats SEC, Yahoo and Hugging Face return, including real Apple figures.

## Paper → our implementation

| Paper (FMP API) | Ours (free) | Same definition? |
|---|---|---|
| 501 S&P 500 companies | Current S&P 500 list (503 tickers), from the `datasets/s-and-p-500-companies` list on GitHub | yes |
| Earnings-call transcripts | Hugging Face `kurry/sp500_earnings_transcripts` | yes (full transcript) |
| Financial metric growth vs. same quarter last year (income statement, balance sheet, cash flow) | Computed from SEC EDGAR XBRL: Revenue, Gross Profit, Operating Income, Net Income, EPS, Operating CF, CapEx, FCF, Total Assets, Total Liabilities, Equity, Cash | yes (FMP derives the same numbers from these filings) |
| Earnings surprise: actual vs. estimated EPS | Yahoo Finance EPS estimate and reported EPS; surprise = (actual − estimate) / \|estimate\| | yes |
| Analyst grades: most frequent grade over the past 30 days | Yahoo upgrades/downgrades history; mode of buy/hold/sell over the 30 days before the call | yes |
| Past-week performance of the stock, SPY, QQQ and DOW | Yahoo daily prices; 5-trading-day return ending at the last close before the call | yes |
| Label: next day, Long if open < close, otherwise Short | Same rule, applied to the first trading session after the call | yes |
| Rows with missing values removed (8,556 rows left) | Same rule (an empty analyst window is kept as "no grades issued") | yes |
| Textualisation ("In the past week, SPY went down by 1.5%" …) | Same sentence templates | yes |

**Not stated in the paper, so we chose and documented these:**
- Period: 2020-01 to 2024-06.
- Chronological split: train until 2023-06, validation for 2023-H2, test for 2024-H1.
- Prompt instruction text: see `config.INSTRUCTION`.

**Known differences to discuss in the report:**
- The data comes from different (free) providers, so some values may differ slightly.
- The company list is today's S&P 500, which introduces survivorship bias, as the paper's list did.
- Transcripts are kept in full here and are cut to the GPU's context length at training time.
