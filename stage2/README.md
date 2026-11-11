# Stage 2: Numerical financial signals with lightweight ML

Research question: can traditional numerical models (Logistic Regression, optionally Random Forest) find a
next-day Long/Short signal in the earnings-related numbers that the Stage 1 LLM saw?

**Short answer from the experiments:** no model shows a reliable signal. All test ROC-AUCs are 0.51 to 0.53, every
95% CI includes 0.5, and no paired AUC difference versus any Stage 1 model is significant.
See the [combined report](../Submission/Final_Project_Report_without_student_details.pdf).

**Repository setup:** the original compressed Stage 1 dataset is required but not committed. See [data setup](../stage1/1_data/README.md). Numerical CSVs and fitted model files are generated locally; metrics, predictions, and plots are included.

Stage 1 files are only read, never written. Everything runs on CPU in a few minutes.

## How to run

```bash
pip install -r stage2/requirements.txt
python stage2/src/run_all.py          # rebuilds data, models, metrics, plots, checks (about 3-4 min)
python stage2/src/predict_demo.py --dataset full --event NVDA_2024-02-21   # demo
```

Run from the project root (the folder that contains `stage1/` and `stage2/`). The notebooks in `notebooks/` run the same
steps one at a time and show the outputs; they call the code in `src/`.

## Folder

| Path | What it is |
| --- | --- |
| `data/stage2_base_numeric.csv` | 8,271 events: `event_id`, `ticker`, `date`, `split`, `label`, `y` + 12 growth metrics (%) |
| `data/stage2_full_numeric.csv` | Same + 4 past-week returns (%), `eps_surprise` (%), `analyst_grade` (buy/hold/sell/none) |
| `data/feature_dictionary.csv` | Feature, source sentence, description, type, Base/Full, missing % (all and per split) |
| `data/extraction_checks.json`, `data/roundtrip_check.json` | Extraction counts and the line-by-line round-trip check |
| `src/data_extraction.py` | Parses the numbers out of the Stage 1 prompt text; stops on any unparsed line |
| `src/verify_extraction.py` | Rebuilds every sentence from the numbers and compares with the original (all 8,271 pass) |
| `src/preprocessing.py` | Leakage-safe ColumnTransformer (signed log, train-median imputation + missing flags, one-hot, scaler) |
| `src/train_logistic_regression.py` | Main model: 10-setting validation search, selection by validation AUC |
| `src/train_random_forest.py` | Optional model: 6-setting validation search, same rule |
| `src/evaluate.py` | Metrics, bootstrap CIs, paired bootstrap AUC test, loading Stage 1 predictions |
| `src/compare.py` | Test evaluation, Stage 1 comparison, paired tests, coefficients and importances |
| `src/visualization.py` | All figures |
| `src/final_checks.py` | Integrity checks (IDs, splits, labels, Stage 1 values, leakage notes) |
| `src/predict_demo.py` | Predict one event or your own numbers, with per-feature contributions |
| `src/run_all.py` | Runs everything in order and writes `results/experiment_config.json` |
| `results/predictions/` | `pred_{val,test}_{lr,rf}_{base,full}.csv`: event_id, date, label, pred, p_long, p_short |
| `results/metrics/` | `comparison_test.csv`, `paired_bootstrap_auc.csv`, `stage2_summary.json`, `config_*.json`, `search_*.csv`, coefficients, importances, `data_checks.json`, `final_checks.json` |
| `results/models/` | Fitted pipelines (`.joblib`, preprocessing + model) |
| `results/plots/` | 10 figures (comparison, ROC, confusion matrices, coefficients, correlation, Base vs Full, paired tests) |
| `results/experiment_config.json` | All experiment configurations and final metrics in one file |
| `../Submission/` | Combined final report (PDF) |

## Key design decisions

- **Data source:** `stage1/1_data/llm_data.jsonl.gz` only. The numbers exist only inside the prompt text, so they are
  parsed back with patterns that mirror `stage1/2_code/dataset_builder/step05_build_base_full.py`. Precision is 0.1%,
  exactly what the LLM saw.
- **12 Base metrics** (Operating Income included), **18 Full input features**.
- **Split:** the Stage 1 chronological split, unchanged (train 6,387 to 2023-06-30, val 938, test 946 in 2024).
- **Preprocessing (fitted on train only):** signed log `sign(x)*log(1+|x|/100)` on growth metrics and EPS surprise;
  train-median imputation plus a missing flag per growth metric; one-hot analyst grade; StandardScaler.
- **Selection:** highest validation ROC-AUC. Test predictions were never used to choose anything. Threshold 0.5.
- **Comparison:** same 946 test IDs as every Stage 1 prediction file (asserted), 2,000-resample bootstrap (seed 42).
