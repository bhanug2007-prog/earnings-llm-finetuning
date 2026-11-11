# Historical workflow notes

These notes describe the earlier raw-transcript workflow. For final summarized runs, use the [root README](../../README.md) and `../3_notebooks/1_Train_and_Evaluate.ipynb`. Old notebook names below are historical and are not included.

# Stage 1: training and evaluation (QLoRA, Llama-3-8B-Instruct, 4-bit)

This reproduces the main experiment of Ni et al. (2024): fine-tune **llama-3-8b-Instruct-4bit** with QLoRA on the **Base** and **Full** datasets, then report **Accuracy, weighted F1 and MCC**.

## Files
| File | What it is |
|---|---|
| `llm_stage1.py` | Prompt building, transcript truncation, scoring, metrics, threshold choice and the comparison table. Tested offline with `python tests/test_llm_stage1.py` (25 checks). |
| `Stage1_QLoRA_Train_Eval.ipynb` | The GPU notebook. It runs on Kaggle (free T4) or Colab. |

## Step 1: make the small data file (Colab, CPU, about 3 minutes)
```python
from google.colab import drive; drive.mount('/content/drive')
import sys; sys.path.insert(0, '/content/drive/MyDrive/stage1_llm')
import llm_stage1 as L
print(L.export_compact('/content/drive/MyDrive/stage1_dataset/data/final',
                       '/content/drive/MyDrive/stage1_llm/llm_data.jsonl.gz'))
```
Download `llm_data.jsonl.gz` and `llm_stage1.py` from Drive. On Kaggle, go to **Datasets → New Dataset** and upload both files (for example as a dataset named `earnings-llm-stage1`).

## Step 2: Kaggle notebook setup
- Upload `Stage1_QLoRA_Train_Eval.ipynb` using **Code → New Notebook → File → Import Notebook**.
- Settings: **Accelerator: GPU T4 x2** and **Internet: On**. Kaggle requires a phone-verified account for both.
- Use **Add Input** to attach your dataset.

## Step 3: run order
For each run, set the config cell, then use **Save Version → Save & Run All**. This runs in the background, so you can close the browser.

| # | DATASET | MODE | Purpose | Approx. time |
|---|---|---|---|---|
| 1 | base | timing | Measures speed, then prints how many training events fit in about 10 h | 20 min |
| 1b | base | pilot | 800 training events, scored on validation only. A cheap check that learning works before the long runs | about 1.5 h |
| 2 | base | train_eval | **Paper experiment: Base** | ≤ 10 h |
| 3 | full | train_eval | **Paper experiment: Full** (same TRAIN_LIMIT and MAX_SEQ_LEN as run 2) | ≤ 10 h |
| 4 | base | zeroshot | No fine-tuning baseline (the paper's comparison is GPT-4) | about 1 h |
| 5 | full | zeroshot | Same, for the Full dataset | about 1 h |
| 6 (optional) | full | train_eval with `SPLIT_MODE="random"` | Paper-like random split | ≤ 10 h |

The results are written to `/kaggle/working`: `results_*.json`, `pred_test_*.csv` and `adapter_*/`. Download them from the version's **Output** tab. Kaggle gives 30 free GPU hours per week, so runs 2–5 fit in one week.

## Choices the paper did not state (all documented)
| Item | Our choice | Reason |
|---|---|---|
| Prompt template | Llama-3 chat format: instruction + textualised features + transcript; the answer is "Long" or "Short" | Matches the paper's description of instruction tuning |
| Training loss | Computed on the answer tokens only | Otherwise about 99.9% of the loss would come from copying the transcript |
| Context length | 3,072 tokens (the paper used 25k) | Free T4 GPU memory and time. Features always come first and are never cut; only the end of the transcript is cut. |
| LoRA rank/alpha | r = 16, alpha = 16, applied to all attention and MLP projections | Unsloth defaults; the paper's other hyper-parameters equal Unsloth defaults |
| Batch | 1 × 8 gradient accumulation | Fits T4 memory |
| Prediction | P(Long) from the next-token logits for "Long" vs "Short". **argmax** (threshold 0.5) is the paper-comparable number. A threshold tuned on validation only is reported separately. | The model can never give an unparseable answer |
| Split | Chronological by default; random split as an option | The paper does not report its split |

## Reading the results
- `argmax_0.5` is the model's own answer. Compare this with the paper.
- `ci95` is the 95% bootstrap interval. If the paper's number falls inside it, the two results are statistically consistent.
- `pred_long_share` shows how often the model predicts Long. If it is near 0 or 1, the model is predicting one class for almost everything (MCC ≈ 0). The `val_tuned` result is there to diagnose this.
- `majority_baseline` always predicts the most common class. The fine-tuned model should beat it on MCC.
