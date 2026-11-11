# Stage 1: Llama 3 QLoRA experiments

Start with the [project README](../README.md) for verified settings and results.

- `1_data/README.md`: required local datasets (not committed).
- `2_code/dataset_builder/`: upstream data construction and offline tests.
- `2_code/llm_stage1.py`: prompt construction, compact export, scoring, and metrics.
- `2_code/summarize_transcripts.py`: deterministic extractive summaries.
- `3_notebooks/1_Train_and_Evaluate.ipynb`: final summarized-transcript experiments.
- `3_notebooks/2_Predict_New_Transcript.ipynb`: prediction with locally supplied adapters.
- `4_results/final_summarized/`: final metrics and per-event predictions.
- `5_reports/Summary_Method.md`: summarization method and limitations.

Notebook outputs are cleared for publication. Saved JSON and CSV files retain the experiment evidence. Early runs, old notebooks, logs, and model artifacts are excluded.
