# Local data setup

The original datasets and trained adapters are deliberately excluded from Git.
No public download for these artifacts is configured in this repository.

To reproduce the saved experiments using the original project archive, copy:

- `stage1/1_data/llm_data.jsonl.gz` into this directory (original stored transcripts; required by Stage 2).
- `stage1/1_data/llm_data_sum.jsonl.gz` into this directory or the Kaggle inputs (summarized transcripts; required by the final Stage 1 notebook).

Both files contain 8,271 events. The Base and Full inputs are fields within the same records.

Alternatively, follow `../2_code/dataset_builder/README.md` to rebuild upstream data, use `export_compact` in `llm_stage1.py`, and apply `summarize_transcripts.py`. Check script arguments before running. Live-source reconstruction can differ from the archived experiment; the Stage 2 final checks expect the original event counts and IDs.

Stage 2 generates its numerical CSV files from the original compressed dataset. Trained adapters are required for fine-tuned inference and must be supplied locally or retrained; weights are not included here.

Respect the upstream model and dataset terms before redistributing artifacts.
