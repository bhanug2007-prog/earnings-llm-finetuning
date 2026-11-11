# Stage 1b: how each transcript is summarized

## Why
The model can read only 3,072 tokens. With the full transcript, it saw only the first ~12,000 characters, which is usually the CEO's opening remarks. It never saw the CFO's outlook or the analysts' questions. A summary of the most informative sentences from the whole call fits into the same space with nothing cut.

## What goes into the model (unchanged except the transcript)
```
<instruction: predict Long or Short>
Company, date
Financial metrics growth ...                 (Base and Full)
Market performance, analyst grades, EPS surprise   (Full only)
Earnings call transcript:
Prepared remarks (key sentences):
  ...
Questions and answers (key sentences):
  ...
```
Output is still one word: **Long** or **Short**.

## The method (same for train, validation, test and any new call)
1. Split the call into speaker turns ("Name: text"). Remove every Operator turn.
2. Find where the Q&A starts (phrases like "question-and-answer session", "first question"). The text before it is the prepared remarks; the text after it is the Q&A. If no marker is found, the last 45% counts as Q&A.
3. Split into sentences. Remove repeated sentences and boilerplate: safe-harbor warnings, greetings, "next question", "I'll turn the call over".
4. Score every sentence:
   - finance keywords: guidance, outlook, raised/lowered, margin, demand, headwinds, pressure, record, beat/miss, buyback ...;
   - plus numbers, percentages and $ amounts;
   - with a small preference for short, dense sentences.
5. Prepared remarks and Q&A each get half of the 8,500-character budget. If one part is short, the other gets the rest. Inside each part, the best sentences are taken evenly from its beginning, middle and end.
6. The chosen sentences go back in their **original order**.

The summary is about 8,500 characters, which is about 2,200 tokens. With the instruction and the Full features (at most about 470 tokens), the prompt stays under 3,072.

The method is deterministic: the same transcript always gives the same summary. It needs no GPU and no LLM, and takes about 0.1 s per call.

## Known limits
- The data file stores at most 40,000 characters per call. This cut 88% of calls and lost about 23% of each call's text on average, mostly the end of the Q&A. The Q&A start is still inside the stored text for about 95% of calls. If the original uncut `base.jsonl` / `full.jsonl` are available, the same script can be run on them instead.
- Extractive means sentences are copied, not rewritten. A copied sentence can lose context, for example "This impacted fixed cost absorption".

## Files
- `summarize_transcripts.py`: the summarizer.
- `check_summaries.py`: checks before any GPU time.
- `llm_data_sum.jsonl.gz`: the summarized dataset, same rows and format as `llm_data.jsonl`.
- `Summary_Checks.md`: counts, lengths and 20 sample summaries to read.
