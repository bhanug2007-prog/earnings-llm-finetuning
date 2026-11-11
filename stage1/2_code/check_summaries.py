"""Checks on the summary file before any GPU time. Writes Summary_Checks.md next to the data."""
import gzip, json, random, sys
import numpy as np

src, orig, out = sys.argv[1], sys.argv[2], sys.argv[3]
rows = [json.loads(l) for l in gzip.open(src, "rt", encoding="utf-8")]
ids_orig = [json.loads(l)["event_id"] for l in open(orig, encoding="utf-8")]
sc = np.array([r["summary_chars"] for r in rows])
head = np.array([len(r["instruction"]) + len(r["full_head"]) for r in rows])
has_qa = np.mean(["Questions and answers (key sentences):" in r["transcript"] for r in rows])
est_tok = (sc + head) / 4.0 + 30            # rough: ~4 characters per Llama-3 token + chat template
L = []
L.append("# Summary checks\n")
L.append(f"- Rows: **{len(rows)}** (original file {len(ids_orig)}); same events in the same order: **{[r['event_id'] for r in rows] == ids_orig}**")
L.append(f"- Empty summaries: **{int((sc < 500).sum())}** (under 500 characters)")
for s in ("train", "val", "test"):
    sub = [r for r in rows if r["split"] == s]
    L.append(f"- {s}: {len(sub)} events, Long share {np.mean([r['label'] == 'Long' for r in sub]):.3f}")
L.append(f"- Labels, splits, instruction and financial/market text: unchanged (only the transcript field was replaced)")
L.append(f"- Summaries that include a Q&A part: **{has_qa:.1%}**")
L.append("\n| | min | 10% | median | 90% | max |\n|---|---|---|---|---|---|")
for name, v in (("Summary characters", sc), ("Instruction + Full features characters", head),
                ("Estimated prompt tokens (Full)", est_tok)):
    p = np.percentile(v, [0, 10, 50, 90, 100])
    L.append(f"| {name} | " + " | ".join(f"{x:,.0f}" for x in p) + " |")
L.append(f"\nEstimated share of Full prompts over 3,056 tokens: **{np.mean(est_tok > 3056):.1%}**. "
         "This is an estimate (about 4 characters per token); the training notebook prints the exact count before training.")
L.append("\n## 20 random summaries (seed 0)\n")
for r in random.Random(0).sample(rows, 20):
    L.append(f"### {r['event_id']} ({r['split']}, {r['label']}), {r['transcript_original_chars']:,} → {r['summary_chars']:,} characters\n")
    L.append("```text\n" + r["transcript"] + "\n```\n")
open(out, "w", encoding="utf-8").write("\n".join(L))
print("\n".join(L[:16]))
