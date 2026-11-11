"""
Stage 1b - extractive summaries of earnings call transcripts for fine-tuning.

Why: the model reads at most 3,072 tokens, so with full transcripts it saw only the first ~12k characters
(about a quarter of a typical call, mostly the CEO's opening remarks). A summary of the key sentences from
the WHOLE call (prepared remarks + Q&A) fits into the same budget without any cutting.

Method (same for train, validation, test and any new call):
  1. Split the call into speaker turns ("Name: text"). Drop all Operator turns.
  2. Find where the Q&A starts. Prepared remarks and Q&A each get half of the character budget
     (unused budget of one part goes to the other).
  3. Split into sentences, drop exact repeats and boilerplate (safe harbor, greetings, "next question" ...).
  4. Score each sentence: finance keywords (guidance, raised/lowered, margin, demand, headwinds ...)
     + numbers / percentages / $ amounts, with a mild preference for short, dense sentences.
  5. Within each part, take the best sentences evenly from its beginning, middle and end
     (round-robin over 3 equal slices), until the budget is used.
  6. Put the chosen sentences back in their ORIGINAL order, under two headers.
Deterministic: the same transcript always gives the same summary. No GPU, no LLM.

Usage:  python summarize_transcripts.py llm_data.jsonl llm_data_sum.jsonl.gz
"""
import gzip
import json
import re
import sys

MAX_CHARS = 8500          # ~2,200 tokens; + instruction & features (<=~470 tokens) stays under 3,072
KEYWORDS = {
    # guidance / outlook
    "guidance": 3, "outlook": 3, "expect": 2, "expects": 2, "expected": 2, "forecast": 3, "anticipate": 2,
    "full-year": 2, "full year": 2, "next quarter": 2, "second half": 2, "fiscal": 1, "target": 1,
    "reaffirm": 3, "reiterate": 2, "raise": 2, "raised": 3, "raising": 3, "lower": 2, "lowered": 3,
    "lowering": 3, "cut": 2, "narrow": 1, "high end": 2, "low end": 2,
    # direction
    "increase": 1, "increased": 1, "decrease": 1, "decreased": 1, "decline": 2, "declined": 2, "grew": 1,
    "growth": 1, "record": 2, "beat": 2, "miss": 2, "missed": 2, "exceeded": 2, "below": 1, "above": 1,
    # tone
    "strong": 1, "strength": 1, "weak": 2, "weakness": 2, "softness": 2, "headwind": 2, "headwinds": 2,
    "tailwind": 2, "tailwinds": 2, "challenging": 2, "pressure": 2, "uncertainty": 2, "uncertain": 2,
    "confident": 2, "momentum": 1, "slowdown": 2, "recovery": 1, "accelerate": 1, "decelerate": 2,
    "inflation": 1, "destocking": 2,
    # financials
    "revenue": 1, "sales": 1, "margin": 2, "margins": 2, "earnings": 1, "eps": 2, "per share": 2,
    "operating income": 1, "cash flow": 1, "demand": 2, "orders": 1, "backlog": 1, "buyback": 1,
    "repurchase": 1, "dividend": 1, "restructuring": 2, "layoffs": 2, "impairment": 2, "pricing": 1,
}
BOILERPLATE = ("forward-looking", "safe harbor", "non-gaap", "reconciliation", "webcast", "replay",
               "press release", "please go ahead", "next question", "your line is open", "thank you for",
               "thanks for taking", "good morning", "good afternoon", "good evening", "welcome to",
               "investor relations", "operator instructions", "sec filings", "risk factors",
               "question-and-answer", "question and answer session", "conference call", "i'll turn",
               "i will turn", "hand it over", "hand the call", "turn the call")
NUM = re.compile(r"(\d+(\.\d+)?\s?%|\$\s?\d|\d+(\.\d+)?\s?(million|billion|basis points|bps))", re.I)
KW_RE = [(re.compile(r"\b" + re.escape(k) + r"\b"), w) for k, w in KEYWORDS.items()]
SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'(“])")
TURN = re.compile(r"^([A-Z][^\n:]{1,60}):\s", re.M)
QA = re.compile(r"(question-and-answer|question and answer|first question|open (it|the (call|line|floor)) "
                r"(up )?(for|to) (your )?questions|operator instructions\].{0,200}question)", re.I | re.S)


def turns(text):
    """[(speaker, text)] in order; text before the first speaker label counts as speaker ''."""
    out, marks = [], list(TURN.finditer(text))
    if not marks or marks[0].start() > 0:
        out.append(("", text[:marks[0].start()] if marks else text))
    for i, m in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(text)
        out.append((m.group(1).strip(), text[m.end():end]))
    return out


def qa_start(text):
    """Character position where Q&A begins (first Q&A phrase after the opening); None if not found."""
    for m in QA.finditer(text):
        if m.start() > 3000:
            return m.start()
    return None


def sentences(text):
    return [s.strip() for s in SPLIT.split(re.sub(r"\s+", " ", text)) if s.strip()]


def score(sent):
    low = sent.lower()
    if len(sent.split()) < 8 or any(b in low for b in BOILERPLATE):
        return 0.0
    s = sum(w for rx, w in KW_RE if rx.search(low))
    s += 1.5 * min(len(NUM.findall(sent)), 3)
    return s / (1 + len(sent) / 400)


def pick(sents, budget, n_slices=3):
    """Best sentences evenly from n_slices consecutive slices, within budget; returns indices in order."""
    idx = [i for i in range(len(sents)) if score(sents[i]) > 0]
    slices = [idx[k * len(idx) // n_slices:(k + 1) * len(idx) // n_slices] for k in range(n_slices)]
    ranked = [sorted(s, key=lambda i: (-score(sents[i]), i)) for s in slices]
    chosen, used, k = [], 0, 0
    while any(k < len(r) for r in ranked):
        for r in ranked:
            if k < len(r) and used + len(sents[r[k]]) + 1 <= budget:
                chosen.append(r[k]); used += len(sents[r[k]]) + 1
        k += 1
    return sorted(chosen), used


def summarize(text, max_chars=MAX_CHARS):
    q = qa_start(text)
    if q is None:                                   # no Q&A marker: treat the last 45% as Q&A
        q = int(len(text) * 0.55)
    parts = {"prepared": [], "qa": []}
    pos = 0
    for spk, body in turns(text):
        start = text.find(body, pos); pos = max(pos, start)
        if spk.lower().startswith("operator"):
            continue
        part = "prepared" if start < q else "qa"
        parts[part].extend(sentences(body))
    seen = set()
    for k in parts:                                 # drop exact repeats
        uniq = []
        for s in parts[k]:
            if s not in seen:
                seen.add(s); uniq.append(s)
        parts[k] = uniq
    head_p, head_q = "Prepared remarks (key sentences):\n", "\n\nQuestions and answers (key sentences):\n"
    budget = max_chars - len(head_p) - len(head_q)
    half = budget // 2
    ip, used_p = pick(parts["prepared"], half)
    iq, used_q = pick(parts["qa"], budget - used_p)          # Q&A may use what prepared left over
    if used_q < budget - used_p - 200:                       # Q&A short -> give prepared the rest
        ip, used_p = pick(parts["prepared"], budget - used_q)
    out = head_p + " ".join(parts["prepared"][i] for i in ip)
    if iq:
        out += head_q + " ".join(parts["qa"][i] for i in iq)
    return out


def main(src, dst):
    op = gzip.open if src.endswith(".gz") else open
    n = 0
    with op(src, "rt", encoding="utf-8") as fin, gzip.open(dst, "wt", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            r = json.loads(line)
            r["transcript_original_chars"] = len(r["transcript"])
            r["transcript"] = summarize(r["transcript"])
            r["summary_chars"] = len(r["transcript"])
            fout.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    print(f"wrote {n} rows to {dst}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
