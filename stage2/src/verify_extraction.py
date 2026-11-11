"""Round-trip check: rebuild every numeric sentence from the extracted CSVs with Stage 1's own wording
(step05_build_base_full.py, utils.pct = f"{abs(x)*100:.1f}%") and compare with the original prompt lines.
Passes only if all 8,271 Base and Full heads are reproduced line by line."""
import gzip
import json
from pathlib import Path

import pandas as pd

from data_extraction import GROWTH, ROOT, SRC

D = ROOT / "stage2" / "data"


def move(v):
    return "was unchanged" if v == 0 else f"went {'up' if v > 0 else 'down'} by {abs(v):.1f}%"


def numeric_lines(head):
    keep = ("Compared to", "In the past", "The company's reported")
    return [l for l in head.splitlines() if l.startswith(keep)]


def rebuild(row, full):
    out = []
    for name, col in GROWTH.items():
        v = row[col]
        if pd.isna(v):
            continue
        if v == 0:
            out.append(f"Compared to the same quarter last year, {name} remained unchanged.")
        else:
            out.append(f"Compared to the same quarter last year, {name} {'grew' if v > 0 else 'declined'} by {abs(v):.1f}%.")
    if full:
        out.append(f"In the past week, the company's stock ({row['ticker']}) {move(row['ret_stock_1w'])}.")
        for n, c in [("SPY", "ret_spy_1w"), ("QQQ", "ret_qqq_1w"), ("DOW", "ret_dow_1w")]:
            out.append(f"In the past week, {n} {move(row[c])}.")
        g = row["analyst_grade"]
        out.append("In the past 30 days, no analyst grades were issued for this stock." if g == "none" else
                   f"In the past 30 days, most grading companies suggest "
                   f"{ {'buy': 'buying', 'hold': 'holding', 'sell': 'selling'}[g] } this stock.")
        s = row["eps_surprise"]
        out.append("The company's reported earnings per share were in line with analysts' consensus estimates."
                   if s == 0 else f"The company's reported earnings per share were {abs(s):.1f}% "
                                  f"{'higher' if s > 0 else 'lower'} than analysts' consensus estimates.")
    return out


def main():
    src = {json.loads(l)["event_id"]: json.loads(l) for l in gzip.open(SRC, "rt", encoding="utf-8")}
    res = {}
    for kind, file, full in [("base", "stage2_base_numeric.csv", False), ("full", "stage2_full_numeric.csv", True)]:
        df = pd.read_csv(D / file, keep_default_na=True, na_values=[""])
        df["analyst_grade"] = df.get("analyst_grade")
        bad = 0
        for _, row in df.iterrows():
            head = src[row["event_id"]][f"{kind}_head"]
            if rebuild(row, full) != numeric_lines(head):
                bad += 1
                if bad <= 3:
                    print("MISMATCH", row["event_id"], rebuild(row, full)[:3], numeric_lines(head)[:3])
            for k in ["split", "label", "ticker", "date"]:
                assert row[k] == src[row["event_id"]][k]
        res[kind] = {"rows": len(df), "mismatched_heads": bad}
        assert len(df) == 8271 and df.event_id.is_unique and bad == 0, res
    print("round-trip OK:", res)
    json.dump(res, open(D / "roundtrip_check.json", "w"), indent=2)


if __name__ == "__main__":
    main()
