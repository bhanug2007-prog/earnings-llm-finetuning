"""
STEP 5 - Merge everything, drop missing rows, textualise numbers, write Base and Full datasets

Paper: numbers are converted to sentences ("textualisation"), then
         Base = financial-metric growth + earnings transcript
         Full = Base + market performance + analyst grades + earnings surprise
       "Following the removal of rows with missing values, the resultant dataset comprised 8,556 rows"
Both datasets contain exactly the SAME events (only the input text differs), as in the paper.

Output: data/final/merged_events.csv        (all numbers, no transcript - for inspection / report)
        data/final/base.jsonl, full.jsonl   (instruction / input / output records)
        data/final/sample_prompts.txt       (3 examples to read by eye)
        data/final/stats.json
"""
import json
import os

import pandas as pd

import config
from step02_financial_growth import growth_columns
from utils import load_df, log, pct

MARKET_NAMES = list(config.MARKET_INDEX_TICKERS)


# --------------------------------------------------------- textualisation --
def fin_sentences(row) -> list:
    out = []
    for col in growth_columns():
        g = row.get(col)
        if g is None or pd.isna(g):
            continue
        name = col.replace("growth_", "")
        if abs(g) < 0.0005:
            out.append(f"Compared to the same quarter last year, {name} remained unchanged.")
        elif g > 0:
            out.append(f"Compared to the same quarter last year, {name} grew by {pct(g)}.")
        else:
            out.append(f"Compared to the same quarter last year, {name} declined by {pct(g)}.")
    return out


def _move(r):
    if abs(r) < 0.0005:
        return "was unchanged"
    return f"went {'up' if r > 0 else 'down'} by {pct(r)}"


def market_sentences(row) -> list:
    out = [f"In the past week, the company's stock ({row['ticker']}) {_move(row['ret_stock_1w'])}."]
    for n in MARKET_NAMES:
        out.append(f"In the past week, {n} {_move(row[f'ret_{n}_1w'])}.")
    return out


def analyst_sentence(row) -> str:
    g = row.get("analyst_grade")
    if g is None or pd.isna(g):
        return f"In the past {config.ANALYST_WINDOW_DAYS} days, no analyst grades were issued for this stock."
    verb = {"buy": "buying", "hold": "holding", "sell": "selling"}[g]
    return f"In the past {config.ANALYST_WINDOW_DAYS} days, most grading companies suggest {verb} this stock."


def eps_sentence(row) -> str:
    s = row["eps_surprise"]
    if abs(s) < 0.0005:
        return "The company's reported earnings per share were in line with analysts' consensus estimates."
    return (f"The company's reported earnings per share were {pct(s)} "
            f"{'higher' if s > 0 else 'lower'} than analysts' consensus estimates.")


def build_input(row, full: bool) -> str:
    parts = [f"Company: {row['company_name']} ({row['ticker']})",
             f"Earnings call date: {pd.Timestamp(row['call_date']).strftime('%Y-%m-%d')}",
             "",
             "Financial metrics growth:",
             *fin_sentences(row)]
    if full:
        parts += ["", "Market performance:", *market_sentences(row),
                  "", "Analyst grades:", analyst_sentence(row),
                  "", "Earnings surprise:", eps_sentence(row)]
    parts += ["", "Earnings call transcript:", row["transcript"]]
    return "\n".join(parts)


def split_of(d: pd.Timestamp) -> str:
    if d <= pd.Timestamp(config.TRAIN_END):
        return "train"
    if d <= pd.Timestamp(config.VAL_END):
        return "val"
    return "test"


# ------------------------------------------------------------------ merge --
def merge_all(events, fin, eps, mkt) -> tuple:
    df = events.merge(fin, on="event_id", how="left").merge(eps, on="event_id", how="left") \
               .merge(mkt, on="event_id", how="left")
    n0 = len(df)
    steps = {}

    def keep(mask, why):
        nonlocal df
        before = len(df)
        df = df[mask.reindex(df.index, fill_value=False)]
        steps[why] = before - len(df)

    keep(df["label"].notna(), "no next-day price / label")
    keep(df["n_fin_metrics"].fillna(0) >= config.MIN_FIN_METRICS, f"< {config.MIN_FIN_METRICS} financial growth metrics")
    keep(df["eps_surprise"].notna(), "no EPS surprise")
    market_cols = ["ret_stock_1w"] + [f"ret_{n}_1w" for n in MARKET_NAMES]
    keep(df[market_cols].notna().all(axis=1), "missing past-week market performance")
    if config.REQUIRE_ANALYST_GRADE:
        keep(df["analyst_grade"].notna(), "no analyst grade in 30 days")
    df = df.reset_index(drop=True)
    df["split"] = df["call_date"].map(split_of)
    return df, {"events_before_filter": n0, "dropped": steps, "final_rows": len(df)}


def write_jsonl(df, path, full: bool):
    with open(path, "w", encoding="utf-8") as fh:
        for _, r in df.iterrows():
            rec = {"event_id": r["event_id"], "ticker": r["ticker"],
                   "date": pd.Timestamp(r["call_date"]).strftime("%Y-%m-%d"), "split": r["split"],
                   "instruction": config.INSTRUCTION, "input": build_input(r, full), "output": r["label"]}
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    log.info(f"wrote {len(df):,} records -> {path}")


def main():
    events = load_df("events")
    fin, eps, mkt = load_df("financials"), load_df("eps_analyst"), load_df("market")
    df, info = merge_all(events, fin, eps, mkt)

    out = config.FINAL_DIR
    df.drop(columns=["transcript"]).to_csv(os.path.join(out, "merged_events.csv"), index=False)
    write_jsonl(df, os.path.join(out, "base.jsonl"), full=False)
    write_jsonl(df, os.path.join(out, "full.jsonl"), full=True)

    with open(os.path.join(out, "sample_prompts.txt"), "w", encoding="utf-8") as fh:
        for _, r in df.head(3).iterrows():
            for name, full in (("BASE", False), ("FULL", True)):
                txt = build_input(r, full)
                fh.write(f"===== {name} | {r['event_id']} | label={r['label']} =====\n")
                fh.write(config.INSTRUCTION + "\n\n" + txt[:3000] + "\n[... transcript truncated in this preview ...]\n\n")

    info.update({
        "companies": int(df["ticker"].nunique()),
        "period": [str(df["call_date"].min())[:10], str(df["call_date"].max())[:10]] if len(df) else None,
        "label_share": df["label"].value_counts(normalize=True).round(4).to_dict(),
        "split_sizes": df["split"].value_counts().to_dict(),
        "analyst_grade_share": df["analyst_grade"].value_counts(normalize=True, dropna=False).round(4)
                                 .rename(index=lambda x: str(x)).to_dict(),
        "transcript_chars_mean": float(df["transcript"].str.len().mean()) if len(df) else 0,
        "approx_tokens_mean_full": float(df["transcript"].str.len().mean() / 4) if len(df) else 0,
        "paper_reference": {"rows": 8556, "companies": 501, "avg_tokens": 12375},
    })
    with open(os.path.join(out, "stats.json"), "w") as fh:
        json.dump(info, fh, indent=2, default=str)
    log.info("dataset summary:\n" + json.dumps(info, indent=2, default=str))


if __name__ == "__main__":
    main()
