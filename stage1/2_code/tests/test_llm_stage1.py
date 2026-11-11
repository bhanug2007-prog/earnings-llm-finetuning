"""Offline tests for llm_stage1.py (no GPU / torch needed).  Run: python tests/test_llm_stage1.py"""
import json
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import llm_stage1 as L  # noqa: E402

OK = []


def check(c, m):
    assert c, "FAILED: " + m
    OK.append(m)


class FakeTok:
    """Word-level tokenizer with a Llama-3 style chat template (same interface as HF tokenizers)."""
    def __init__(self):
        self.vocab, self.inv = {}, {}

    def _id(self, w):
        if w not in self.vocab:
            self.vocab[w] = len(self.vocab)
            self.inv[self.vocab[w]] = w
        return self.vocab[w]

    def __call__(self, text, add_special_tokens=False, return_tensors=None):
        return {"input_ids": [self._id(w) for w in text.replace("\n", " \n ").split(" ") if w != ""]}

    def decode(self, ids):
        return " ".join(self.inv[i] for i in ids).replace(" \n ", "\n")

    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=False):
        s = "<|begin_of_text|>"
        for m in msgs:
            s += f"<|start_header_id|>{m['role']}<|end_header_id|>\n\n{m['content']}<|eot_id|>"
        if add_generation_prompt:
            s += "<|start_header_id|>assistant<|end_header_id|>\n\n"
        return s


def make_final_dir():
    d = tempfile.mkdtemp()
    rng = np.random.default_rng(0)
    with open(f"{d}/base.jsonl", "w") as fb, open(f"{d}/full.jsonl", "w") as ff:
        for i in range(300):
            split = "train" if i < 200 else ("val" if i < 250 else "test")
            tr = "Operator: welcome . " + " ".join(f"w{j}" for j in range(int(rng.integers(50, 3000))))
            head_b = f"Company: X{i} (X{i})\nEarnings call date: 2021-01-01\n\nFinancial metrics growth:\nCompared to the same quarter last year, Revenue grew by 5.0%.\n\n"
            head_f = head_b + "Market performance:\nIn the past week, SPY went up by 1.0%.\n\n"
            lab = "Long" if rng.random() < 0.5 else "Short"
            base = {"event_id": f"X{i}_2021", "ticker": f"X{i}", "date": "2021-01-01", "split": split,
                    "instruction": "Predict Long or Short.", "input": head_b + L.MARKER + tr, "output": lab}
            full = dict(base, input=head_f + L.MARKER + tr)
            fb.write(json.dumps(base) + "\n")
            ff.write(json.dumps(full) + "\n")
    return d


def main():
    d = make_final_dir()
    out = os.path.join(d, "llm_data.jsonl.gz")
    info = L.export_compact(d, out, max_transcript_chars=40000)
    recs = L.load_records(out)
    check(info["rows"] == 300 and len(recs) == 300, "export: all rows written")
    check(recs[0]["base_head"].endswith(L.MARKER) and "Market performance" in recs[0]["full_head"]
          and "Market performance" not in recs[0]["base_head"], "export: base/full heads separated correctly")

    sp = L.split_records(recs, "time")
    check([len(sp[s]) for s in ("train", "val", "test")] == [200, 50, 50], "split: chronological sizes")
    rp = L.split_records(recs, "random")
    check([len(rp[s]) for s in ("train", "val", "test")] == [200, 50, 50], "split: random keeps sizes")
    check({r["event_id"] for r in rp["train"]} != {r["event_id"] for r in sp["train"]}, "split: random differs")
    s1, s2 = L.subsample(sp["train"], 50), L.subsample(sp["train"], 50)
    check([r["event_id"] for r in s1] == [r["event_id"] for r in s2], "subsample: deterministic (same rows Base/Full)")

    tok = FakeTok()
    for ds in ("base", "full"):
        texts, st = L.build_texts(sp["train"], ds, tok, max_len=600, with_answer=True)
        lens = [len(tok(t)["input_ids"]) for t in texts]
        check(max(lens) <= 600, f"{ds}: every training text fits max_len")
        check(all(t.rstrip().endswith(("Long<|eot_id|>", "Short<|eot_id|>")) for t in texts), f"{ds}: answer kept at the end")
        check(all("Revenue grew by 5.0%" in t for t in texts), f"{ds}: features never truncated")
        check(0 < st["truncated_share"] < 1, f"{ds}: only long transcripts are truncated")
    prompts, _ = L.build_texts(sp["test"], "full", tok, max_len=600, with_answer=False)
    check(all(p.endswith("<|start_header_id|>assistant<|end_header_id|>\n\n") for p in prompts),
          "scoring prompt ends at the assistant header (same prefix as training)")
    tr_texts, _ = L.build_texts(sp["test"], "full", tok, max_len=600, with_answer=True)
    check(all(t.startswith(p) for t, p in zip(tr_texts, prompts)), "training text = scoring prompt + answer")
    try:
        L.fit_prompt(tok, "x " * 50, "y " * 600, "z", max_len=100)
        check(False, "too-small max_len raises")
    except ValueError:
        check(True, "too-small max_len raises")

    labels = ["Long", "Short"] * 50
    perfect = [0.9 if l == "Long" else 0.1 for l in labels]
    r = L.evaluate(labels, perfect)
    check(r["accuracy"] == 1 and r["mcc"] == 1 and r["f1_weighted"] == 1, "metrics: perfect predictions")
    r = L.evaluate(labels, [0.2] * 100)
    check(r["accuracy"] == 0.5 and r["mcc"] == 0 and r["pred_long_share"] == 0, "metrics: one-class collapse -> MCC 0")
    check(r["confusion"]["true_Long"]["pred_Short"] == 50, "metrics: confusion matrix")
    rng = np.random.default_rng(1)
    noisy = np.clip(np.array([0.48 + (0.04 if l == "Long" else 0) for l in labels]) + rng.normal(0, 0.02, 100), 0, 1)
    t = L.pick_threshold(labels, noisy)
    check(0.45 < t < 0.55 and L.evaluate(labels, noisy, t)["mcc"] > 0.3, "threshold: tuned on validation separates classes")
    r = L.evaluate(labels, noisy, 0.5)
    check(r["ci95"]["accuracy"][0] <= r["accuracy"] <= r["ci95"]["accuracy"][1], "metrics: bootstrap CI contains estimate")
    mb = L.majority_baseline(["Short"] * 6 + ["Long"] * 4, labels)
    check(mb["accuracy"] == 0.5 and mb["mcc"] == 0, "majority baseline")
    tab = L.comparison_table({("ours QLoRA", "full"): r})
    check("| llama-3-8b-Instruct-4bit QLoRA | Full | 0.573 | 0.565 | 0.154 | paper |" in tab and "ours" in tab, "comparison table")
    check(abs(L.estimate_hours(30, 6387, 1, 8) - 30 * 799 / 3600) < 1e-9, "time estimate")
    print(f"ALL {len(OK)} CHECKS PASSED")


if __name__ == "__main__":
    main()
