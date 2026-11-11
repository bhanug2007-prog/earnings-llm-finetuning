"""
Stage 1 - model side of the reproduction of
  Ni et al. (2024) "Harnessing Earnings Reports for Stock Predictions: A QLoRA-Enhanced LLM Approach"
  arXiv:2408.06634

Everything that is NOT GPU-specific lives here so it can be unit-tested and explained:
  * export_compact()      - (Colab, CPU) shrink base.jsonl/full.jsonl into one small file for Kaggle
  * load_records()        - read the compact file
  * split_records()       - chronological split (default) or random split (paper-like)
  * fit_prompt()          - build the chat prompt and cut ONLY the transcript tail to fit the context
  * build_texts()         - training texts / scoring prompts for Base or Full
  * score_prompts()       - P(Long) from the model's next-token logits (needs torch)
  * evaluate()            - Accuracy, weighted F1, MCC (+ bootstrap 95% CI), confusion matrix
  * pick_threshold()      - decision threshold tuned on VALIDATION only
  * comparison_table()    - our numbers next to the paper's Table
"""
import gzip
import json
import math
import os
import random

import numpy as np

# ------------------------------------------------------------------ paper --
PAPER = {  # Table in Ni et al. 2024 - llama-3-8b-Instruct-4bit (QLoRA) and GPT-4 baseline
    ("llama-3-8b-Instruct-4bit QLoRA", "base"): {"accuracy": 0.550, "f1_weighted": 0.533, "mcc": 0.104},
    ("llama-3-8b-Instruct-4bit QLoRA", "full"): {"accuracy": 0.573, "f1_weighted": 0.565, "mcc": 0.154},
    ("GPT-4 (no fine-tuning)", "base"): {"accuracy": 0.363, "f1_weighted": 0.482, "mcc": 0.023},
    ("GPT-4 (no fine-tuning)", "full"): {"accuracy": 0.494, "f1_weighted": 0.512, "mcc": 0.031},
}
# Paper hyper-parameters (Section "Fine-tuning")
PAPER_HPARAMS = dict(num_train_epochs=1, learning_rate=2e-4, optim="adamw_8bit", weight_decay=0.01,
                     lr_scheduler_type="linear", warmup_steps=5)

MARKER = "Earnings call transcript:\n"
TRUNC_NOTE = "\n[... transcript truncated to fit the model's context window ...]"
LABELS = ("Long", "Short")


# ============================================================ data export ==
def export_compact(final_dir: str, out_path: str, max_transcript_chars: int = 40000) -> dict:
    """Read base.jsonl + full.jsonl (same events, same order) and write one gzipped jsonl with
    base_head / full_head / transcript / label / split.  40k characters (~9-10k tokens) is far more
    than any free GPU can use, so nothing that could reach the model is lost."""
    n, longest = 0, 0
    with open(os.path.join(final_dir, "base.jsonl"), encoding="utf-8") as fb, \
         open(os.path.join(final_dir, "full.jsonl"), encoding="utf-8") as ff, \
         gzip.open(out_path, "wt", encoding="utf-8") as out:
        for lb, lf in zip(fb, ff):
            b, f = json.loads(lb), json.loads(lf)
            assert b["event_id"] == f["event_id"], "base/full files are not aligned"
            bh, bt = b["input"].split(MARKER, 1)
            fh, ft = f["input"].split(MARKER, 1)
            assert bt == ft, f"transcripts differ for {b['event_id']}"
            longest = max(longest, len(bt))
            rec = {"event_id": b["event_id"], "ticker": b["ticker"], "date": b["date"], "split": b["split"],
                   "label": b["output"], "instruction": b["instruction"],
                   "base_head": bh + MARKER, "full_head": fh + MARKER,
                   "transcript": bt[:max_transcript_chars], "transcript_chars": len(bt)}
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    return {"rows": n, "out": out_path, "size_mb": round(os.path.getsize(out_path) / 1e6, 1),
            "longest_transcript_chars": longest}


def load_records(path: str) -> list:
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def split_records(records: list, mode: str = "time", seed: int = 3407) -> dict:
    """'time'  : the dataset's own chronological split (train <= 2023-06, val 2023-H2, test 2024-H1)
       'random': shuffle and cut to the SAME sizes (closest to what the paper most likely did)."""
    if mode == "time":
        out = {s: [r for r in records if r["split"] == s] for s in ("train", "val", "test")}
    elif mode == "random":
        sizes = {s: sum(r["split"] == s for r in records) for s in ("train", "val", "test")}
        rs = list(records)
        random.Random(seed).shuffle(rs)
        a, b = sizes["train"], sizes["train"] + sizes["val"]
        out = {"train": rs[:a], "val": rs[a:b], "test": rs[b:]}
    else:
        raise ValueError("mode must be 'time' or 'random'")
    return out


def subsample(records: list, n, seed: int = 3407) -> list:
    """Same seed -> the SAME events for Base and Full (needed for a fair Base vs Full comparison)."""
    if n is None or n >= len(records):
        return list(records)
    rs = sorted(records, key=lambda r: r["event_id"])
    random.Random(seed).shuffle(rs)
    return rs[:n]


# ================================================================ prompts ==
def _n_tokens(tokenizer, text: str) -> int:
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


def chat_text(tokenizer, user: str, answer: str = None) -> str:
    msgs = [{"role": "user", "content": user}]
    if answer is not None:
        msgs.append({"role": "assistant", "content": answer})
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=answer is None)


def fit_prompt(tokenizer, instruction: str, head: str, transcript: str, max_len: int, reserve: int = 16):
    """Return (user_text, n_prompt_tokens, truncated?).  The instruction and ALL financial / market /
    analyst / EPS sentences are always kept; only the END of the transcript is cut."""
    def user_of(t, cut):
        return f"{instruction}\n\n{head}{t}{TRUNC_NOTE if cut else ''}"

    full = user_of(transcript, False)
    n = _n_tokens(tokenizer, chat_text(tokenizer, full))
    if n + reserve <= max_len:
        return full, n, False
    fixed = _n_tokens(tokenizer, chat_text(tokenizer, user_of("", True)))
    budget = max_len - reserve - fixed
    if budget <= 0:
        raise ValueError(f"max_len={max_len} too small for the instruction + features ({fixed} tokens)")
    ids = tokenizer(transcript, add_special_tokens=False)["input_ids"]
    for _ in range(6):                      # decode/encode can shift a few tokens -> shrink until it fits
        user = user_of(tokenizer.decode(ids[:budget]), True)
        n = _n_tokens(tokenizer, chat_text(tokenizer, user))
        if n + reserve <= max_len:
            return user, n, True
        budget -= (n + reserve - max_len) + 8
    raise RuntimeError("could not fit prompt")


def build_texts(records: list, dataset: str, tokenizer, max_len: int, with_answer: bool):
    """dataset: 'base' or 'full'. with_answer=True -> training texts; False -> scoring prompts."""
    assert dataset in ("base", "full")
    texts, lens, cut = [], [], 0
    for r in records:
        user, n, t = fit_prompt(tokenizer, r["instruction"], r[f"{dataset}_head"], r["transcript"], max_len)
        texts.append(chat_text(tokenizer, user, r["label"] if with_answer else None))
        lens.append(n)
        cut += t
    stats = {"n": len(texts), "mean_tokens": float(np.mean(lens)) if lens else 0,
             "max_tokens": int(max(lens)) if lens else 0, "truncated_share": cut / max(len(texts), 1)}
    return texts, stats


def label_token_ids(tokenizer):
    """First token of 'Long' / 'Short' as the model would generate them after the assistant header."""
    ids = {lab: tokenizer(lab, add_special_tokens=False)["input_ids"][0] for lab in LABELS}
    assert ids["Long"] != ids["Short"], "Long and Short share a first token"
    return ids


# ================================================================ scoring ==
def score_prompts(model, tokenizer, prompts: list, log_every: int = 100) -> np.ndarray:
    """P(Long) = softmax over the two answer tokens at the first generated position.
    Same as constrained greedy decoding to {Long, Short}; never produces an invalid answer."""
    import time
    import torch
    os.environ["UNSLOTH_RETURN_LOGITS"] = "1"
    ids = label_token_ids(tokenizer)
    lt, st = ids["Long"], ids["Short"]
    probs, t0 = [], time.time()
    model.eval()
    for i, p in enumerate(prompts):
        enc = tokenizer(p, add_special_tokens=False, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            try:
                out = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"], logits_to_keep=1)
            except TypeError:
                out = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
        logit = out.logits[0, -1].float()
        two = torch.stack([logit[lt], logit[st]])
        probs.append(torch.softmax(two, 0)[0].item())
        if log_every and (i + 1) % log_every == 0:
            el = time.time() - t0
            print(f"  scored {i + 1}/{len(prompts)}  ({el / (i + 1):.2f}s each, ~{el / (i + 1) * (len(prompts) - i - 1) / 60:.0f} min left)")
    return np.array(probs)


# ================================================================ metrics ==
def _metrics(y_true, y_pred):
    from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
    return {"accuracy": float(accuracy_score(y_true, y_pred)),
            "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
            "mcc": float(matthews_corrcoef(y_true, y_pred))}


def evaluate(labels: list, p_long, threshold: float = 0.5, n_boot: int = 1000, seed: int = 0) -> dict:
    from sklearn.metrics import confusion_matrix
    y = np.array([1 if l == "Long" else 0 for l in labels])
    p = np.asarray(p_long, dtype=float)
    yhat = (p >= threshold).astype(int)
    res = _metrics(y, yhat)
    rng = np.random.default_rng(seed)
    boots = {k: [] for k in res}
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        m = _metrics(y[idx], yhat[idx])
        for k in m:
            boots[k].append(m[k])
    res["ci95"] = {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in boots.items()}
    res["threshold"] = float(threshold)
    res["n"] = int(len(y))
    res["pred_long_share"] = float(yhat.mean()) if len(y) else 0.0
    res["true_long_share"] = float(y.mean()) if len(y) else 0.0
    cm = confusion_matrix(y, yhat, labels=[1, 0])
    res["confusion"] = {"true_Long": {"pred_Long": int(cm[0, 0]), "pred_Short": int(cm[0, 1])},
                        "true_Short": {"pred_Long": int(cm[1, 0]), "pred_Short": int(cm[1, 1])}}
    return res


def pick_threshold(val_labels: list, val_p_long, metric: str = "mcc") -> float:
    """Choose the decision threshold on the VALIDATION set only (test set never used).
    Candidates are the validation probabilities themselves (quantiles 10%..90%) so neither
    class can vanish."""
    p = np.asarray(val_p_long, dtype=float)
    y = np.array([1 if l == "Long" else 0 for l in val_labels])
    cands = np.unique(np.quantile(p, np.linspace(0.10, 0.90, 81)))
    best_t, best = 0.5, -math.inf
    for t in cands:
        m = _metrics(y, (p >= t).astype(int))[metric]
        if m > best + 1e-12:
            best, best_t = m, float(t)
    return best_t


def majority_baseline(train_labels: list, test_labels: list) -> dict:
    maj = max(set(train_labels), key=train_labels.count)
    return evaluate(test_labels, [1.0 if maj == "Long" else 0.0] * len(test_labels), threshold=0.5)


def comparison_table(ours: dict) -> str:
    """ours: {(model_name, dataset): metrics_dict}. Returns a markdown table with the paper's rows."""
    rows = ["| Model | Dataset | Accuracy | Weighted F1 | MCC | Source |", "|---|---|---|---|---|---|"]
    for (m, d), v in PAPER.items():
        rows.append(f"| {m} | {d.title()} | {v['accuracy']:.3f} | {v['f1_weighted']:.3f} | {v['mcc']:.3f} | paper |")
    for (m, d), v in ours.items():
        ci = v.get("ci95", {})
        acc = f"{v['accuracy']:.3f}" + (f" [{ci['accuracy'][0]:.3f}-{ci['accuracy'][1]:.3f}]" if ci else "")
        rows.append(f"| {m} | {d.title()} | {acc} | {v['f1_weighted']:.3f} | {v['mcc']:.3f} | ours |")
    return "\n".join(rows)


def estimate_hours(sec_per_step: float, n_train: int, batch: int, grad_acc: int) -> float:
    return sec_per_step * math.ceil(n_train / (batch * grad_acc)) / 3600
