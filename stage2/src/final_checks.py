"""Final integrity checks for Stage 2 (fails loudly). Writes results/metrics/final_checks.json."""
import gzip
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import S1, STAGE1_RUNS  # noqa: E402
from preprocessing import GRADE, ROOT, feature_columns  # noqa: E402

RES = ROOT / "stage2" / "results"
D = ROOT / "stage2" / "data"


def main():
    chk = {}
    src = [json.loads(l) for l in gzip.open(ROOT / "stage1/1_data/llm_data.jsonl.gz", "rt", encoding="utf-8")]
    src_ids = {r["event_id"]: r for r in src}
    for ds in ["base", "full"]:
        df = pd.read_csv(D / f"stage2_{ds}_numeric.csv")
        assert len(df) == 8271 and df.event_id.is_unique and set(df.event_id) == set(src_ids)
        assert all(df.split == df.event_id.map(lambda e: src_ids[e]["split"]))
        assert all(df.label == df.event_id.map(lambda e: src_ids[e]["label"]))
        cols = feature_columns(ds)
        banned = {"event_id", "ticker", "date", "split", "label", "y", "transcript", "instruction"}
        assert not banned & set(cols)
        chk[f"{ds}_rows"] = len(df)
        chk[f"{ds}_features"] = cols
    rt = json.load(open(D / "roundtrip_check.json"))
    assert rt["base"]["mismatched_heads"] == 0 and rt["full"]["mismatched_heads"] == 0
    chk["roundtrip_unparsed_or_mismatched"] = 0

    test_ids = set(pd.read_csv(D / "stage2_base_numeric.csv").query("split == 'test'").event_id)
    for tag in ["lr_base", "lr_full", "rf_base", "rf_full"]:
        p = pd.read_csv(RES / "predictions" / f"pred_test_{tag}.csv")
        assert len(p) == 946 and p.event_id.is_unique and set(p.event_id) == test_ids, tag
        assert ((p.p_long + p.p_short - 1).abs() < 1e-9).all()
        assert (p.pred == p.p_long.ge(0.5).map({True: "Long", False: "Short"})).all()
        cfg = json.load(open(RES / "metrics" / f"config_{tag}.json"))
        assert cfg["n_train"] == 6387 and cfg["n_val"] == 938 and cfg["n_test"] == 946
        assert cfg["selection"].startswith("max validation ROC-AUC")
    for name, f in STAGE1_RUNS.items():
        s1 = pd.read_csv(S1 / f)
        assert set(s1.event_id) == test_ids, f
    chk["test_ids_match_stage1_all_4_runs"] = True

    # Stage 1 files untouched: record hashes of what we read
    chk["stage1_file_sha256"] = {f: hashlib.sha256(open(S1 / f, "rb").read()).hexdigest()[:16]
                                 for f in list(STAGE1_RUNS.values()) + ["summary_all_runs.json"]}
    # Stage 1 point metrics reproduced from its own prediction files must equal the Stage 1 report
    s1sum = json.load(open(S1 / "summary_all_runs.json"))
    s2sum = json.load(open(RES / "metrics" / "stage2_summary.json"))
    chk["stage1_report_values_recomputed"] = {
        k: {m: round(v[m], 3) for m in ["accuracy", "f1_weighted", "mcc", "roc_auc"]}
        for k, v in s2sum["test_metrics"].items() if not k.startswith("Stage 2")}
    exp = {"Our QLoRA (Base)": (0.523, 0.480, 0.033, 0.532), "Our QLoRA (Full)": (0.517, 0.352, 0.000, 0.510),
           "Our zero-shot (Base)": (0.511, 0.497, 0.034, 0.509), "Our zero-shot (Full)": (0.496, 0.481, 0.003, 0.507)}
    for k, e in exp.items():
        got = chk["stage1_report_values_recomputed"][k]
        assert tuple(got[m] for m in ["accuracy", "f1_weighted", "mcc", "roc_auc"]) == e, (k, got, e)
    chk["stage1_values_equal_stage1_report"] = True
    chk["leakage_notes"] = ["imputer/scaler/encoder fitted inside Pipeline on train only",
                            "hyper-parameters chosen by validation AUC; test scored once",
                            "past-week returns end the trading day before the label day (stage1 step04)"]
    json.dump(chk, open(RES / "metrics" / "final_checks.json", "w"), indent=2)
    print("ALL CHECKS PASSED")
    print(json.dumps({k: v for k, v in chk.items() if k not in ("base_features", "full_features")}, indent=1))


if __name__ == "__main__":
    main()
