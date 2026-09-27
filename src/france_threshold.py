"""
france_threshold.py
Re-applies a finished cross-encoder run's test scores with a different
threshold for France only, and writes one matching_results file per value.

Why: France has no training data, so its threshold falls back to the global
one tuned on India + US. The leaderboard implies France scores lower than
the other countries, while French entities get as many predicted matches as
US ones, which points to too many wrong matches. Because matches never cross
countries, changing only the French threshold leaves India and US predictions
exactly as they were: any leaderboard change comes from France alone.

CPU only, a few minutes. No retraining.

Usage (Kaggle notebook with the LightGBM run, the cross-encoder run and the
dataset attached):
    python src/france_threshold.py \
        --run-dir /kaggle/input/notebooks --ce-dir /kaggle/input/notebooks \
        --data-dir .../student_resource/dataset \
        --france 0.7,0.8,0.9
Outputs: /kaggle/working/france/matching_results_fr<value>.tsv
         plus matching_results_base.tsv (unchanged settings, for comparison)
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from postprocess import select_matches  # noqa: E402


def find(root, name):
    hits = sorted(glob.glob(os.path.join(root, "**", name), recursive=True))
    if not hits:
        raise FileNotFoundError(f"{name} not found under {root}")
    return hits[0]


def blend(scores, ce_rows, w):
    out = scores[["s1_entity_id", "candidate_entity_id", "score", "country"]].copy()
    mixed = ce_rows[["s1_entity_id", "candidate_entity_id"]].copy()
    mixed["new"] = w * ce_rows["lgb"].to_numpy() + (1 - w) * ce_rows["ce"].to_numpy()
    out = out.merge(mixed, on=["s1_entity_id", "candidate_entity_id"], how="left")
    out["score"] = out["new"].fillna(out["score"])
    return out.drop(columns="new")


def write(matches, all_s1, path):
    grouped = matches.groupby("s1_entity_id")["candidate_entity_id"].apply(",".join)
    out = pd.DataFrame({"source1_entity_id": all_s1})
    out["matched_entity_ids"] = out["source1_entity_id"].map(grouped).fillna("")
    out.to_csv(path, sep="\t", index=False)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, help="LightGBM run (test_scores.parquet)")
    ap.add_argument("--ce-dir", required=True, help="cross-encoder run (ce_test_scores.parquet)")
    ap.add_argument("--data-dir", required=True, help="student_resource/dataset")
    ap.add_argument("--france", default="0.7,0.8,0.9", help="comma-separated thresholds")
    ap.add_argument("--out-dir", default="/kaggle/working/france")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    with open(find(args.ce_dir, "ce_params.json")) as fh:
        p = json.load(fh)
    test = pd.read_parquet(find(args.run_dir, "test_scores.parquet"))
    ce_path = glob.glob(os.path.join(args.ce_dir, "**", "ce_test_scores.parquet"),
                        recursive=True)
    if ce_path:
        scores = blend(test, pd.read_parquet(sorted(ce_path)[0]), p["w_lgb"])
    else:   # the cross-encoder did not help on validation (w_lgb = 1): LightGBM only
        if p["w_lgb"] != 1.0:
            raise FileNotFoundError(f"ce_test_scores.parquet not found under {args.ce_dir}")
        scores = test[["s1_entity_id", "candidate_entity_id", "score", "country"]].copy()
    all_s1 = pd.read_csv(os.path.join(args.data_dir, "test", "test_source1.tsv"), sep="\t",
                         dtype=str, keep_default_na=False, usecols=["entity_id"])["entity_id"]
    ct = dict(p.get("country_thresholds") or {})
    print(f"settings: global threshold {p['threshold']}, country thresholds {ct}, "
          f"w_lgb {p['w_lgb']}, top1 {p['top1_threshold']}")
    print(f"countries in test scores: {sorted(scores['country'].dropna().unique())}")
    fr_key = next((c for c in scores["country"].dropna().unique()
                   if str(c).lower().startswith("fr")), "france")
    country_of = scores.drop_duplicates("s1_entity_id").set_index("s1_entity_id")["country"]

    def summary(label, matches):
        m = matches.assign(country=matches["s1_entity_id"].map(country_of))
        per = m.groupby("country").size()
        ents = country_of.value_counts()
        fr_n = int(per.get(fr_key, 0))
        print(f"  {label:<12} France: {fr_n:,} matches, "
              f"{fr_n / max(int(ents.get(fr_key, 1)), 1):.2f} per entity with candidates | "
              f"India {int(per.get('india', 0)):,}  US {int(per.get('us', 0)):,}")

    base = select_matches(scores, p["threshold"], p["one_owner"], p["top1_threshold"], ct)
    write(base, all_s1, os.path.join(args.out_dir, "matching_results_base.tsv"))
    summary("base", base)
    for t in [float(x) for x in args.france.split(",")]:
        m = select_matches(scores, p["threshold"], p["one_owner"], p["top1_threshold"],
                           dict(ct, **{fr_key: t}))
        name = f"matching_results_fr{t:.2f}.tsv"
        write(m, all_s1, os.path.join(args.out_dir, name))
        summary(f"france {t:.2f}", m)
    print(f"\nwritten to {args.out_dir}")


if __name__ == "__main__":
    main()
