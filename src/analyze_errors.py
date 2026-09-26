"""
analyze_errors.py
Where do we lose F0.5 points on validation? CPU only, a few minutes.

Takes a finished LightGBM run (val_scores.parquet, gt_links.parquet,
val_s1_ids.txt, params.json) and, optionally, a finished cross-encoder run
(ce_val_scores.parquet, ce_params.json). With the cross-encoder it analyses
validation half B (the half the cross-encoder was not trained on), using the
same blend and thresholds as the submission; without it, all validation.

Report:
  1. Points lost per error type (every entity scores 0..1, F0.5 is the mean,
     so "points lost" = sum of (1 - entity F0.5) / number of entities):
       - singleton with a wrong prediction (the entity has no true match)
       - true matches exist but nothing predicted
       - some true matches missed
       - some wrong matches predicted
     split by country and by source of the missed / wrong record.
  2. Why true matches were missed: never a candidate (blocking), scored
     below the threshold, or given to another Source 1 entity (one-owner).
  3. Wrong matches: does the record belong to a different Source 1 entity,
     or to none?
  4. Score distribution of misses and wrong matches near the threshold.
  5. Examples (raw name | address) of both, saved to CSV.
  6. Quick post-processing trials: separate Source 2 / Source 3 thresholds
     per country, different top-1 rules. These only need --apply-only style
     re-application, no retraining.

Usage (Kaggle CPU notebook with the run outputs attached):
    python src/analyze_errors.py \
        --run-dir /kaggle/input/notebooks/<user>/ber-run3 \
        --ce-dir  /kaggle/input/notebooks/<user>/ber-run5 \
        --data-dir /kaggle/input/datasets/<user>/<slug>/student_resource/dataset
"""

import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from postprocess import (f05_macro, select_matches, tune_per_country,  # noqa: E402
                         tune_per_country_source)

START = time.time()


def log(msg):
    print(f"[{time.time() - START:7.1f}s] {msg}", flush=True)


def find(root, name, required=True):
    hits = sorted(glob.glob(os.path.join(root, "**", name), recursive=True))
    if not hits:
        if required:
            raise FileNotFoundError(f"{name} not found under {root}")
        return None
    return hits[0]


def half_of(ids):
    """Same split as cross_encoder.py (0 = fine-tuning half A, 1 = half B)."""
    h = pd.util.hash_array(np.asarray(ids, dtype=object), hash_key="0123456789abcdef")
    return (h % 2).astype(np.int8)


def blend(scores, ce_rows, w):
    """LightGBM score everywhere; for re-scored pairs w*lgb + (1-w)*ce."""
    out = scores[["s1_entity_id", "candidate_entity_id", "score", "country"]].copy()
    mixed = ce_rows[["s1_entity_id", "candidate_entity_id"]].copy()
    mixed["new"] = w * ce_rows["lgb"].to_numpy() + (1 - w) * ce_rows["ce"].to_numpy()
    out = out.merge(mixed, on=["s1_entity_id", "candidate_entity_id"], how="left")
    out["rescored"] = out["new"].notna()
    out["score"] = out["new"].fillna(out["score"])
    return out.drop(columns="new")


def apply(scores, p):
    return select_matches(scores, p["threshold"], p["one_owner"], p["top1_threshold"],
                          p.get("country_thresholds"))


def per_entity(matches, links, ids):
    """One row per Source 1 entity: n_pred, n_true, tp, f (entity F0.5)."""
    ids = pd.Index(pd.unique(np.asarray(ids)))
    pred_n = matches.groupby("s1_entity_id").size().reindex(ids, fill_value=0)
    true_n = links.groupby("source1_entity_id").size().reindex(ids, fill_value=0)
    tp = (matches.merge(links, left_on=["s1_entity_id", "candidate_entity_id"],
                        right_on=["source1_entity_id", "matched_entity_id"])
          .groupby("s1_entity_id").size().reindex(ids, fill_value=0))
    e = pd.DataFrame({"n_pred": pred_n.to_numpy(), "n_true": true_n.to_numpy(),
                      "tp": tp.to_numpy()}, index=ids)
    p = np.where(e["n_pred"] > 0, e["tp"] / e["n_pred"].clip(lower=1), 0.0)
    r = np.where(e["n_true"] > 0, e["tp"] / e["n_true"].clip(lower=1), 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        f = np.where(e["tp"] > 0, 1.25 * p * r / (0.25 * p + r), 0.0)
    e["f"] = np.where((e["n_true"] == 0) & (e["n_pred"] == 0), 1.0, f)
    e["fp"] = e["n_pred"] - e["tp"]
    e["fn"] = e["n_true"] - e["tp"]
    return e


def error_type(e):
    t = np.full(len(e), "correct", dtype=object)
    t[(e["n_true"] == 0) & (e["n_pred"] > 0)] = "singleton, wrong prediction"
    t[(e["n_true"] > 0) & (e["n_pred"] == 0)] = "matches exist, none predicted"
    t[(e["n_true"] > 0) & (e["n_pred"] > 0) & (e["fn"] > 0) & (e["fp"] == 0)] = \
        "missed some matches only"
    t[(e["n_true"] > 0) & (e["n_pred"] > 0) & (e["fp"] > 0) & (e["fn"] == 0)] = \
        "wrong matches only"
    t[(e["n_true"] > 0) & (e["n_pred"] > 0) & (e["fp"] > 0) & (e["fn"] > 0)] = \
        "both missed and wrong"
    return t


def load_raw_text(data_dir, ids):
    ids = set(ids)
    parts = []
    for i in (1, 2, 3):
        path = os.path.join(data_dir, "train", f"train_source{i}.tsv")
        for chunk in pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False,
                                 chunksize=1_000_000):
            chunk = chunk[chunk["entity_id"].isin(ids)]
            if len(chunk):
                parts.append(chunk)
    df = pd.concat(parts, ignore_index=True)
    return pd.Series((df["business_name"] + " | " + df["business_address"]).to_numpy(),
                     index=df["entity_id"].to_numpy())


def section(title):
    print("\n" + "=" * 78 + "\n" + title + "\n" + "=" * 78)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, help="finished LightGBM run output")
    ap.add_argument("--ce-dir", default=None, help="finished cross-encoder run output")
    ap.add_argument("--data-dir", required=True, help="student_resource/dataset")
    ap.add_argument("--out-dir", default="/kaggle/working/errors")
    ap.add_argument("--examples", type=int, default=15, help="examples printed per type")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # ------------------------------------------------------------ load
    val = pd.read_parquet(find(args.run_dir, "val_scores.parquet"))
    links = pd.read_parquet(find(args.run_dir, "gt_links.parquet"),
                            columns=["source1_entity_id", "matched_entity_id"])
    ids_path = find(args.run_dir, "val_s1_ids.txt", required=False)
    val_ids = (pd.read_csv(ids_path, header=None, dtype=str)[0] if ids_path
               else pd.Series(pd.unique(val["s1_entity_id"])))
    links = links[links["source1_entity_id"].isin(set(val_ids))]
    with open(find(args.run_dir, "params.json")) as fh:
        params = json.load(fh)
    log(f"LightGBM run: {len(val):,} validation pairs, {len(val_ids):,} entities")

    ce_path = find(args.ce_dir, "ce_val_scores.parquet", required=False) if args.ce_dir else None
    if ce_path:
        ce = pd.read_parquet(ce_path)
        with open(find(args.ce_dir, "ce_params.json")) as fh:
            params = json.load(fh)
        keep = val_ids[half_of(val_ids) == 1]
        val_ids = keep.reset_index(drop=True)
        ids_set = set(val_ids)
        val = val[val["s1_entity_id"].isin(ids_set)].reset_index(drop=True)
        links = links[links["source1_entity_id"].isin(ids_set)]
        scores = blend(val, ce, params["w_lgb"])
        log(f"cross-encoder run: analysing half B ({len(val_ids):,} entities), "
            f"w_lgb={params['w_lgb']}")
    else:
        scores = val[["s1_entity_id", "candidate_entity_id", "score", "country"]].copy()
        scores["rescored"] = False
    print("settings:", {k: v for k, v in params.items()
                        if k in ("threshold", "top1_threshold", "country_thresholds", "w_lgb")})

    country = val.drop_duplicates("s1_entity_id").set_index("s1_entity_id")["country"]
    country = country.reindex(val_ids).fillna("(no candidates)")
    matches = apply(scores, params)
    f_all, p_all, r_all = f05_macro(matches, links, val_ids)
    log(f"F0.5 {f_all:.4f}  precision {p_all:.4f}  recall {r_all:.4f}")

    # ------------------------------------------------------------ 1. points lost
    e = per_entity(matches, links, val_ids)
    e["type"] = error_type(e)
    e["country"] = country.reindex(e.index).to_numpy()
    n = len(e)
    section("1. F0.5 POINTS LOST BY ERROR TYPE  (points = sum(1 - entity F0.5) / entities)")
    t = e.groupby("type").agg(entities=("f", "size"), lost=("f", lambda s: (1 - s).sum() / n))
    t["share_of_entities"] = t["entities"] / n
    print(t.sort_values("lost", ascending=False).round(4).to_string())
    print(f"total lost: {(1 - e['f']).sum() / n:.4f}  (= 1 - F0.5)")
    print("\nBy country:")
    c = e.groupby(["country", "type"])["f"].apply(lambda s: (1 - s).sum() / n).unstack(fill_value=0)
    print(c.round(4).to_string())
    print(f"\nSingletons: {int((e['n_true'] == 0).sum()):,} "
          f"({(e['n_true'] == 0).mean():.1%} of entities); "
          f"predicted non-empty for {int(((e['n_true'] == 0) & (e['n_pred'] > 0)).sum()):,}")

    # ------------------------------------------------------------ 2. misses
    true_pairs = links.rename(columns={"source1_entity_id": "s1_entity_id",
                                       "matched_entity_id": "candidate_entity_id"})
    m = true_pairs.merge(scores, on=["s1_entity_id", "candidate_entity_id"], how="left")
    m = m.merge(matches[["s1_entity_id", "candidate_entity_id"]].assign(kept=1),
                on=["s1_entity_id", "candidate_entity_id"], how="left")
    owner = matches.drop_duplicates("candidate_entity_id").set_index(
        "candidate_entity_id")["s1_entity_id"]
    missed = m[m["kept"].isna()].copy()
    missed["reason"] = np.where(
        missed["score"].isna(), "not a candidate (blocking)",
        np.where(missed["candidate_entity_id"].map(owner).notna(),
                 "given to another S1 (one-owner)", "scored below threshold"))
    missed["source"] = missed["candidate_entity_id"].str[:2]
    missed["country"] = missed["s1_entity_id"].map(country)
    section(f"2. MISSED TRUE MATCHES: {len(missed):,} of {len(m):,} ({len(missed) / len(m):.2%})")
    print(missed.groupby(["reason"]).size().rename("pairs").to_frame()
          .assign(share=lambda d: d["pairs"] / len(m)).round(4).to_string())
    print("\nBy country and source:")
    print(missed.groupby(["country", "source", "reason"]).size().unstack(fill_value=0).to_string())
    below = missed[missed["reason"] == "scored below threshold"]
    if len(below):
        print("\nScores of below-threshold misses:")
        print(pd.cut(below["score"], [0, .05, .1, .2, .3, .4, .5, .6, .7, .8, 1.0])
              .value_counts().sort_index().to_string())
        if ce_path:
            print(f"re-scored by the cross-encoder: {below['rescored'].mean():.1%} "
                  "(the rest ranked below the top 10)")

    # ------------------------------------------------------------ 3. wrong matches
    wrong = matches.merge(true_pairs.assign(ok=1), on=["s1_entity_id", "candidate_entity_id"],
                          how="left")
    wrong = wrong[wrong["ok"].isna()].copy()
    true_owner = true_pairs.set_index("candidate_entity_id")["s1_entity_id"]
    wrong["belongs_to"] = np.where(wrong["candidate_entity_id"].map(true_owner).notna(),
                                   "another S1 entity", "no S1 entity")
    wrong["source"] = wrong["candidate_entity_id"].str[:2]
    wrong["country"] = wrong["s1_entity_id"].map(country)
    section(f"3. WRONG MATCHES: {len(wrong):,} of {len(matches):,} predicted "
            f"({len(wrong) / max(len(matches), 1):.2%})")
    print(wrong.groupby(["country", "source", "belongs_to"]).size()
          .unstack(fill_value=0).to_string())
    print("\nScores of wrong matches:")
    print(pd.cut(wrong["score"], [0, .5, .6, .7, .8, .9, .95, 1.0])
          .value_counts().sort_index().to_string())

    # ------------------------------------------------------------ 4. examples
    section("4. EXAMPLES")
    rng = np.random.default_rng(2026)
    ex_w = wrong.iloc[rng.permutation(len(wrong))[:200]]
    ex_m = missed[missed["reason"] != "not a candidate (blocking)"]
    ex_m = ex_m.iloc[rng.permutation(len(ex_m))[:200]]
    ex_b = missed[missed["reason"] == "not a candidate (blocking)"]
    ex_b = ex_b.iloc[rng.permutation(len(ex_b))[:100]]
    need = pd.concat([ex_w["s1_entity_id"], ex_w["candidate_entity_id"],
                      ex_m["s1_entity_id"], ex_m["candidate_entity_id"],
                      ex_b["s1_entity_id"], ex_b["candidate_entity_id"]]).unique()
    text = load_raw_text(args.data_dir, need)
    for name, df, cols in [("wrong_matches", ex_w, ["score", "belongs_to"]),
                           ("missed_scored", ex_m, ["score", "reason"]),
                           ("missed_blocking", ex_b, [])]:
        out = pd.DataFrame({"country": df["country"].to_numpy(),
                            "s1": df["s1_entity_id"].map(text).to_numpy(),
                            "candidate": df["candidate_entity_id"].map(text).to_numpy(),
                            "candidate_id": df["candidate_entity_id"].to_numpy()})
        for col in cols:
            out[col] = df[col].to_numpy()
        out.to_csv(os.path.join(args.out_dir, f"{name}.csv"), index=False)
        print(f"\n--- {name} ({len(df):,} saved to {name}.csv), first {args.examples}:")
        with pd.option_context("display.max_colwidth", 70, "display.width", 250):
            print(out.head(args.examples).to_string(index=False))

    # ------------------------------------------------------------ 5. trials
    section("5. QUICK POST-PROCESSING TRIALS (validation; no retraining needed)")
    base = dict(params)
    rows = [("current settings", f_all)]
    for t1 in (None, 0.1, 0.2, 0.3):
        p = dict(base, top1_threshold=t1)
        rows.append((f"top1_threshold={t1}", f05_macro(apply(scores, p), links, val_ids)[0]))
    ct, _ = tune_per_country(scores, links, val_ids, country, base)
    p_c = dict(base, country_thresholds=ct)
    f_c = f05_macro(apply(scores, p_c), links, val_ids)[0]
    rows.append((f"re-tuned per-country thresholds {ct}", f_c))
    st, _ = tune_per_country_source(scores, links, val_ids, country, p_c)
    p_s = dict(p_c, country_thresholds=dict(ct, **st))
    rows.append((f"per-country + source thresholds {st}",
                 f05_macro(apply(scores, p_s), links, val_ids)[0]))
    for label, f in rows:
        print(f"  {f:.4f}  ({f - f_all:+.4f})  {label}")
    print("\nNote: trials are tuned and measured on the same entities, so small gains "
          "(< +0.001) may not carry over to the leaderboard.")

    e.to_parquet(os.path.join(args.out_dir, "entity_errors.parquet"))
    log(f"saved CSVs to {args.out_dir}")


if __name__ == "__main__":
    main()
