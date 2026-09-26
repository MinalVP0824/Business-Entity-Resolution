"""
train_predict.py
Matcher: features -> LightGBM -> thresholds (global + per country) +
one-owner rule -> matching_results.tsv and candidate_pairs.tsv.

Needs (from blocking.py):
    --train-cands   labelled candidates for a sample of TRAIN Source 1 ids
    --val-cands     labelled candidates for the VALIDATION Source 1 ids
    --test-cands    candidates for all TEST Source 1 ids

Usage (Kaggle):
    python src/train_predict.py \
        --train-cands /kaggle/working/work/blocking/train_trainsample_s150000 \
        --val-cands   /kaggle/working/work/blocking/train_val \
        --test-cands  /kaggle/working/work/blocking/test_all \
        --resource-dir /kaggle/input/datasets/<user>/<slug>/student_resource

    --skip-test    only train + validate (fast experiments)
    --skip-train   reuse the saved model and tuned parameters
    --apply-only   no scoring at all: re-apply (new) post-processing settings to
                   the saved test_scores.parquet and rewrite matching_results.tsv
                   (takes a minute; use after changing thresholds)

Outputs:
    <out-dir>/matching_results.tsv, <out-dir>/candidate_pairs.tsv
    <work-dir>/model/model.txt      trained LightGBM model
    <work-dir>/model/params.json    tuned post-processing settings + val F0.5
    <work-dir>/model/val_scores.parquet   validation pairs with scores + labels
    <work-dir>/model/test_scores.parquet  test pairs with score >= 0.02
    <work-dir>/model/tuning.csv, tuning_country.csv
"""

import argparse
import glob
import json
import os
import subprocess
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import FEATURES, TextStore, compute_features  # noqa: E402
from postprocess import f05_macro, select_matches, tune, tune_per_country  # noqa: E402

START = time.time()
KEEP_SCORE = 0.02   # test pairs below this score are never kept


def log(msg):
    print(f"[{time.time() - START:7.1f}s] {msg}", flush=True)


def read_parts(folder):
    files = sorted(glob.glob(os.path.join(folder, "part-*.parquet")))
    if not files:
        raise FileNotFoundError(f"no part-*.parquet files in {folder}")
    return files


def load_pairs(folder, store):
    pairs = pd.concat([pd.read_parquet(f) for f in read_parts(folder)], ignore_index=True)
    pairs = store.attach(pairs)
    X = compute_features(pairs)
    return pairs, X


def apply_params(scores, params):
    return select_matches(scores, params["threshold"], params["one_owner"],
                          params["top1_threshold"], params.get("country_thresholds"))


# ================================================================== train

def train(args, model_dir):
    store = TextStore(os.path.join(args.work_dir, "norm"), "train")
    s1_country = store.s1["country"]
    log("features for training pairs ...")
    tr_pairs, Xtr = load_pairs(args.train_cands, store)
    ytr = tr_pairs["label"].to_numpy()
    log(f"train pairs: {len(Xtr):,} (positives {ytr.mean():.2%})")
    del tr_pairs

    log("features for validation pairs ...")
    va_pairs, Xva = load_pairs(args.val_cands, store)
    yva = va_pairs["label"].to_numpy()
    log(f"val pairs: {len(Xva):,} (positives {yva.mean():.2%})")
    del store

    params = {
        "objective": "binary", "metric": "binary_logloss",
        "learning_rate": 0.08, "num_leaves": 127, "min_data_in_leaf": 100,
        "feature_fraction": 0.9, "bagging_fraction": 0.8, "bagging_freq": 1,
        "lambda_l2": 1.0, "num_threads": os.cpu_count(), "verbose": -1,
        "seed": 2026,
    }
    dtr = lgb.Dataset(Xtr, ytr, feature_name=FEATURES)
    dva = lgb.Dataset(Xva, yva, reference=dtr)
    log("training LightGBM ...")
    model = lgb.train(params, dtr, num_boost_round=args.rounds, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)])
    model.save_model(os.path.join(model_dir, "model.txt"))
    del dtr, Xtr

    imp = pd.Series(model.feature_importance("gain"), index=FEATURES).sort_values(ascending=False)
    print("\nTop features by gain:")
    print((imp / imp.sum()).head(15).round(3).to_string())

    # --------------------------------------------------- validation + tuning
    va_pairs["score"] = model.predict(Xva, num_threads=os.cpu_count())
    va_scores = va_pairs[["s1_entity_id", "candidate_entity_id", "score", "label"]].copy()
    va_scores["country"] = va_scores["s1_entity_id"].map(s1_country)
    va_scores.to_parquet(os.path.join(model_dir, "val_scores.parquet"), index=False)
    del va_pairs, Xva

    val_ids = pd.read_csv(os.path.join(args.work_dir, "splits", "val_s1_ids.txt"),
                          header=None, dtype=str)[0]
    links = pd.read_parquet(os.path.join(args.work_dir, "splits", "gt_links.parquet"),
                            columns=["source1_entity_id", "matched_entity_id"])
    links = links[links["source1_entity_id"].isin(set(val_ids))]

    log("tuning global settings on validation ...")
    best, res = tune(va_scores, links, val_ids)
    res.to_csv(os.path.join(model_dir, "tuning.csv"), index=False)
    print("\nBest global settings on validation:")
    print(res.head(6).to_string(index=False))
    f_global, _, _ = f05_macro(apply_params(va_scores, best), links, val_ids)

    log("tuning a threshold per country ...")
    country_t, res_c = tune_per_country(va_scores, links, val_ids, s1_country, best)
    res_c.to_csv(os.path.join(model_dir, "tuning_country.csv"), index=False)
    with_country = dict(best, country_thresholds=country_t)
    f_country, _, _ = f05_macro(apply_params(va_scores, with_country), links, val_ids)
    print(f"\nPer-country thresholds: {country_t}")
    print(f"Validation F0.5  global threshold: {f_global:.4f}   "
          f"per-country thresholds: {f_country:.4f}")
    if f_country > f_global:
        best = with_country
        print("-> using per-country thresholds (unseen countries use the global one)")
    else:
        print("-> keeping the single global threshold")

    matches = apply_params(va_scores, best)
    ent_country = pd.DataFrame({"id": val_ids, "country": val_ids.map(s1_country)})
    print("\nValidation F0.5 by country (final settings):")
    for country, grp in ent_country.groupby("country"):
        ids = set(grp["id"])
        f, p, r = f05_macro(matches[matches["s1_entity_id"].isin(ids)],
                            links[links["source1_entity_id"].isin(ids)], grp["id"])
        print(f"  {country:<8} F0.5 {f:.4f}  precision {p:.4f}  recall {r:.4f}")
    f, p, r = f05_macro(matches, links, val_ids)
    print(f"  {'ALL':<8} F0.5 {f:.4f}  precision {p:.4f}  recall {r:.4f}")

    best["val_f05"] = f
    with open(os.path.join(model_dir, "params.json"), "w") as fh:
        json.dump(best, fh, indent=2)
    log(f"saved model and params to {model_dir}")
    return model, best


# ================================================================ outputs

def write_matches(scores, best, all_s1, out_dir):
    matches = apply_params(scores, best)
    grouped = matches.groupby("s1_entity_id")["candidate_entity_id"].apply(",".join)
    out = pd.DataFrame({"source1_entity_id": all_s1})
    out["matched_entity_ids"] = out["source1_entity_id"].map(grouped).fillna("")
    path = os.path.join(out_dir, "matching_results.tsv")
    out.to_csv(path, sep="\t", index=False)
    has = out["matched_entity_ids"] != ""
    n_pred = out["matched_entity_ids"].str.count(",").add(1).where(has, 0)
    log(f"wrote {path}: {len(out):,} rows, {has.mean():.1%} with matches, "
        f"{n_pred.mean():.2f} matches per entity on average")
    return path


def run_validator(args, match_path, cand_path):
    if not args.resource_dir:
        return
    validator = os.path.join(args.resource_dir, "utils", "validate_submission.py")
    cmd = [sys.executable, validator, "--matching", match_path, "--candidate", cand_path,
           "--test-dir", os.path.join(args.resource_dir, "dataset", "test")]
    log("running the official validator ...")
    subprocess.run(cmd, check=False)


def predict_test(args, model, best, model_dir):
    store = TextStore(os.path.join(args.work_dir, "norm"), "test")
    all_s1 = store.s1.index.to_numpy()
    s1_country = store.s1["country"]
    os.makedirs(args.out_dir, exist_ok=True)
    cand_path = os.path.join(args.out_dir, "candidate_pairs.tsv")
    kept, seen = [], set()

    with open(cand_path, "w", encoding="utf-8") as fh:
        fh.write("source1_entity_id\tcandidate_entity_ids\n")
        for n, path in enumerate(read_parts(args.test_cands)):
            pairs = pd.read_parquet(path)
            pairs = store.attach(pairs)
            X = compute_features(pairs)
            pairs["score"] = model.predict(X, num_threads=os.cpu_count())
            kept.append(pairs.loc[pairs["score"] >= KEEP_SCORE,
                                  ["s1_entity_id", "candidate_entity_id", "score"]])
            for s1, grp in pairs.groupby("s1_entity_id", sort=False)["candidate_entity_id"]:
                fh.write(f"{s1}\t{','.join(pd.unique(grp))}\n")
                seen.add(s1)
            log(f"test part {n + 1}: {len(pairs):,} pairs scored")
            del pairs, X
        for s1 in all_s1:                      # entities blocking found nothing for
            if s1 not in seen:
                fh.write(f"{s1}\t\n")
    log(f"wrote {cand_path}")

    scores = pd.concat(kept, ignore_index=True)
    scores["country"] = scores["s1_entity_id"].map(s1_country)
    scores.to_parquet(os.path.join(model_dir, "test_scores.parquet"), index=False)
    log(f"saved {len(scores):,} test scores for later post-processing")

    match_path = write_matches(scores, best, all_s1, args.out_dir)
    run_validator(args, match_path, cand_path)


def apply_only(args, model_dir):
    """Rewrite matching_results.tsv from saved test scores + params.json."""
    with open(os.path.join(model_dir, "params.json")) as fh:
        best = json.load(fh)
    log(f"params: {best}")
    scores = pd.read_parquet(os.path.join(model_dir, "test_scores.parquet"))
    all_s1 = pd.read_parquet(os.path.join(args.work_dir, "norm", "test_source1.parquet"),
                             columns=["entity_id"])["entity_id"].to_numpy()
    os.makedirs(args.out_dir, exist_ok=True)
    match_path = write_matches(scores, best, all_s1, args.out_dir)
    cand_path = os.path.join(args.out_dir, "candidate_pairs.tsv")
    if os.path.exists(cand_path):
        run_validator(args, match_path, cand_path)
    else:
        log("candidate_pairs.tsv not found in out-dir: copy it there before submitting")


# =================================================================== main

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", default="/kaggle/working/work")
    parser.add_argument("--out-dir", default="/kaggle/working/output")
    parser.add_argument("--train-cands")
    parser.add_argument("--val-cands")
    parser.add_argument("--test-cands")
    parser.add_argument("--resource-dir", default=None,
                        help="student_resource folder, to run the official validator")
    parser.add_argument("--rounds", type=int, default=1000)
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-test", action="store_true")
    parser.add_argument("--apply-only", action="store_true")
    args = parser.parse_args()

    model_dir = os.path.join(args.work_dir, "model")
    os.makedirs(model_dir, exist_ok=True)

    if args.apply_only:
        apply_only(args, model_dir)
        log("done")
        return

    if args.skip_train:
        model = lgb.Booster(model_file=os.path.join(model_dir, "model.txt"))
        with open(os.path.join(model_dir, "params.json")) as fh:
            best = json.load(fh)
        log(f"loaded model and params: {best}")
    else:
        if not (args.train_cands and args.val_cands):
            parser.error("--train-cands and --val-cands are required unless --skip-train")
        model, best = train(args, model_dir)

    if not args.skip_test:
        if not args.test_cands:
            parser.error("--test-cands is required unless --skip-test")
        predict_test(args, model, best, model_dir)
    log("done")


if __name__ == "__main__":
    main()
