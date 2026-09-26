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
    --stage2       two-stage re-ranker (see stage2.py); ~2x training time
    --transitive   (with --stage2) add similarity-to-the-best-candidate features
                   to the second stage
    --source-thresholds   also try separate thresholds for Source 2 and
                   Source 3 candidates of each country (kept only if better)
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
from postprocess import (f05_macro, select_matches, tune, tune_per_country,  # noqa: E402
                         tune_per_country_source)
from stage2 import (SCORE_FEATURES, TRANSITIVE_FEATURES, fold_of,  # noqa: E402
                    score_group_features, transitive_features)

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


def stage2_extra(pairs, s1_scores, transitive):
    """Stage-2 extra features for one set of pairs from its stage-1 scores."""
    s1_ids = pairs["s1_entity_id"].to_numpy()
    parts = [score_group_features(s1_ids, s1_scores)]
    if transitive:
        parts.append(transitive_features(s1_ids, s1_scores, pairs["b_name_core"].to_numpy(),
                                         pairs["b_addr_norm"].to_numpy()))
    return pd.concat(parts, axis=1)


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
    tr_s1 = tr_pairs["s1_entity_id"].to_numpy()
    log(f"train pairs: {len(Xtr):,} (positives {ytr.mean():.2%})")
    # keep only what stage 2 needs from the training pairs
    tr_keep = tr_pairs[["s1_entity_id", "b_name_core", "b_addr_norm"]].copy()
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
    def fit(X, y, Xv, yv, names, label):
        log(f"training LightGBM ({label}) ...")
        d = lgb.Dataset(X, y, feature_name=names)
        dv = lgb.Dataset(Xv, yv, reference=d)
        return lgb.train(params, d, num_boost_round=args.rounds, valid_sets=[dv],
                         callbacks=[lgb.early_stopping(50), lgb.log_evaluation(200)])

    nthreads = os.cpu_count()
    if not args.stage2:
        model = fit(Xtr, ytr, Xva, yva, FEATURES, "single stage")
        model.save_model(os.path.join(model_dir, "model.txt"))
        names = FEATURES
        va_score = model.predict(Xva, num_threads=nthreads)
        stage1 = None
    else:
        # ---- stage 1: two models on two halves of the training entities, so
        #      every training pair gets an out-of-fold score
        fold = fold_of(tr_s1)
        oof = np.zeros(len(Xtr))
        stage1 = []
        for k in (0, 1):
            m = fit(Xtr[fold == k], ytr[fold == k], Xva, yva, FEATURES, f"stage 1, fold {k}")
            oof[fold != k] = m.predict(Xtr[fold != k], num_threads=nthreads)
            m.save_model(os.path.join(model_dir, f"model_stage1_{k}.txt"))
            stage1.append(m)
        va_s1 = np.mean([m.predict(Xva, num_threads=nthreads) for m in stage1], axis=0)
        # ---- stage 2: original features + score-context (+ transitive) features
        log("stage-2 features ...")
        Xtr = pd.concat([Xtr.reset_index(drop=True),
                         stage2_extra(tr_keep, oof, args.transitive)], axis=1)
        Xva = pd.concat([Xva.reset_index(drop=True),
                         stage2_extra(va_pairs, va_s1, args.transitive)], axis=1)
        names = FEATURES + SCORE_FEATURES + (TRANSITIVE_FEATURES if args.transitive else [])
        model = fit(Xtr, ytr, Xva, yva, names, "stage 2")
        model.save_model(os.path.join(model_dir, "model.txt"))
        va_score = model.predict(Xva, num_threads=nthreads)
    del Xtr, tr_keep

    imp = pd.Series(model.feature_importance("gain"), index=names).sort_values(ascending=False)
    print("\nTop features by gain:")
    print((imp / imp.sum()).head(15).round(3).to_string())

    # --------------------------------------------------- validation + tuning
    va_pairs["score"] = va_score
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

    if args.source_thresholds:
        log("tuning separate Source 2 / Source 3 thresholds per country ...")
        src_t, res_s = tune_per_country_source(va_scores, links, val_ids, s1_country, best)
        res_s.to_csv(os.path.join(model_dir, "tuning_source.csv"), index=False)
        merged_t = dict(best.get("country_thresholds") or {}, **src_t)
        with_src = dict(best, country_thresholds=merged_t)
        f_before, _, _ = f05_macro(apply_params(va_scores, best), links, val_ids)
        f_src, _, _ = f05_macro(apply_params(va_scores, with_src), links, val_ids)
        print(f"\nPer-country-and-source thresholds: {src_t}")
        print(f"Validation F0.5  before: {f_before:.4f}   with source thresholds: {f_src:.4f}")
        if f_src > f_before:
            best = with_src
            print("-> using per-country-and-source thresholds")
        else:
            print("-> keeping the previous thresholds")

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
    best["stage2"] = bool(args.stage2)
    best["transitive"] = bool(args.stage2 and args.transitive)
    with open(os.path.join(model_dir, "params.json"), "w") as fh:
        json.dump(best, fh, indent=2)
    log(f"saved model and params to {model_dir}")
    return (model, stage1), best


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
            final, stage1 = model
            if stage1:
                s1 = np.mean([m.predict(X, num_threads=os.cpu_count()) for m in stage1], axis=0)
                X = pd.concat([X, stage2_extra(pairs, s1, best.get("transitive", False))],
                              axis=1)
            pairs["score"] = final.predict(X, num_threads=os.cpu_count())
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
    parser.add_argument("--stage2", action="store_true",
                        help="two-stage model: out-of-fold stage-1 scores + score-context "
                             "features feed a second LightGBM (re-ranker)")
    parser.add_argument("--transitive", action="store_true",
                        help="with --stage2: add similarity-to-best-candidate features")
    parser.add_argument("--source-thresholds", action="store_true",
                        help="also tune separate Source 2 / Source 3 thresholds per country")
    args = parser.parse_args()
    if args.transitive and not args.stage2:
        parser.error("--transitive needs --stage2")

    model_dir = os.path.join(args.work_dir, "model")
    os.makedirs(model_dir, exist_ok=True)

    if args.apply_only:
        apply_only(args, model_dir)
        log("done")
        return

    if args.skip_train:
        with open(os.path.join(model_dir, "params.json")) as fh:
            best = json.load(fh)
        stage1 = None
        if best.get("stage2"):
            stage1 = [lgb.Booster(model_file=os.path.join(model_dir, f"model_stage1_{k}.txt"))
                      for k in (0, 1)]
        model = (lgb.Booster(model_file=os.path.join(model_dir, "model.txt")), stage1)
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
