"""
make_candidates.py
Writes candidate_pairs.tsv as the candidate set the final matching step
actually runs on, and reports how much of the truth it keeps.

Pipeline stages (see README):
  1. key-based blocking (blocking.py)          ~145 candidates per entity
  2. LightGBM candidate scorer (train_predict)  keeps pairs with score >= 0.02
  3. final matcher: cross-encoder re-scoring of each entity's top candidates,
     blended with the LightGBM score, F0.5-tuned thresholds, one-owner rule
Stage 3 only ever sees the pairs kept by stage 2, so those are the candidate
set reported in candidate_pairs.tsv (per the challenge rules: "whatever your
model actually runs inference over"). Every final match is in this set.

Inputs (found recursively under --run-dir):
  test_scores.parquet   test pairs kept by stage 2 (score >= 0.02)
  val_scores.parquet    all validation pairs with stage-2 scores + labels
  gt_links.parquet, val_s1_ids.txt   validation truth
Optional --matching: a matching_results.tsv to check is a subset.

Usage:
    python src/make_candidates.py --run-dir /kaggle/input/notebooks \
        --data-dir .../student_resource/dataset \
        --matching /kaggle/input/notebooks/<user>/ber-run5/output/matching_results.tsv
Output: /kaggle/working/candidates/candidate_pairs.tsv
"""

import argparse
import glob
import os

import numpy as np
import pandas as pd

KEEP_SCORE = 0.02


def find(root, name, required=True):
    hits = sorted(glob.glob(os.path.join(root, "**", name), recursive=True))
    if not hits and required:
        raise FileNotFoundError(f"{name} not found under {root}")
    return hits[0] if hits else None


def ceiling(found, true):
    """F0.5 per entity if the final matcher were perfect on the candidates."""
    rec = np.where(true > 0, found / np.maximum(true, 1), 1.0)
    f = np.where(true > 0, 1.25 * rec / (0.25 + rec), 1.0)
    return np.where((true > 0) & (found == 0), 0.0, f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--data-dir", required=True, help="student_resource/dataset")
    ap.add_argument("--min-score", type=float, default=KEEP_SCORE)
    ap.add_argument("--matching", default=None, help="matching_results.tsv to check")
    ap.add_argument("--out-dir", default="/kaggle/working/candidates")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # ---------------------------------------------------- validation report
    val = pd.read_parquet(find(args.run_dir, "val_scores.parquet"),
                          columns=["s1_entity_id", "candidate_entity_id", "score", "label"])
    links = pd.read_parquet(find(args.run_dir, "gt_links.parquet"),
                            columns=["source1_entity_id", "matched_entity_id"])
    ids_path = find(args.run_dir, "val_s1_ids.txt", required=False)
    val_ids = (pd.read_csv(ids_path, header=None, dtype=str)[0] if ids_path
               else pd.Series(pd.unique(val["s1_entity_id"])))
    links = links[links["source1_entity_id"].isin(set(val_ids))]
    true_n = links.groupby("source1_entity_id").size().reindex(val_ids, fill_value=0).to_numpy()
    n = len(val_ids)
    print("VALIDATION (candidate set quality)")
    print(f"{'stage':<34}{'pairs/entity':>13}{'recall':>9}{'F0.5 ceiling':>14}")
    for label, df in [("1. blocking (all candidates)", val),
                      (f"2. after LightGBM filter >= {args.min_score}",
                       val[val["score"] >= args.min_score])]:
        found = (df[df["label"] == 1].groupby("s1_entity_id").size()
                 .reindex(val_ids, fill_value=0).to_numpy())
        print(f"{label:<34}{len(df) / n:>13.1f}{found.sum() / max(true_n.sum(), 1):>9.2%}"
              f"{ceiling(found, true_n).mean():>14.4f}")

    # ---------------------------------------------------- test candidates
    test = pd.read_parquet(find(args.run_dir, "test_scores.parquet"),
                           columns=["s1_entity_id", "candidate_entity_id", "score"])
    test = test[test["score"] >= args.min_score]
    test = test.sort_values(["s1_entity_id", "score"], ascending=[True, False])
    test = test.drop_duplicates(["s1_entity_id", "candidate_entity_id"])
    grouped = test.groupby("s1_entity_id", sort=False)["candidate_entity_id"].apply(",".join)
    all_s1 = pd.read_csv(os.path.join(args.data_dir, "test", "test_source1.tsv"), sep="\t",
                         dtype=str, keep_default_na=False, usecols=["entity_id"])["entity_id"]
    out = pd.DataFrame({"source1_entity_id": all_s1})
    out["candidate_entity_ids"] = out["source1_entity_id"].map(grouped).fillna("")
    path = os.path.join(args.out_dir, "candidate_pairs.tsv")
    out.to_csv(path, sep="\t", index=False)
    per = test.groupby("s1_entity_id").size().reindex(all_s1, fill_value=0)
    print(f"\nTEST: wrote {path}")
    print(f"  {len(test):,} pairs for {len(all_s1):,} entities: mean {per.mean():.2f}, "
          f"median {per.median():.0f}, p90 {per.quantile(0.9):.0f}, max {per.max()}, "
          f"empty {(per == 0).mean():.1%}")

    # ---------------------------------------------------- subset check
    if args.matching:
        m = pd.read_csv(args.matching, sep="\t", dtype=str, keep_default_na=False)
        m = m[m["matched_entity_ids"] != ""]
        pairs = m.assign(c=m["matched_entity_ids"].str.split(",")).explode("c")
        cand = set(zip(test["s1_entity_id"], test["candidate_entity_id"]))
        missing = sum((a, b) not in cand for a, b in zip(pairs["source1_entity_id"], pairs["c"]))
        print(f"\nmatches checked: {len(pairs):,}; not in candidate set: {missing:,} "
              f"({'OK' if missing == 0 else 'PROBLEM: use a lower --min-score'})")


if __name__ == "__main__":
    main()
