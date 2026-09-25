"""
make_split.py
Creates a fixed train / validation split of Source 1 entities and saves the
ground truth in an easy-to-use "one row per link" format.

Both teammates must use the SAME split, so the seed is fixed and the ID lists
are saved to disk.

Usage:
    python src/make_split.py --data-dir <path>/student_resource/dataset \
                             --work-dir /kaggle/working/work

Outputs (in <work-dir>/splits/):
    val_s1_ids.txt        one Source 1 entity_id per line (validation set)
    train_s1_ids.txt      one Source 1 entity_id per line (training set)
    gt_entities.parquet   source1_entity_id, country, n_matches, split
    gt_links.parquet      source1_entity_id, matched_entity_id, split
"""

import argparse
import os
import time

import numpy as np
import pandas as pd

START = time.time()


def log(msg):
    print(f"[{time.time() - START:7.1f}s] {msg}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True,
                        help="folder that contains train/ and test/")
    parser.add_argument("--work-dir", default="/kaggle/working/work")
    parser.add_argument("--val-size", type=int, default=50_000,
                        help="number of Source 1 entities held out for validation")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    out_dir = os.path.join(args.work_dir, "splits")
    os.makedirs(out_dir, exist_ok=True)
    train_dir = os.path.join(args.data_dir, "train")

    # ---------------------------------------------------------------- load
    log("loading ground truth ...")
    gt = pd.read_csv(os.path.join(train_dir, "train_ground_truth.tsv"),
                     sep="\t", dtype=str, keep_default_na=False)
    log("loading Source 1 countries ...")
    s1 = pd.read_csv(os.path.join(train_dir, "train_source1.tsv"),
                     sep="\t", dtype=str, keep_default_na=False,
                     usecols=["entity_id", "country"])

    gt = gt.merge(s1.rename(columns={"entity_id": "source1_entity_id"}),
                  on="source1_entity_id", how="left")
    ids = gt["matched_entity_ids"].str.strip()
    gt["n_matches"] = np.where(ids == "", 0, ids.str.count(",") + 1)

    # --------------------------------------------------------------- split
    # Sort first so the split does not depend on the file's row order.
    gt = gt.sort_values("source1_entity_id").reset_index(drop=True)
    rng = np.random.default_rng(args.seed)
    val_idx = rng.choice(len(gt), size=min(args.val_size, len(gt)), replace=False)
    gt["split"] = "train"
    gt.loc[val_idx, "split"] = "val"

    # ------------------------------------------------------------- entities
    entities = gt[["source1_entity_id", "country", "n_matches", "split"]]
    entities.to_parquet(os.path.join(out_dir, "gt_entities.parquet"), index=False)

    for split in ("val", "train"):
        path = os.path.join(out_dir, f"{split}_s1_ids.txt")
        entities.loc[entities["split"] == split, "source1_entity_id"].to_csv(
            path, index=False, header=False)

    # ---------------------------------------------------------------- links
    log("exploding links ...")
    links = pd.DataFrame({"source1_entity_id": gt["source1_entity_id"],
                          "matched_entity_id": ids.str.split(","),
                          "split": gt["split"]}).explode("matched_entity_id")
    links["matched_entity_id"] = links["matched_entity_id"].str.strip()
    links = links[links["matched_entity_id"].notna()
                  & (links["matched_entity_id"] != "")].reset_index(drop=True)
    links.to_parquet(os.path.join(out_dir, "gt_links.parquet"), index=False)

    # -------------------------------------------------------------- summary
    print("\nSplit summary:")
    summary = entities.groupby(["split", "country"]).agg(
        entities=("n_matches", "size"),
        singleton_rate=("n_matches", lambda x: (x == 0).mean()),
        avg_matches=("n_matches", "mean"))
    print(summary.to_string())
    print(f"\nLinks: {len(links):,} total, "
          f"{int((links['split'] == 'val').sum()):,} in validation")
    log(f"saved to {out_dir}")


if __name__ == "__main__":
    main()
