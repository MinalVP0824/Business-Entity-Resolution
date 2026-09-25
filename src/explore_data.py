"""
explore_data.py
Memory-aware first look at the Business Entity Resolution dataset.

Usage (Kaggle):
    python explore_data.py --data-dir /kaggle/input/datasets/<user>/<slug>/student_resource/dataset
"""

import argparse
import gc
import os
import time

import numpy as np
import pandas as pd

pd.set_option("display.max_colwidth", 80)
pd.set_option("display.width", 200)

START = time.time()


def log(msg):
    print(f"[{time.time() - START:7.1f}s] {msg}", flush=True)


def section(title):
    print("\n" + "=" * 90, flush=True)
    print(title, flush=True)
    print("=" * 90, flush=True)


def count_rows(path):
    """Fast newline count in 16 MB chunks; minus 1 for the header."""
    n = 0
    with open(path, "rb") as f:
        while True:
            buf = f.read(1 << 24)
            if not buf:
                break
            n += buf.count(b"\n")
    return max(n - 1, 0)


def load_source(path):
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False,
                     index_col="entity_id")
    df["country"] = df["country"].astype("category")
    return df


def normalize(s):
    """Very light normalisation, only for measuring difficulty."""
    return (s.str.lower()
             .str.replace(r"[^\w\s]", " ", regex=True)
             .str.split()
             .str.join(" "))


def describe_source(name, df, expected_rows):
    mem = df.memory_usage(deep=True).sum() / 1e9
    print(f"\n--- {name}: {len(df):,} rows loaded "
          f"(file has ~{expected_rows:,} data lines), {mem:.2f} GB in RAM ---")
    if abs(len(df) - expected_rows) > 1:
        print("  WARNING: loaded rows != line count -> quote characters may be "
              "merging rows. Tell Claude about this.")
    print(f"  entity_id unique: {df.index.is_unique}")
    print("  Country counts:")
    print(df["country"].value_counts().to_string())
    for col in ["business_name", "business_address"]:
        s = df[col]
        lengths = s.str.len()
        empty = int((s.str.strip() == "").sum())
        print(f"  {col}: empty={empty:,}, avg_len={lengths.mean():.1f}, "
              f"max_len={lengths.max()}")
    dups = int(df.duplicated(subset=["business_name", "business_address"]).sum())
    print(f"  Exact duplicate (name, address) rows: {dups:,}")
    print("  Sample rows:")
    print(df.sample(3, random_state=0).to_string())


def lookup(tr, eid):
    src = tr[1] if eid.startswith("S1-") else tr[2] if eid.startswith("S2-") else tr[3]
    if eid in src.index:
        row = src.loc[eid]
        return row["business_name"], row["business_address"], row["country"]
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="dataset")
    parser.add_argument("--examples", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_dir = os.path.join(args.data_dir, "train")
    test_dir = os.path.join(args.data_dir, "test")
    tr_paths = {i: os.path.join(train_dir, f"train_source{i}.tsv") for i in (1, 2, 3)}
    te_paths = {i: os.path.join(test_dir, f"test_source{i}.tsv") for i in (1, 2, 3)}
    gt_path = os.path.join(train_dir, "train_ground_truth.tsv")

    # ------------------------------------------------------------ 1. row counts
    section("1. ROW COUNTS (fast line count)")
    counts = {}
    all_files = ([(f"train_source{i}", tr_paths[i]) for i in (1, 2, 3)]
                 + [("train_ground_truth", gt_path)]
                 + [(f"test_source{i}", te_paths[i]) for i in (1, 2, 3)])
    for label, path in all_files:
        counts[label] = count_rows(path)
        print(f"  {label:<20} {counts[label]:>12,} rows   "
              f"({os.path.getsize(path) / 1e6:,.0f} MB)")

    # --------------------------------------------------------- 2. train sources
    section("2. TRAIN SOURCES")
    tr = {}
    for i in (1, 2, 3):
        log(f"loading train_source{i} ...")
        tr[i] = load_source(tr_paths[i])
        describe_source(f"train_source{i}", tr[i], counts[f"train_source{i}"])

    # ------------------------------------------------ 3. test (country only)
    section("3. TEST SOURCES (country column only, to save RAM)")
    for i in (1, 2, 3):
        c = pd.read_csv(te_paths[i], sep="\t", dtype=str, keep_default_na=False,
                        usecols=["country"])["country"]
        print(f"\n--- test_source{i}: {len(c):,} rows ---")
        print(c.value_counts().to_string())
        del c
        gc.collect()

    # --------------------------------------------------------- 4. ground truth
    section("4. GROUND TRUTH")
    log("loading ground truth ...")
    gt = pd.read_csv(gt_path, sep="\t", dtype=str, keep_default_na=False)
    ids = gt["matched_entity_ids"].str.strip()
    gt["n_matches"] = np.where(ids == "", 0, ids.str.count(",") + 1)

    n = len(gt)
    singletons = int((gt["n_matches"] == 0).sum())
    print(f"S1 entities in ground truth: {n:,} (train_source1 has {len(tr[1]):,})")
    in_s1 = int(gt["source1_entity_id"].isin(tr[1].index).sum())
    print(f"GT S1 ids found in train_source1: {in_s1:,}/{n:,}")
    print(f"Singletons (no matches): {singletons:,} ({singletons / n:.1%})")

    vc = gt["n_matches"].value_counts().sort_index()
    print("\nNumber of matches per S1 entity (up to 15):")
    print(vc[vc.index <= 15].to_string())
    print(f"  >15 matches: {int(vc[vc.index > 15].sum()):,}   max: {int(gt['n_matches'].max())}")

    log("exploding match lists ...")
    links = pd.DataFrame({"s1": gt["source1_entity_id"],
                          "m": ids.str.split(",")}).explode("m")
    links["m"] = links["m"].str.strip()
    links = links[links["m"].notna() & (links["m"] != "")].reset_index(drop=True)
    is_s2 = links["m"].str.startswith("S2-").to_numpy()
    print(f"\nTotal links: {len(links):,}  (S2: {int(is_s2.sum()):,}, "
          f"S3: {int((~is_s2).sum()):,})")

    for i in (2, 3):
        matched = int(tr[i].index.isin(links["m"]).sum())
        print(f"train_source{i}: {matched:,}/{len(tr[i]):,} records matched to some S1 "
              f"({matched / max(len(tr[i]), 1):.1%})")
    link_counts = links["m"].value_counts()
    print(f"S2/S3 records linked to more than one S1: {int((link_counts > 1).sum()):,}")
    known = links["m"].isin(tr[2].index) | links["m"].isin(tr[3].index)
    print(f"Linked ids missing from S2/S3 files: {int((~known).sum()):,}")

    # --------------------------------------------------------- 5. countries
    section("5. COUNTRIES: DO MATCHES CROSS COUNTRIES? SINGLETON RATE PER COUNTRY")
    c1 = tr[1]["country"].astype(str)
    links["c1"] = links["s1"].map(c1)
    links["c2"] = np.where(is_s2,
                           links["m"].map(tr[2]["country"].astype(str)),
                           links["m"].map(tr[3]["country"].astype(str)))
    print("Rows = S1 country, columns = matched record country:")
    print(pd.crosstab(links["c1"], links["c2"]).to_string())

    gt["country"] = gt["source1_entity_id"].map(c1)
    per_country = gt.groupby("country").agg(
        entities=("n_matches", "size"),
        singleton_rate=("n_matches", lambda x: (x == 0).mean()),
        avg_matches=("n_matches", "mean"))
    print("\n" + per_country.to_string())

    # ------------------------------------------------- 6. how hard is it
    section("6. HOW HARD IS IT? (on a sample of up to 200k true links)")
    sample = links.sample(min(200_000, len(links)), random_state=args.seed)
    s_is_s2 = sample["m"].str.startswith("S2-").to_numpy()
    for col, label in [("business_name", "name"), ("business_address", "address")]:
        left = normalize(sample["s1"].map(tr[1][col]))
        right = pd.Series(np.where(s_is_s2,
                                   sample["m"].map(tr[2][col]),
                                   sample["m"].map(tr[3][col])),
                          index=sample.index)
        right = normalize(right)
        sample[f"{label}_eq"] = (left == right)
        print(f"Exact {label} match after light cleaning: {sample[f'{label}_eq'].mean():.1%}")
    both = (sample["name_eq"] & sample["address_eq"]).mean()
    print(f"Both name AND address exact: {both:.1%}")

    # --------------------------------------------------------- 7. examples
    section("7. MATCHED EXAMPLES (side by side)")
    matched_rows = gt[gt["n_matches"] > 0].sample(
        min(args.examples, int((gt["n_matches"] > 0).sum())), random_state=args.seed)
    for _, row in matched_rows.iterrows():
        s1 = row["source1_entity_id"]
        rec = lookup(tr, s1)
        if rec is None:
            continue
        print(f"\n[{s1}] {rec[2]}")
        print(f"  S1 name: {rec[0]}")
        print(f"  S1 addr: {rec[1]}")
        for m in [x.strip() for x in row["matched_entity_ids"].split(",") if x.strip()]:
            r = lookup(tr, m)
            if r:
                print(f"  -> {m} name: {r[0]}")
                print(f"  -> {m} addr: {r[1]}")

    section("8. SINGLETON EXAMPLES (no matches)")
    single_rows = gt[gt["n_matches"] == 0]
    for _, row in single_rows.sample(min(4, len(single_rows)), random_state=args.seed).iterrows():
        rec = lookup(tr, row["source1_entity_id"])
        if rec:
            print(f"[{row['source1_entity_id']}] {rec[2]} | {rec[0]} | {rec[1]}")

    log("DONE - paste sections 1 to 6 back to Claude")


if __name__ == "__main__":
    main()
