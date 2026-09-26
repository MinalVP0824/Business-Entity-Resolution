"""
analyze_misses.py
Which true matches does blocking miss, and why?

Loads a labelled blocking output for validation entities (from blocking.py
--split train), finds true links that are NOT among the candidates, and
reports what those missed pairs look like: name / address similarity, shared
numbers, shared words, and side-by-side examples per country.

Usage (Kaggle, after blocking.py --split train --sample 10000 --tag base):
    python src/analyze_misses.py --cands /kaggle/working/work/blocking/train_base_s10000
"""

import argparse
import glob
import os

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process

COLS = ["entity_id", "country", "name_core", "addr_norm", "addr_nums", "postcode"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", default="/kaggle/working/work")
    parser.add_argument("--cands", required=True, help="blocking output folder (train split)")
    parser.add_argument("--examples", type=int, default=12)
    args = parser.parse_args()

    norm = os.path.join(args.work_dir, "norm")
    cands = pd.concat([pd.read_parquet(f, columns=["s1_entity_id", "candidate_entity_id"])
                       for f in sorted(glob.glob(os.path.join(args.cands, "part-*.parquet")))])
    queried = set(cands["s1_entity_id"])
    links = pd.read_parquet(os.path.join(args.work_dir, "splits", "gt_links.parquet"),
                            columns=["source1_entity_id", "matched_entity_id"])
    links = links[links["source1_entity_id"].isin(queried)]
    # note: entities with zero candidates are missing from `queried`; they are
    # rare (about 0.3%) and do not change the picture.

    found = links.merge(cands, left_on=["source1_entity_id", "matched_entity_id"],
                        right_on=["s1_entity_id", "candidate_entity_id"], how="left")
    missed = links[found["candidate_entity_id"].isna().to_numpy()].copy()
    print(f"true links: {len(links):,}   missed by blocking: {len(missed):,} "
          f"({len(missed) / len(links):.1%})")

    s1 = pd.read_parquet(os.path.join(norm, "train_source1.parquet"), columns=COLS) \
        .set_index("entity_id")
    s23 = pd.concat([pd.read_parquet(os.path.join(norm, f"train_source{i}.parquet"), columns=COLS)
                     for i in (2, 3)]).set_index("entity_id")
    a = s1.loc[missed["source1_entity_id"]].reset_index(drop=True)
    b = s23.loc[missed["matched_entity_id"]].reset_index(drop=True)

    def sim(col, scorer):
        return process.cpdist(a[col].tolist(), b[col].tolist(), scorer=scorer, workers=-1)

    m = pd.DataFrame({
        "country": a["country"].to_numpy(),
        "name_tset": sim("name_core", fuzz.token_set_ratio),
        "addr_tset": sim("addr_norm", fuzz.token_set_ratio),
    })
    na, nb = a["addr_nums"].str.split(), b["addr_nums"].str.split()
    m["a_has_num"] = na.str.len() > 0
    m["b_has_num"] = nb.str.len() > 0
    m["share_num"] = [bool(set(x) & set(y)) for x, y in zip(na, nb)]
    m["share_name_word"] = [bool(set(x.split()) & set(y.split()))
                            for x, y in zip(a["name_core"], b["name_core"])]
    m["b_addr_empty"] = b["addr_norm"].to_numpy() == ""

    print("\nMissed links by country:")
    print(m["country"].value_counts().to_string())
    print("\nShare of missed links where ...")
    for col, label in [("share_name_word", "they share at least one core-name word"),
                       ("share_num", "they share a house/flat number"),
                       ("b_has_num", "the candidate has any number"),
                       ("b_addr_empty", "the candidate's address is empty")]:
        print(f"  {label:<45} {m[col].mean():6.1%}")
    print(f"  {'name token-set similarity >= 80':<45} {(m['name_tset'] >= 80).mean():6.1%}")
    print(f"  {'address token-set similarity >= 80':<45} {(m['addr_tset'] >= 80).mean():6.1%}")
    print(f"  {'BOTH similarities < 50':<45} "
          f"{((m['name_tset'] < 50) & (m['addr_tset'] < 50)).mean():6.1%}")

    print("\nBy country (median similarities):")
    print(m.groupby("country")[["name_tset", "addr_tset"]].median().round(0).to_string())

    rng = np.random.default_rng(0)
    for country in sorted(m["country"].unique()):
        idx = np.flatnonzero(m["country"].to_numpy() == country)
        pick = rng.choice(idx, size=min(args.examples, len(idx)), replace=False)
        print(f"\n===== MISSED EXAMPLES: {country} =====")
        for i in pick:
            print(f"S1  name: {a.at[i, 'name_core']!r:45} addr: {a.at[i, 'addr_norm']!r}")
            print(f"S23 name: {b.at[i, 'name_core']!r:45} addr: {b.at[i, 'addr_norm']!r}")
            print(f"    name_tset {m.at[i, 'name_tset']:.0f}  addr_tset {m.at[i, 'addr_tset']:.0f}  "
                  f"share_num {m.at[i, 'share_num']}\n")


if __name__ == "__main__":
    main()
