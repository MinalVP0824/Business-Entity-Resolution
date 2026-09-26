"""
blocking.py
Candidate generation (blocking) for Business Entity Resolution.

For every Source 1 record it finds a shortlist of Source 2 / Source 3 records
that could be the same business, using cheap "key" passes. Two records
become a candidate pair when they share at least one key. All keys include the
country, so records are only ever compared within the same country.

Passes (bit value in the `passes` column):
    1  name_pair    the two rarest words of the core name        (word order / suffix noise)
    2  name_rare    any single rare word of the core name        (very distinctive names)
    4  name_prefix  4-letter prefixes of the two rarest words    (typos after the 4th letter)
    8  addr         house/flat number + a rare address word      (name changed, address same)
    16 postcode     postcode + rare name word, postcode + number (India PIN, FR/US codes)
    32 addr_pair    pairs of rare address words, no number       (house number missing)
    64 addr_num_all house number + any address word              (truncated addresses)
    128 name_nospace whole name with spaces removed, and its first 8 letters
                    ("eye care associates" = "eyecare associates"; typos late in the name)
    256 addr_prefix house number + 5-letter prefix of a rare address word, and
                    pairs of such prefixes (typos in street / area names)
    (128 and 256 only with --typo-passes)
    (the Source 1 side uses more address words than the index side, so a long
     address can still meet a short version of itself; --no-new-passes = v1)

Keys shared by more than --max-block index records are dropped as too common.

Usage (on Kaggle, after make_split.py and normalize.py):
    # validation: recall report for the held-out Source 1 entities
    python src/blocking.py --split train

    # quick experiment on 10k validation entities
    python src/blocking.py --split train --sample 10000

    # test set (all Source 1 test entities, processed in chunks)
    python src/blocking.py --split test

Outputs:
    <work-dir>/blocking/<split>_<tag>/part-*.parquet
        s1_entity_id, candidate_entity_id, passes, n_passes[, label (train only)]
"""

import argparse
import os
import time
from collections import Counter

import numpy as np
import pandas as pd

START = time.time()

PASS_BITS = {"name_pair": 1, "name_rare": 2, "name_prefix": 4, "addr": 8, "postcode": 16,
             "addr_pair": 32, "addr_num_all": 64, "name_nospace": 128, "addr_prefix": 256}
POPCOUNT = np.array([bin(i).count("1") for i in range(512)], dtype=np.int8)

# Address words too common to identify a place on their own.
GENERIC_ADDR = {
    "street", "road", "avenue", "boulevard", "drive", "lane", "circle", "court",
    "place", "parkway", "highway", "square", "terrace", "trail", "crescent",
    "floor", "flat", "near", "opposite", "apartment", "building", "suite",
    "unit", "room", "house", "city", "district", "village", "taluka", "nagar",
    "colony", "sector", "phase", "main", "cross", "block", "plot", "shop",
    "north", "south", "east", "west", "the", "and", "rue", "chemin", "route",
    "saint", "des", "del", "office", "post", "market", "station", "gali",
}

INDEX_COLS = ["entity_id", "country", "name_core", "addr_norm", "addr_nums", "postcode"]


def log(msg):
    print(f"[{time.time() - START:7.1f}s] {msg}", flush=True)


# ============================================================ frequencies

def count_frequencies(df):
    """Document frequency of name words, name prefixes and address words,
    per country, counted on the index side (Source 2 + Source 3)."""
    name_df, prefix_df, addr_df = Counter(), Counter(), Counter()
    for c, name, addr in zip(df["country"], df["name_core"], df["addr_norm"]):
        toks = set(name.split())
        for t in toks:
            name_df[c + "|" + t] += 1
        for p in {t[:4] for t in toks if len(t) >= 3}:
            prefix_df[c + "|" + p] += 1
        for t in set(addr.split()):
            addr_df[c + "|" + t] += 1
    return name_df, prefix_df, addr_df


# ================================================================== keys

def record_keys(c, name_core, addr_norm, addr_nums, postcode, freqs, cfg, query=False):
    """Returns (keys, bits) for one record.
    query=True (Source 1 side) generates more address keys than the index side,
    so a long, complete address can still meet a short, truncated one."""
    name_df, prefix_df, addr_df = freqs
    keys, bits = [], []

    # ---- name words: rarest first, ignoring words the index never contains
    toks = [t for t in set(name_core.split()) if len(t) >= 2]
    known = sorted((t for t in toks if name_df.get(c + "|" + t, 0) > 0),
                   key=lambda t: (name_df[c + "|" + t], t))
    rare = known[:2]
    if rare:
        keys.append("np|" + c + "|" + "|".join(sorted(rare)))
        bits.append(1)
    for t in known[:3]:
        if name_df[c + "|" + t] <= cfg.rare_df:
            keys.append("nr|" + c + "|" + t)
            bits.append(2)

    # ---- name prefixes (typo tolerant)
    prefixes = {t[:4] for t in toks if len(t) >= 3}
    known_p = sorted((p for p in prefixes if prefix_df.get(c + "|" + p, 0) > 0),
                     key=lambda p: (prefix_df[c + "|" + p], p))[:2]
    if len(known_p) == 2:
        keys.append("pp|" + c + "|" + "|".join(sorted(known_p)))
        bits.append(4)

    # ---- address: number + rare address word
    nums = sorted(addr_nums.split(), key=lambda n: (-len(n), n))[:cfg.max_nums]
    words = {t for t in addr_norm.split()
             if len(t) >= 3 and not t.isdigit() and t not in GENERIC_ADDR}
    all_words = sorted((t for t in words if addr_df.get(c + "|" + t, 0) > 0),
                       key=lambda t: (addr_df[c + "|" + t], t))
    words = all_words[:2]
    for n in nums:
        for t in words:
            keys.append("ad|" + c + "|" + n + "|" + t)
            bits.append(8)

    if not cfg.no_new_passes:
        # ---- address word pairs, no number needed (house number missing on
        #      one side). Pairs among the rarest words; the query side uses
        #      more words so its pairs cover a shorter version of the address.
        k = cfg.pair_words_query if query else cfg.pair_words_index
        top = all_words[:k]
        for a in range(len(top)):
            for b in range(a + 1, len(top)):
                x, y = (top[a], top[b]) if top[a] < top[b] else (top[b], top[a])
                keys.append("aw|" + c + "|" + x + "|" + y)
                bits.append(32)

        # ---- number + ANY address word (not only the two rarest): catches
        #      truncated addresses such as "79 ahmedabad gj" vs a long version.
        k = cfg.num_words_query if query else cfg.num_words_index
        for n in nums[:2]:
            for t in all_words[2:k]:
                keys.append("ad|" + c + "|" + n + "|" + t)
                bits.append(64)

    if cfg.typo_passes:
        # ---- name with spaces removed: glued / split words, and typos that
        #      come after the 8th letter
        glued = name_core.replace(" ", "")
        if len(glued) >= 6:
            keys.append("ns|" + c + "|" + glued)
            bits.append(128)
        if len(glued) >= 10:
            keys.append("n8|" + c + "|" + glued[:8])
            bits.append(128)

        # ---- 5-letter prefixes of the rarest address words. Words the index
        #      has never seen count as rarest (df 0): that is usually the word
        #      with the typo, and its prefix is often still correct.
        raw = {t for t in addr_norm.split()
               if len(t) >= 5 and not t.isdigit() and t not in GENERIC_ADDR}
        pw = sorted(raw, key=lambda t: (addr_df.get(c + "|" + t, 0), t))
        pref = sorted({t[:5] for t in pw[:cfg.prefix_words]})
        for n in nums[:2]:
            for p in pref:
                keys.append("ax|" + c + "|" + n + "|" + p)
                bits.append(256)
        for a in range(len(pref)):
            for b in range(a + 1, len(pref)):
                keys.append("aq|" + c + "|" + pref[a] + "|" + pref[b])
                bits.append(256)

    # ---- postcode
    if postcode:
        if rare:
            keys.append("pc|" + c + "|" + postcode + "|" + rare[0])
            bits.append(16)
        for n in nums[:2]:
            keys.append("pn|" + c + "|" + postcode + "|" + n)
            bits.append(16)

    return keys, bits


def build_keys(df, freqs, cfg, chunk=500_000, label="", query=False):
    """Returns hashed keys (uint64), row ids (int32) and pass bits (int16)."""
    key_parts, row_parts, bit_parts = [], [], []
    cols = [df[c].tolist() for c in ["country", "name_core", "addr_norm", "addr_nums", "postcode"]]
    n = len(df)
    for start in range(0, n, chunk):
        ks, rs, bs = [], [], []
        stop = min(start + chunk, n)
        for j in range(start, stop):
            k, b = record_keys(cols[0][j], cols[1][j], cols[2][j], cols[3][j],
                               cols[4][j], freqs, cfg, query)
            ks.extend(k)
            bs.extend(b)
            rs.extend([j] * len(k))
        if ks:
            key_parts.append(pd.util.hash_array(np.array(ks, dtype=object)))
            row_parts.append(np.array(rs, dtype=np.int32))
            bit_parts.append(np.array(bs, dtype=np.int16))
        if label:
            log(f"  {label}: keys for {stop:,}/{n:,} records")
    if not key_parts:
        return (np.array([], dtype=np.uint64), np.array([], dtype=np.int32),
                np.array([], dtype=np.int16))
    return np.concatenate(key_parts), np.concatenate(row_parts), np.concatenate(bit_parts)


# ================================================================= index

class KeyIndex:
    """Sorted key -> record lookup for the Source 2 + Source 3 side."""

    def __init__(self, keys, rows, max_block):
        order = np.argsort(keys, kind="stable")
        keys, rows = keys[order], rows[order]
        _, first, counts = np.unique(keys, return_index=True, return_counts=True)
        keep = np.repeat(counts <= max_block, counts)
        self.dropped_keys = int((counts > max_block).sum())
        self.keys = keys[keep]
        self.rows = rows[keep]

    def lookup(self, qkeys, qrows, qbits):
        left = np.searchsorted(self.keys, qkeys, side="left")
        right = np.searchsorted(self.keys, qkeys, side="right")
        lens = (right - left).astype(np.int64)
        hit = lens > 0
        left, lens = left[hit], lens[hit]
        qrows, qbits = qrows[hit], qbits[hit]
        total = int(lens.sum())
        if total == 0:
            return (np.array([], dtype=np.int64), np.array([], dtype=np.int64),
                    np.array([], dtype=np.int16))
        starts = np.cumsum(lens) - lens
        offsets = np.arange(total, dtype=np.int64) - np.repeat(starts, lens)
        idx = np.repeat(left, lens) + offsets
        return (np.repeat(qrows, lens).astype(np.int64),
                self.rows[idx].astype(np.int64),
                np.repeat(qbits, lens))


# ============================================================= candidates

def combine_pairs(q, cand, bits, n_index, max_cands):
    """Unique (query, candidate) pairs with OR-ed pass bits, capped per query."""
    pairs = pd.DataFrame({"code": q * n_index + cand, "bit": bits})
    pairs = pairs.drop_duplicates()
    agg = pairs.groupby("code", sort=False)["bit"].sum().astype(np.int16)
    code = agg.index.to_numpy()
    out = pd.DataFrame({"q": code // n_index, "c": code % n_index,
                        "passes": agg.to_numpy()})
    out["n_passes"] = POPCOUNT[out["passes"].to_numpy()]
    before = len(out)
    if max_cands:
        out = out.sort_values(["q", "n_passes"], ascending=[True, False])
        out = out[out.groupby("q").cumcount() < max_cands]
    return out.reset_index(drop=True), before - len(out)


# ============================================================= evaluation

def f05_ceiling(found, true):
    """F0.5 if the matching model were perfect on the candidates."""
    rec = np.where(true > 0, found / np.maximum(true, 1), 1.0)
    f = np.where(true > 0, 1.25 * rec / (0.25 + rec), 1.0)
    return np.where((true > 0) & (found == 0), 0.0, f)


def evaluate(cands, queries, links):
    """cands: s1_entity_id, candidate_entity_id, passes; links: true pairs."""
    section = "=" * 78
    print(f"\n{section}\nBLOCKING REPORT\n{section}")
    n_q = len(queries)
    per_q = cands.groupby("s1_entity_id").size().reindex(queries["entity_id"], fill_value=0)
    print(f"Queries: {n_q:,}   candidate pairs: {len(cands):,}")
    print(f"Candidates per S1: mean {per_q.mean():.1f}, median {per_q.median():.0f}, "
          f"p90 {per_q.quantile(0.9):.0f}, p99 {per_q.quantile(0.99):.0f}, "
          f"max {per_q.max()}, zero {int((per_q == 0).sum()):,}")

    merged = links.merge(cands, how="left",
                         left_on=["source1_entity_id", "matched_entity_id"],
                         right_on=["s1_entity_id", "candidate_entity_id"])
    found = merged["passes"].notna()
    print(f"\nTrue links: {len(links):,}   found: {int(found.sum()):,}   "
          f"RECALL: {found.mean():.2%}")

    passes = merged["passes"].fillna(0).astype(int)
    print("\nPer pass (recall = true links this pass finds; only = found by no other pass):")
    for name, bit in PASS_BITS.items():
        has = (passes & bit) > 0
        only = passes == bit
        print(f"  {name:<12} recall {has.mean():6.2%}   only-this-pass {only.mean():6.2%}")

    pos_rate = found.sum() / max(len(cands), 1)
    print(f"\nPositives among candidates: {pos_rate:.2%} "
          f"(about 1 true match per {1 / max(pos_rate, 1e-9):.0f} candidates)")

    true_n = links.groupby("source1_entity_id").size()
    found_n = merged[found].groupby("source1_entity_id").size()
    ent = queries[["entity_id", "country"]].copy()
    ent["true"] = ent["entity_id"].map(true_n).fillna(0).to_numpy()
    ent["found"] = ent["entity_id"].map(found_n).fillna(0).to_numpy()
    ent["ceiling"] = f05_ceiling(ent["found"].to_numpy(), ent["true"].to_numpy())
    print(f"\nF0.5 CEILING (perfect model on these candidates): {ent['ceiling'].mean():.4f}")
    print("\nBy country:")
    by_c = ent.groupby("country").apply(
        lambda g: pd.Series({"entities": len(g),
                             "recall": g["found"].sum() / max(g["true"].sum(), 1),
                             "f05_ceiling": g["ceiling"].mean()}),
        include_groups=False)
    print(by_c.to_string())


# =================================================================== main

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", default="/kaggle/working/work")
    parser.add_argument("--split", choices=["train", "test"], required=True)
    parser.add_argument("--ids-file", default=None,
                        help="Source 1 ids to query (train default: splits/val_s1_ids.txt)")
    parser.add_argument("--sample", type=int, default=None,
                        help="randomly keep only this many query ids")
    parser.add_argument("--tag", default=None, help="output folder suffix")
    parser.add_argument("--rare-df", type=int, default=500)
    parser.add_argument("--max-block", type=int, default=200)
    parser.add_argument("--max-nums", type=int, default=3)
    parser.add_argument("--max-cands", type=int, default=200)
    parser.add_argument("--query-chunk", type=int, default=200_000)
    parser.add_argument("--no-new-passes", action="store_true",
                        help="use only the original five passes (v1 behaviour)")
    parser.add_argument("--pair-words-index", type=int, default=4)
    parser.add_argument("--pair-words-query", type=int, default=6)
    parser.add_argument("--num-words-index", type=int, default=8)
    parser.add_argument("--num-words-query", type=int, default=12)
    parser.add_argument("--typo-passes", action="store_true",
                        help="add passes 128 (name without spaces) and 256 (address "
                             "word prefixes)")
    parser.add_argument("--prefix-words", type=int, default=3,
                        help="rare address words whose prefixes the typo pass uses")
    parser.add_argument("--seed", type=int, default=2026)
    cfg = parser.parse_args()

    norm_dir = os.path.join(cfg.work_dir, "norm")
    tag = cfg.tag or ("val" if cfg.split == "train" and not cfg.ids_file else "all")
    if cfg.sample:
        tag += f"_s{cfg.sample}"
    out_dir = os.path.join(cfg.work_dir, "blocking", f"{cfg.split}_{tag}")
    os.makedirs(out_dir, exist_ok=True)
    for f in os.listdir(out_dir):
        if f.startswith("part-"):
            os.remove(os.path.join(out_dir, f))

    # ------------------------------------------------------------ index side
    log("loading Source 2 + Source 3 ...")
    index_df = pd.concat(
        [pd.read_parquet(os.path.join(norm_dir, f"{cfg.split}_source{i}.parquet"),
                         columns=INDEX_COLS) for i in (2, 3)],
        ignore_index=True)
    n_index = len(index_df)
    log(f"index records: {n_index:,}")

    log("counting word frequencies ...")
    freqs = count_frequencies(index_df)

    log("building index keys ...")
    ikeys, irows, _ = build_keys(index_df, freqs, cfg, chunk=1_000_000, label="index")
    index = KeyIndex(ikeys, irows, cfg.max_block)
    del ikeys, irows
    log(f"index keys kept: {len(index.keys):,} "
        f"(dropped {index.dropped_keys:,} keys shared by > {cfg.max_block} records)")
    index_ids = index_df["entity_id"].to_numpy()
    del index_df

    # ------------------------------------------------------------ query side
    queries = pd.read_parquet(os.path.join(norm_dir, f"{cfg.split}_source1.parquet"),
                              columns=INDEX_COLS)
    ids_file = cfg.ids_file
    if ids_file is None and cfg.split == "train":
        ids_file = os.path.join(cfg.work_dir, "splits", "val_s1_ids.txt")
    if ids_file:
        wanted = pd.read_csv(ids_file, header=None, dtype=str)[0]
        queries = queries[queries["entity_id"].isin(set(wanted))]
    if cfg.sample and cfg.sample < len(queries):
        queries = queries.sample(cfg.sample, random_state=cfg.seed)
    queries = queries.reset_index(drop=True)
    log(f"query (Source 1) records: {len(queries):,}")

    all_parts = []
    total_capped = 0
    for part_no, start in enumerate(range(0, len(queries), cfg.query_chunk)):
        chunk = queries.iloc[start:start + cfg.query_chunk].reset_index(drop=True)
        qkeys, qrows, qbits = build_keys(chunk, freqs, cfg, query=True)
        q, c, bits = index.lookup(qkeys, qrows, qbits)
        pairs, capped = combine_pairs(q, c, bits, n_index, cfg.max_cands)
        total_capped += capped
        out = pd.DataFrame({
            "s1_entity_id": chunk["entity_id"].to_numpy()[pairs["q"].to_numpy()],
            "candidate_entity_id": index_ids[pairs["c"].to_numpy()],
            "passes": pairs["passes"].to_numpy(),
            "n_passes": pairs["n_passes"].to_numpy(),
        })
        out.to_parquet(os.path.join(out_dir, f"part-{part_no:04d}.parquet"), index=False)
        if cfg.split == "train":
            all_parts.append(out)
        log(f"queries {min(start + cfg.query_chunk, len(queries)):,}/{len(queries):,}: "
            f"{len(out):,} candidate pairs")

    log(f"pairs removed by --max-cands cap: {total_capped:,}")
    log(f"saved to {out_dir}")

    # ------------------------------------------------------------ evaluation
    if cfg.split == "train":
        cands = pd.concat(all_parts, ignore_index=True)
        links = pd.read_parquet(os.path.join(cfg.work_dir, "splits", "gt_links.parquet"),
                                columns=["source1_entity_id", "matched_entity_id"])
        links = links[links["source1_entity_id"].isin(set(queries["entity_id"]))]

        # add the label column to the saved files for the model step
        true_set = set(zip(links["source1_entity_id"], links["matched_entity_id"]))
        for f in sorted(os.listdir(out_dir)):
            if f.startswith("part-"):
                path = os.path.join(out_dir, f)
                part = pd.read_parquet(path)
                part["label"] = [
                    (a, b) in true_set
                    for a, b in zip(part["s1_entity_id"], part["candidate_entity_id"])]
                part["label"] = part["label"].astype(np.int8)
                part.to_parquet(path, index=False)

        evaluate(cands, queries, links)
    log("done")


if __name__ == "__main__":
    main()
