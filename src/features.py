"""
features.py
Similarity features for (Source 1 record, candidate record) pairs.

Every pair gets numbers describing how alike the two records are: name
similarity, address similarity, shared house numbers, same postcode / state,
which blocking passes found it, and how it compares with the other candidates
of the same Source 1 record. The model (train_predict.py) learns from these.

Country is deliberately NOT a feature: France has no training data, so the
model must rely on similarity signals that work for any country.
"""

import os

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler

TEXT_COLS = ["name_norm", "name_core", "addr_norm", "addr_nums", "postcode", "state"]

FEATURES = [
    # names
    "name_ratio", "name_tsort", "name_tset", "name_partial", "name_jw",
    "name_full_tset", "name_exact", "name_tok_jacc", "name_first_eq",
    "name_len_diff", "name_ntok_a", "name_ntok_b",
    # addresses
    "addr_ratio", "addr_tsort", "addr_tset", "addr_partial",
    "addr_empty_a", "addr_empty_b",
    # numbers / postcode / state
    "num_jacc", "num_common", "num_long_eq", "num_both", "num_conflict",
    "pc_eq", "pc_conflict", "state_eq", "state_conflict",
    # blocking + source
    "pass_name_pair", "pass_name_rare", "pass_name_prefix", "pass_addr",
    "pass_postcode", "n_passes", "cand_is_s3",
    # context within the Source 1 record's candidate list
    "n_cands", "name_tset_rank", "name_tset_gap", "addr_tset_rank",
    "addr_tset_gap", "combo_rank", "combo_gap",
]


# ============================================================ text lookup

class TextStore:
    """Holds normalized text for one split so pairs can be joined to it fast."""

    def __init__(self, norm_dir, split):
        cols = ["entity_id"] + TEXT_COLS
        self.s1 = pd.read_parquet(os.path.join(norm_dir, f"{split}_source1.parquet"),
                                  columns=cols + ["country"]).set_index("entity_id")
        self.s23 = pd.concat(
            [pd.read_parquet(os.path.join(norm_dir, f"{split}_source{i}.parquet"),
                             columns=cols) for i in (2, 3)],
            ignore_index=True).set_index("entity_id")

    def attach(self, pairs):
        """Adds a_<col> (Source 1 side) and b_<col> (candidate side) columns."""
        i1 = self.s1.index.get_indexer(pairs["s1_entity_id"])
        i2 = self.s23.index.get_indexer(pairs["candidate_entity_id"])
        if (i1 < 0).any() or (i2 < 0).any():
            raise ValueError("some pair ids are missing from the normalized files")
        for col in TEXT_COLS:
            pairs["a_" + col] = self.s1[col].to_numpy()[i1]
            pairs["b_" + col] = self.s23[col].to_numpy()[i2]
        return pairs


# ============================================================ helpers

def _pairwise(a, b, scorer):
    """Element-wise similarity of two equal-length string lists (C++ speed)."""
    return process.cpdist(a, b, scorer=scorer, workers=-1).astype(np.float32)


def _set_features(a_names, b_names, a_nums, b_nums):
    """Features that need Python sets, computed in one pass."""
    n = len(a_names)
    tok_jacc = np.zeros(n, np.float32)
    first_eq = np.zeros(n, np.int8)
    num_jacc = np.zeros(n, np.float32)
    num_common = np.zeros(n, np.int16)
    num_long_eq = np.zeros(n, np.int8)
    num_both = np.zeros(n, np.int8)
    for i in range(n):
        ta, tb = a_names[i].split(), b_names[i].split()
        if ta and tb:
            sa, sb = set(ta), set(tb)
            tok_jacc[i] = len(sa & sb) / len(sa | sb)
            first_eq[i] = ta[0] == tb[0]
        na, nb = a_nums[i].split(), b_nums[i].split()
        if na and nb:
            num_both[i] = 1
            sa, sb = set(na), set(nb)
            common = len(sa & sb)
            num_common[i] = common
            num_jacc[i] = common / len(sa | sb)
            num_long_eq[i] = max(na, key=len) == max(nb, key=len)
    return tok_jacc, first_eq, num_jacc, num_common, num_long_eq, num_both


def _group_rank(s1_ids, values):
    """Rank (1 = best) and gap to the best value within each Source 1 record."""
    df = pd.DataFrame({"g": s1_ids, "v": values})
    rank = df.groupby("g")["v"].rank(ascending=False, method="min").to_numpy(np.float32)
    gap = (df.groupby("g")["v"].transform("max") - df["v"]).to_numpy(np.float32)
    return rank, gap


# ============================================================ main function

def compute_features(pairs):
    """pairs must have s1_entity_id, candidate_entity_id, passes, n_passes and the
    a_/b_ text columns from TextStore.attach. Returns a float32 DataFrame."""
    f = {}
    an, bn = pairs["a_name_core"].tolist(), pairs["b_name_core"].tolist()
    f["name_ratio"] = _pairwise(an, bn, fuzz.ratio)
    f["name_tsort"] = _pairwise(an, bn, fuzz.token_sort_ratio)
    f["name_tset"] = _pairwise(an, bn, fuzz.token_set_ratio)
    f["name_partial"] = _pairwise(an, bn, fuzz.partial_ratio)
    f["name_jw"] = _pairwise(an, bn, JaroWinkler.normalized_similarity)
    f["name_full_tset"] = _pairwise(pairs["a_name_norm"].tolist(),
                                    pairs["b_name_norm"].tolist(), fuzz.token_set_ratio)
    f["name_exact"] = (pairs["a_name_core"].to_numpy() == pairs["b_name_core"].to_numpy())
    la = pairs["a_name_core"].str.len().to_numpy()
    lb = pairs["b_name_core"].str.len().to_numpy()
    f["name_len_diff"] = np.abs(la - lb)
    f["name_ntok_a"] = pairs["a_name_core"].str.count(" ").to_numpy() + 1
    f["name_ntok_b"] = pairs["b_name_core"].str.count(" ").to_numpy() + 1

    aa, ba = pairs["a_addr_norm"].tolist(), pairs["b_addr_norm"].tolist()
    f["addr_ratio"] = _pairwise(aa, ba, fuzz.ratio)
    f["addr_tsort"] = _pairwise(aa, ba, fuzz.token_sort_ratio)
    f["addr_tset"] = _pairwise(aa, ba, fuzz.token_set_ratio)
    f["addr_partial"] = _pairwise(aa, ba, fuzz.partial_ratio)
    f["addr_empty_a"] = pairs["a_addr_norm"].to_numpy() == ""
    f["addr_empty_b"] = pairs["b_addr_norm"].to_numpy() == ""

    (f["name_tok_jacc"], f["name_first_eq"], f["num_jacc"], f["num_common"],
     f["num_long_eq"], f["num_both"]) = _set_features(
        an, bn, pairs["a_addr_nums"].tolist(), pairs["b_addr_nums"].tolist())
    f["num_conflict"] = (f["num_both"] == 1) & (f["num_common"] == 0)

    apc, bpc = pairs["a_postcode"].to_numpy(), pairs["b_postcode"].to_numpy()
    both_pc = (apc != "") & (bpc != "")
    f["pc_eq"] = both_pc & (apc == bpc)
    f["pc_conflict"] = both_pc & (apc != bpc)
    ast, bst = pairs["a_state"].to_numpy(), pairs["b_state"].to_numpy()
    both_st = (ast != "") & (bst != "")
    f["state_eq"] = both_st & (ast == bst)
    f["state_conflict"] = both_st & (ast != bst)

    passes = pairs["passes"].to_numpy().astype(np.int16)
    for name, bit in [("pass_name_pair", 1), ("pass_name_rare", 2),
                      ("pass_name_prefix", 4), ("pass_addr", 8), ("pass_postcode", 16)]:
        f[name] = (passes & bit) > 0
    f["n_passes"] = pairs["n_passes"].to_numpy()
    f["cand_is_s3"] = pairs["candidate_entity_id"].str.startswith("S3-").to_numpy()

    s1 = pairs["s1_entity_id"].to_numpy()
    f["n_cands"] = pd.Series(s1).map(pd.Series(s1).value_counts()).to_numpy()
    f["name_tset_rank"], f["name_tset_gap"] = _group_rank(s1, f["name_tset"])
    f["addr_tset_rank"], f["addr_tset_gap"] = _group_rank(s1, f["addr_tset"])
    combo = f["name_tset"] + f["addr_tset"]
    f["combo_rank"], f["combo_gap"] = _group_rank(s1, combo)

    out = pd.DataFrame({k: np.asarray(f[k]).astype(np.float32) for k in FEATURES})
    return out
