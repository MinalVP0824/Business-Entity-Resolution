"""
stage2.py
Second-stage re-ranker features.

The first model judges each (Source 1, candidate) pair on its own. The true
matches of one business are all variants of the same record, so a candidate's
score is more informative when compared with the OTHER candidates of the same
Source 1 entity: is it the best one, how far below the best is it, how many
strong candidates does the entity have?

Two groups of features, both per Source 1 entity:
  SCORE_FEATURES       from first-stage scores only (rank, gap, share, ...)
  TRANSITIVE_FEATURES  (optional, --transitive) how similar the candidate is
                       to the entity's BEST candidate (for the best candidate:
                       to the second best). If A matches S1 and B looks just like
                       A, B probably matches too, even when B's own name or
                       address is written differently from the S1 record.

Candidate lists are always complete per entity (blocking never splits an
entity across parts), so the features are the same whether computed on all
pairs at once or part by part.

During training the first-stage scores must be OUT-OF-FOLD (each training pair
scored by a model that did not see it), otherwise the second model would learn
from over-confident scores. train_predict.py handles that with 2 folds.
"""

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process

SCORE_FEATURES = ["s1_score", "s1_rank", "s1_gap", "s1_rel", "s1_max", "s1_second",
                  "s1_n_ge50", "s1_n_ge20", "s1_sum", "s1_share"]

TRANSITIVE_FEATURES = ["t_name_tset", "t_name_ratio", "t_addr_tset", "t_addr_ratio",
                       "t_ref_score", "t_has_ref", "t_combo_x_ref"]


def fold_of(s1_ids, n_folds=2):
    """Deterministic fold per Source 1 entity (all its pairs in the same fold)."""
    h = pd.util.hash_array(np.asarray(s1_ids, dtype=object))
    return (h % n_folds).astype(np.int8)


def _best_and_second(s1_ids, scores):
    """For every row: position (in the input order) of its entity's best row
    and second-best row (-1 if the entity has only one candidate)."""
    g = pd.factorize(np.asarray(s1_ids))[0]
    s = np.asarray(scores, dtype=np.float64)
    order = np.lexsort((-s, g))                  # entity, then score descending
    g_sorted = g[order]
    start = np.r_[True, g_sorted[1:] != g_sorted[:-1]]
    first_pos = np.maximum.accumulate(np.where(start, np.arange(len(g)), 0))
    best_sorted = order[first_pos]
    second_idx = first_pos + 1
    has_second = (second_idx < len(g)) & (g_sorted[np.minimum(second_idx, len(g) - 1)]
                                           == g_sorted)
    second_sorted = np.where(has_second, order[np.minimum(second_idx, len(g) - 1)], -1)
    best = np.empty(len(g), np.int64)
    second = np.empty(len(g), np.int64)
    best[order] = best_sorted
    second[order] = second_sorted
    return best, second


def score_group_features(s1_ids, scores):
    """Features describing each pair's first-stage score relative to the other
    candidates of the same Source 1 entity. Returns a float32 DataFrame."""
    df = pd.DataFrame({"g": np.asarray(s1_ids), "s": np.asarray(scores, dtype=np.float64)})
    grp = df.groupby("g", sort=False)["s"]
    s_max = grp.transform("max")
    s_sum = grp.transform("sum")
    rank = grp.rank(ascending=False, method="min")
    _, second_pos = _best_and_second(df["g"].to_numpy(), df["s"].to_numpy())
    # second best score of the entity (0 if only one candidate)
    sec_score = np.where(second_pos >= 0, df["s"].to_numpy()[np.maximum(second_pos, 0)], 0.0)
    out = pd.DataFrame({
        "s1_score": df["s"],
        "s1_rank": rank,
        "s1_gap": s_max - df["s"],
        "s1_rel": df["s"] / s_max.clip(lower=1e-6),
        "s1_max": s_max,
        "s1_second": sec_score,
        "s1_n_ge50": (df["s"] >= 0.5).groupby(df["g"], sort=False).transform("sum"),
        "s1_n_ge20": (df["s"] >= 0.2).groupby(df["g"], sort=False).transform("sum"),
        "s1_sum": s_sum,
        "s1_share": df["s"] / s_sum.clip(lower=1e-6),
    })
    return out[SCORE_FEATURES].astype(np.float32).reset_index(drop=True)


def transitive_features(s1_ids, scores, b_name, b_addr):
    """Similarity of each candidate to its entity's reference candidate: the
    best-scored one, or the second best for the best candidate itself.
    b_name / b_addr: the candidates' normalized core name and address.
    Returns a float32 DataFrame (TRANSITIVE_FEATURES)."""
    s = np.asarray(scores, dtype=np.float64)
    best, second = _best_and_second(s1_ids, s)
    n = len(s)
    rows = np.arange(n)
    ref = np.where(best == rows, second, best)       # -1: no other candidate
    has = ref >= 0
    ref_safe = np.where(has, ref, rows)
    names = np.asarray(b_name, dtype=object)
    addrs = np.asarray(b_addr, dtype=object)
    ra, rn = addrs[ref_safe].tolist(), names[ref_safe].tolist()

    def sim(a, b, scorer):
        v = process.cpdist(a, b, scorer=scorer, workers=-1).astype(np.float32)
        return np.where(has, v, -1.0).astype(np.float32)

    out = pd.DataFrame({
        "t_name_tset": sim(names.tolist(), rn, fuzz.token_set_ratio),
        "t_name_ratio": sim(names.tolist(), rn, fuzz.ratio),
        "t_addr_tset": sim(addrs.tolist(), ra, fuzz.token_set_ratio),
        "t_addr_ratio": sim(addrs.tolist(), ra, fuzz.ratio),
        "t_ref_score": np.where(has, s[ref_safe], -1.0),
        "t_has_ref": has,
    })
    # strong reference AND close to it: the "friend of a match" signal
    close = (np.maximum(out["t_name_tset"], 0) + np.maximum(out["t_addr_tset"], 0)) / 200.0
    out["t_combo_x_ref"] = np.where(has, close * s[ref_safe], -1.0)
    return out[TRANSITIVE_FEATURES].astype(np.float32).reset_index(drop=True)
