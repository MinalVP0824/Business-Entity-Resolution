"""
stage2.py
Second-stage re-ranker features.

The first model judges each (Source 1, candidate) pair on its own. The true
matches of one business are all variants of the same record, so a candidate's
score is more informative when compared with the OTHER candidates of the same
Source 1 entity: is it the best one, how far below the best is it, how many
strong candidates does the entity have?

These features are computed from first-stage scores only, per Source 1 entity.
Candidate lists are always complete per entity (blocking never splits an
entity across parts), so the features are the same whether computed on all
pairs at once or part by part.

During training the first-stage scores must be OUT-OF-FOLD (each training pair
scored by a model that did not see it), otherwise the second model would learn
from over-confident scores. train_predict.py handles that with 2 folds.
"""

import numpy as np
import pandas as pd

SCORE_FEATURES = ["s1_score", "s1_rank", "s1_gap", "s1_rel", "s1_max", "s1_second",
                  "s1_n_ge50", "s1_n_ge20", "s1_sum", "s1_share"]


def fold_of(s1_ids, n_folds=2):
    """Deterministic fold per Source 1 entity (all its pairs in the same fold)."""
    h = pd.util.hash_array(np.asarray(s1_ids, dtype=object))
    return (h % n_folds).astype(np.int8)


def score_group_features(s1_ids, scores):
    """Features describing each pair's first-stage score relative to the other
    candidates of the same Source 1 entity. Returns a float32 DataFrame."""
    df = pd.DataFrame({"g": np.asarray(s1_ids), "s": np.asarray(scores, dtype=np.float64)})
    grp = df.groupby("g", sort=False)["s"]
    s_max = grp.transform("max")
    s_sum = grp.transform("sum")
    rank = grp.rank(ascending=False, method="min")
    # second best score of the entity (0 if only one candidate), vectorised:
    # sort by entity then score descending, take the 2nd row of each entity
    order = np.lexsort((-df["s"].to_numpy(), pd.factorize(df["g"])[0]))
    srt = df.iloc[order]
    pos = srt.groupby("g", sort=False).cumcount().to_numpy()
    sec = srt.loc[pos == 1].set_index("g")["s"]
    second = df["g"].map(sec).fillna(0.0)
    out = pd.DataFrame({
        "s1_score": df["s"],
        "s1_rank": rank,
        "s1_gap": s_max - df["s"],
        "s1_rel": df["s"] / s_max.clip(lower=1e-6),
        "s1_max": s_max,
        "s1_second": second,
        "s1_n_ge50": (df["s"] >= 0.5).groupby(df["g"], sort=False).transform("sum"),
        "s1_n_ge20": (df["s"] >= 0.2).groupby(df["g"], sort=False).transform("sum"),
        "s1_sum": s_sum,
        "s1_share": df["s"] / s_sum.clip(lower=1e-6),
    })
    return out[SCORE_FEATURES].astype(np.float32).reset_index(drop=True)
