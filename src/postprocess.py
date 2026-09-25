"""
postprocess.py
Turns scored candidate pairs into final matches, and scores them with the
challenge metric (macro F0.5 per Source 1 entity, singletons included).

select_matches() rules, applied in this order:
  1. drop pairs below the threshold (or below top1_threshold, see 3)
  2. one-owner rule: a Source 2/3 record can belong to only ONE Source 1
     entity (true in all training data), so it goes to its highest score
  3. optional: an entity whose best pair scores >= top1_threshold keeps that
     one pair even when it is below the main threshold (fewer empty lists)
"""

import numpy as np
import pandas as pd


def select_matches(scores, threshold, one_owner=True, top1_threshold=None):
    """scores: s1_entity_id, candidate_entity_id, score. Returns kept pairs."""
    scores = scores.reset_index(drop=True)
    floor = threshold if top1_threshold is None else min(threshold, top1_threshold)
    df = scores[scores["score"] >= floor]
    if one_owner:
        df = df.sort_values("score", ascending=False).drop_duplicates("candidate_entity_id")
    keep = df["score"].to_numpy() >= threshold
    if top1_threshold is not None and len(df):
        best_idx = df.groupby("s1_entity_id")["score"].idxmax()
        keep |= df.index.isin(best_idx).astype(bool)
    return df[keep][["s1_entity_id", "candidate_entity_id", "score"]]


def f05_macro(matches, links, entity_ids):
    """Challenge metric.
    matches: predicted s1_entity_id, candidate_entity_id
    links:   true s1_entity_id / source1_entity_id, matched id
    entity_ids: every Source 1 id being evaluated (including singletons and
                entities with no candidates)."""
    links = links.rename(columns={"source1_entity_id": "s1_entity_id",
                                  "matched_entity_id": "candidate_entity_id"})
    ids = pd.Index(pd.unique(np.asarray(entity_ids)))
    pred_n = matches.groupby("s1_entity_id").size().reindex(ids, fill_value=0).to_numpy()
    true_n = links.groupby("s1_entity_id").size().reindex(ids, fill_value=0).to_numpy()
    tp = (matches.merge(links, on=["s1_entity_id", "candidate_entity_id"])
                 .groupby("s1_entity_id").size().reindex(ids, fill_value=0).to_numpy())

    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.where(pred_n > 0, tp / np.maximum(pred_n, 1), 0.0)
        r = np.where(true_n > 0, tp / np.maximum(true_n, 1), 0.0)
        f = np.where(tp > 0, 1.25 * p * r / (0.25 * p + r), 0.0)
    f = np.where((true_n == 0) & (pred_n == 0), 1.0, f)   # correct singleton
    return float(f.mean()), float(p[pred_n > 0].mean() if (pred_n > 0).any() else 0), \
        float(r[true_n > 0].mean() if (true_n > 0).any() else 0)


def tune(scores, links, entity_ids, thresholds=None, top1_options=(None, 0.1, 0.2, 0.3)):
    """Grid search on validation. Returns (best_params, results DataFrame)."""
    thresholds = thresholds if thresholds is not None else np.round(np.arange(0.2, 0.96, 0.05), 2)
    rows = []
    for one_owner in (True, False):
        for t in thresholds:
            for t1 in top1_options:
                if t1 is not None and t1 >= t:
                    continue
                m = select_matches(scores, t, one_owner, t1)
                f, p, r = f05_macro(m, links, entity_ids)
                rows.append({"threshold": t, "one_owner": one_owner, "top1_threshold": t1,
                             "f05": f, "precision": p, "recall": r,
                             "avg_pred": len(m) / len(entity_ids)})
    res = pd.DataFrame(rows).sort_values("f05", ascending=False).reset_index(drop=True)
    best = res.iloc[0]
    params = {"threshold": float(best["threshold"]), "one_owner": bool(best["one_owner"]),
              "top1_threshold": None if pd.isna(best["top1_threshold"])
              else float(best["top1_threshold"])}
    return params, res
