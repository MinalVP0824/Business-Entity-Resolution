"""
postprocess.py
Turns scored candidate pairs into final matches, and scores them with the
challenge metric (macro F0.5 per Source 1 entity, singletons included).

select_matches() rules, applied in this order:
  1. drop pairs below the threshold for their country (or below
     top1_threshold, see 3). Countries without their own threshold (e.g.
     France, which has no training data) use the global threshold.
  2. one-owner rule: a Source 2/3 record can belong to only ONE Source 1
     entity (true in all training data), so it goes to its highest score
  3. optional: an entity whose best pair scores >= top1_threshold keeps that
     one pair even when it is below its threshold (fewer empty lists)

Per-country tuning is exact: F0.5 is averaged per entity and matches never
cross countries, so each country's threshold can be optimised on its own.

country_thresholds may also hold "<country>|S2" / "<country>|S3" keys: a
separate threshold for Source 2 and Source 3 candidates of that country
(tune_per_country_source). Lookup order per pair: country|source, country,
global.
"""

import numpy as np
import pandas as pd

DEFAULT_THRESHOLDS = np.round(np.arange(0.3, 0.931, 0.025), 3)


def _row_thresholds(scores, threshold, country_thresholds):
    if not country_thresholds or "country" not in scores.columns:
        return np.full(len(scores), threshold, dtype=np.float64)
    thr = scores["country"].map(country_thresholds)
    if any("|" in k for k in country_thresholds):
        src = scores["candidate_entity_id"].str[:2]            # "S2" / "S3"
        by_src = (scores["country"] + "|" + src).map(country_thresholds)
        thr = by_src.fillna(thr)
    return thr.fillna(threshold).to_numpy(np.float64)


def select_matches(scores, threshold, one_owner=True, top1_threshold=None,
                   country_thresholds=None):
    """scores: s1_entity_id, candidate_entity_id, score [, country].
    Returns the kept pairs."""
    scores = scores.reset_index(drop=True)
    thr = _row_thresholds(scores, threshold, country_thresholds)
    floor = thr if top1_threshold is None else np.minimum(thr, top1_threshold)
    keep_mask = scores["score"].to_numpy() >= floor
    df = scores[keep_mask]
    thr = thr[keep_mask]
    if one_owner and len(df):
        order = np.argsort(-df["score"].to_numpy(), kind="stable")
        df, thr = df.iloc[order], thr[order]
        first = ~df["candidate_entity_id"].duplicated().to_numpy()
        df, thr = df[first], thr[first]
    keep = df["score"].to_numpy() >= thr
    if top1_threshold is not None and len(df):
        best_idx = df.groupby("s1_entity_id")["score"].idxmax()
        keep |= df.index.isin(best_idx)
    return df[keep][["s1_entity_id", "candidate_entity_id", "score"]]


def f05_macro(matches, links, entity_ids):
    """Challenge metric.
    matches: predicted s1_entity_id, candidate_entity_id
    links:   true s1_entity_id / source1_entity_id, matched id
    entity_ids: every Source 1 id being evaluated (including singletons and
                entities with no candidates).
    Returns (F0.5, mean precision, mean recall)."""
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
    return (float(f.mean()),
            float(p[pred_n > 0].mean()) if (pred_n > 0).any() else 0.0,
            float(r[true_n > 0].mean()) if (true_n > 0).any() else 0.0)


def tune(scores, links, entity_ids, thresholds=None, top1_options=(None, 0.1, 0.2, 0.3),
         one_owner_options=(True,)):
    """Grid search of the global settings on validation.
    Returns (best_params, results DataFrame)."""
    thresholds = DEFAULT_THRESHOLDS if thresholds is None else thresholds
    rows = []
    for one_owner in one_owner_options:
        for t in thresholds:
            for t1 in top1_options:
                if t1 is not None and t1 >= t:
                    continue
                m = select_matches(scores, t, one_owner, t1)
                f, p, r = f05_macro(m, links, entity_ids)
                rows.append({"threshold": float(t), "one_owner": one_owner,
                             "top1_threshold": t1, "f05": f, "precision": p,
                             "recall": r, "avg_pred": len(m) / max(len(entity_ids), 1)})
    res = pd.DataFrame(rows).sort_values("f05", ascending=False).reset_index(drop=True)
    best = res.iloc[0]
    params = {"threshold": float(best["threshold"]), "one_owner": bool(best["one_owner"]),
              "top1_threshold": None if pd.isna(best["top1_threshold"])
              else float(best["top1_threshold"])}
    return params, res


def tune_per_country(scores, links, entity_ids, entity_country, params, thresholds=None):
    """Keeps the global one_owner / top1 settings and finds the best threshold
    for each country seen in validation.
    entity_country: Series mapping Source 1 id -> country (lowercase)."""
    thresholds = DEFAULT_THRESHOLDS if thresholds is None else thresholds
    ids = pd.Series(pd.unique(np.asarray(entity_ids)))
    countries = ids.map(entity_country)
    out, rows = {}, []
    for country in sorted(countries.dropna().unique()):
        c_ids = ids[countries == country]
        c_set = set(c_ids)
        c_scores = scores[scores["s1_entity_id"].isin(c_set)]
        c_links = links[links["source1_entity_id"].isin(c_set)]
        best_t, best_f = None, -1.0
        for t in thresholds:
            m = select_matches(c_scores, t, params["one_owner"], params["top1_threshold"])
            f, p, r = f05_macro(m, c_links, c_ids)
            rows.append({"country": country, "threshold": float(t), "f05": f,
                         "precision": p, "recall": r})
            if f > best_f:
                best_t, best_f = float(t), f
        out[country] = best_t
    return out, pd.DataFrame(rows)


def tune_per_country_source(scores, links, entity_ids, entity_country, params,
                            thresholds=None, rounds=2):
    """Separate thresholds for Source 2 and Source 3 candidates of each country.
    Starts from params["country_thresholds"] (or the global threshold) and does
    coordinate descent: best S2 threshold with S3 fixed, then S3 with S2 fixed.
    Returns ({country|S2: t, country|S3: t, ...}, results DataFrame)."""
    thresholds = DEFAULT_THRESHOLDS if thresholds is None else thresholds
    base = params.get("country_thresholds") or {}
    ids = pd.Series(pd.unique(np.asarray(entity_ids)))
    countries = ids.map(entity_country)
    out, rows = {}, []
    for country in sorted(countries.dropna().unique()):
        c_ids = ids[countries == country]
        c_set = set(c_ids)
        c_scores = scores[scores["s1_entity_id"].isin(c_set)].copy()
        c_scores["country"] = country
        c_links = links[links["source1_entity_id"].isin(c_set)]
        start = base.get(country, params["threshold"])
        cur = {country + "|S2": start, country + "|S3": start}

        def score_of(th):
            m = select_matches(c_scores, params["threshold"], params["one_owner"],
                               params["top1_threshold"], th)
            return f05_macro(m, c_links, c_ids)[0]

        best_f = score_of(cur)
        for _ in range(rounds):
            for key in (country + "|S2", country + "|S3"):
                for t in thresholds:
                    trial = dict(cur, **{key: float(t)})
                    f = score_of(trial)
                    rows.append({"country": country, "key": key, "threshold": float(t),
                                 "f05": f})
                    if f > best_f + 1e-9:
                        best_f, cur = f, trial
        out.update(cur)
    return out, pd.DataFrame(rows)
