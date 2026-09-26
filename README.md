# Business Entity Resolution: Amazon ML Challenge 2026

For every Source 1 business, find the Source 2 / Source 3 records that describe the same business.

Pipeline: `normalize → split → blocking → features → LightGBM → thresholds + one-owner rule → output`

## Setup

Python 3.10+.

```
pip install -r requirements.txt
```

`requirements.txt`: pandas, numpy, pyarrow, rapidfuzz, lightgbm.

The data is expected in the layout of the provided `student_resource/` folder:

```
student_resource/
├── dataset/
│   ├── train/  train_source1.tsv  train_source2.tsv  train_source3.tsv  train_ground_truth.tsv
│   └── test/   test_source1.tsv   test_source2.tsv   test_source3.tsv
└── utils/validate_submission.py
```

## Reproduce the submission

Run from this folder. Set `DATA` to the `student_resource` path. Intermediate files go to `work/` and the final files to `output/`.

```
DATA=/path/to/student_resource

# 1. normalize names and addresses of all six source files      (~25 min)
python src/normalize.py  --data-dir $DATA/dataset --work-dir work

# 2. fixed train / validation split of Source 1 entities         (~1 min)
python src/make_split.py --data-dir $DATA/dataset --work-dir work

# 3. candidate generation (blocking)                            (~15 min each)
python src/blocking.py --split train --work-dir work
python src/blocking.py --split train --work-dir work \
    --ids-file work/splits/train_s1_ids.txt --sample 100000 --tag trainsample
python src/blocking.py --split test  --work-dir work --query-chunk 100000

# 4. features + LightGBM + threshold tuning + test prediction    (~3-5 h)
python src/train_predict.py --work-dir work --out-dir output \
    --train-cands work/blocking/train_trainsample_s100000 \
    --val-cands   work/blocking/train_val \
    --test-cands  work/blocking/test_all \
    --resource-dir $DATA
```

This writes `output/matching_results.tsv` and `output/candidate_pairs.tsv`, and ends by running the official validator. Times are for a 4-CPU machine (Kaggle CPU notebook). Peak memory is about 20 GB.

To re-apply new post-processing settings (edited `work/model/params.json`) without rescoring the test set:

```
python src/train_predict.py --work-dir work --out-dir output --apply-only --resource-dir $DATA
```

## Code

| File | What it does |
|---|---|
| `src/normalize.py` | Lowercasing, junk removal, abbreviation expansion (name and address), state mapping, Indian-script romanisation (via Python's built-in Unicode names), digit-for-letter fixes; extracts house numbers, postcode and state. `--self-test` shows examples. |
| `src/make_split.py` | Holds out 50,000 Source 1 entities (fixed seed) for validation; saves the ground truth as one row per link. |
| `src/blocking.py` | Candidate generation with 7 country-scoped key passes: rarest name words, rare name word, name prefixes, house number + rare address word, postcode, **address word pairs** and **house number + any address word**. On the train split it prints recall per pass and the F0.5 ceiling. |
| `src/features.py` | About 45 pair features: fuzzy name and address similarity (RapidFuzz), shared numbers, postcode and state agreement, blocking passes, and rank within the entity's candidates. |
| `src/postprocess.py` | Global and per-country thresholds, the one-owner rule, the official macro F0.5 metric, tuning. |
| `src/train_predict.py` | Trains LightGBM, tunes on validation, scores the test set, writes both output files, runs the validator. |
| `src/analyze_misses.py` | Diagnostics: shows which true matches blocking misses and why. |
| `src/explore_data.py` | Initial EDA. |

## Outputs of intermediate steps

| Path | Content |
|---|---|
| `work/norm/*.parquet` | entity_id, country, name_norm, name_core, addr_norm, addr_nums, postcode, state |
| `work/splits/` | val / train Source 1 ids, gt_links.parquet (one row per true link) |
| `work/blocking/<split>_<tag>/part-*.parquet` | s1_entity_id, candidate_entity_id, passes (bit mask), n_passes[, label] |
| `work/model/` | model.txt, params.json (thresholds + val F0.5), val_scores.parquet, test_scores.parquet, tuning tables |

## Compliance

No external data, APIs, geocoding or lookups. All normalization rules are hand-written from patterns in the provided data. The only model is LightGBM (MIT license); no pretrained models are used.
