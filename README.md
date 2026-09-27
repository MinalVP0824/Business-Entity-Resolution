# Business Entity Resolution: Amazon ML Challenge 2026 (team Pink Pixels)

For every Source 1 business, find the Source 2 / Source 3 records that describe the same business.

```
normalize → split → blocking (9 key passes) → LightGBM scorer/filter (2 stages)
          → cross-encoder re-scoring → blend + thresholds + one-owner rule → output
```

Final public leaderboard score: 0.951. The methodology is in `Documentation_template.md`.

## Setup

Python 3.10+. Install the pinned versions:

```
pip install -r requirements.txt
```

Steps 1–4 and 6 run on CPU (4 cores, about 20 GB RAM). Step 5 needs a CUDA GPU (a Kaggle T4 takes about 1 hour); on CPU it works but takes many hours. The pre-trained cross-encoder `cross-encoder/ms-marco-MiniLM-L-6-v2` (Apache-2.0, 22M parameters) is downloaded from the Hugging Face hub on first use. No other external resource is used.

The data is expected in the layout of the provided `student_resource/` folder:

```
student_resource/
├── dataset/
│   ├── train/  train_source1.tsv  train_source2.tsv  train_source3.tsv  train_ground_truth.tsv
│   └── test/   test_source1.tsv   test_source2.tsv   test_source3.tsv
└── utils/validate_submission.py
```

## Reproduce the submission

Run from this folder. Set `DATA` to the `student_resource` path. All intermediate files go to `work/`, and the two final files to `output/`.

```
DATA=/path/to/student_resource

# 1. normalize names and addresses of all six source files          (~25 min)
python src/normalize.py  --data-dir $DATA/dataset --work-dir work

# 2. fixed train / validation split of Source 1 entities (seed 2026)  (~1 min)
python src/make_split.py --data-dir $DATA/dataset --work-dir work

# 3. candidate generation: 9 blocking passes                          (~1 h in total)
python src/blocking.py --split train --work-dir work --typo-passes
python src/blocking.py --split train --work-dir work --typo-passes \
    --ids-file work/splits/train_s1_ids.txt --sample 100000 --tag trainsample
python src/blocking.py --split test  --work-dir work --typo-passes --query-chunk 100000

# 4. features + two-stage LightGBM + thresholds + test scoring         (~8-9 h)
#    writes work/lgb/candidate_pairs.tsv = blocking candidates that pass the
#    LightGBM filter (score >= 0.02): the set the final matching step runs on
python src/train_predict.py --work-dir work --out-dir work/lgb \
    --train-cands work/blocking/train_trainsample_s100000 \
    --val-cands   work/blocking/train_val \
    --test-cands  work/blocking/test_all \
    --stage2 --transitive --source-thresholds --resource-dir $DATA

# 5. cross-encoder: fine-tune on validation half A, tune blend + thresholds on
#    half B, re-score the top 20 candidates of every test entity      (~1 h on GPU)
python src/cross_encoder.py --run-dir work --data-dir $DATA/dataset \
    --resource-dir $DATA --out-dir work/ce --topk 20

# 6. final French threshold (France has no training data; see documentation)
python src/france_threshold.py --run-dir work --ce-dir work/ce \
    --data-dir $DATA/dataset --france 0.9 --out-dir work/fr

# final files
mkdir -p output
cp work/fr/matching_results_fr0.90.tsv output/matching_results.tsv
cp work/lgb/candidate_pairs.tsv        output/candidate_pairs.tsv
python $DATA/utils/validate_submission.py --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv --test-dir $DATA/dataset/test
```

`python src/make_candidates.py --run-dir work --data-dir $DATA/dataset --matching output/matching_results.tsv --out-dir work/cand`
rebuilds `candidate_pairs.tsv` from the saved scores (same content as step 4), prints its recall on validation, and checks that every final match is inside it.

On Kaggle we ran these steps as several notebooks (Kaggle's 12-hour session limit), attaching each notebook's output to the next: steps 1–3 plus training (`--skip-test`), then test scoring (`--skip-train`), then steps 5–6 on a GPU notebook. The commands are the same.

## Code

| File | What it does |
|---|---|
| `src/normalize.py` | Indian-script romanisation (Python's built-in Unicode names), lowercasing, accent removal, junk removal, record IDs / long numbers removed from names, digit-for-letter fixes, name and address abbreviation expansion, state mapping; extracts house numbers, postcode and state. `--self-test` shows examples. |
| `src/make_split.py` | Holds out 50,000 Source 1 entities for validation; saves the ground truth as one row per link. |
| `src/blocking.py` | 9 country-scoped key passes (rare name words, name prefixes, name without spaces, house number + address word, address word pairs, address-word prefixes, postcode). On the train split it prints recall per pass and the F0.5 ceiling. |
| `src/features.py` | About 50 pair features: fuzzy name / address similarity (RapidFuzz), containment, shared numbers, postcode and state agreement, blocking passes, rank within the entity's candidates. |
| `src/stage2.py` | Second-stage features: score context within the entity and similarity to the entity's best candidate. |
| `src/postprocess.py` | Thresholds (global, per country, per country and source), one-owner rule, the official macro F0.5 metric, tuning. |
| `src/train_predict.py` | Trains the two-stage LightGBM, tunes on validation, scores the test set, writes the filtered `candidate_pairs.tsv`, runs the validator. |
| `src/cross_encoder.py` | Fine-tunes and applies the cross-encoder, tunes the blend weight and thresholds on validation half B, writes `matching_results.tsv`. |
| `src/france_threshold.py` | Re-applies the final scores with a different French threshold only. |
| `src/make_candidates.py` | Rebuilds / checks `candidate_pairs.tsv` and reports its recall. |
| `src/analyze_misses.py`, `src/analyze_errors.py`, `src/explore_data.py` | Diagnostics used during development (blocking misses, error breakdown, EDA). |

## Compliance

No external data, APIs, geocoding or lookups. All normalization rules are hand-written from patterns in the provided data. Models: LightGBM (MIT) and `cross-encoder/ms-marco-MiniLM-L-6-v2` (Apache-2.0, 22M parameters), fine-tuned only on the provided training data.
