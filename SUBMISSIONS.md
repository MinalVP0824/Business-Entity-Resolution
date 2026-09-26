# Submission log

| # | Date / time (IST) | Val F0.5 | Leaderboard | Commit | Notes |
|---|---|---|---|---|---|
| 1 | 25 Sep 2026, 9:36 PM | 0.9180 | 0.909 | 99e61783bb7d29cebf45dd45f244db1c913349f2 | Baseline: normalize v1, blocking v1 (5 passes, max 200 cands), LightGBM (693 rounds), threshold 0.65, one-owner rule. Val: India 0.8854, US 0.9394. Trained on 100k train S1 sample. |

| 2 | 26 Sep 2026, 3:55 PM | 0.9199 | 0.911 | (commit, see below) | Run 2: same pipeline as #1, trained on 150k train S1 entities (vs 100k), per-country thresholds (india 0.65, us 0.7), saves test_scores.parquet. Val: India 0.8875, US 0.9411. |
