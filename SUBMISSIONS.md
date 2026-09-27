# Submission log

| # | Date / time (IST) | Val F0.5 | Leaderboard | Commit | Notes |
|---|---|---|---|---|---|
| 1 | 25 Sep 2026, 9:36 PM | 0.9180 | 0.909 | 99e61783bb7d29cebf45dd45f244db1c913349f2 | Baseline: normalize v1, blocking v1 (5 passes, max 200 cands), LightGBM (693 rounds), threshold 0.65, one-owner rule. Val: India 0.8854, US 0.9394. Trained on 100k train S1 sample. |
| 2 | 26 Sep 2026, 3:55 PM | 0.9199 | 0.911 | 04487bb673792074bf6a97ded2b6d4266bbd8556 | Run 2: same pipeline as #1, trained on 150k train S1 entities (vs 100k), per-country thresholds (india 0.65, us 0.7), saves test_scores.parquet. Val: India 0.8875, US 0.9411. |
| 3 | 26 Sep 2026, 7:40 PM | 0.9382 | 0.928 | a24bb1a9bb2e2aac8971e0b3f09fb39597915aaa | Run 3: normalize v2 (Indic transliteration, leetspeak fixes, state variants), blocking v2 (+2 passes: address word pairs, all address numbers), new features (no-space name similarity, address containment, number subset). Official validator re-run separately: PASS. |
| 4 | 26 Sep 2026, 9:23 PM | 0.9522 (half B) | 0.941 | 37a1b98c1f5f1d6ad48a19a6719c81c0fb19c8d4 | Run 5: cross-encoder re-ranking on top of run 3. Fine-tuned cross-encoder/ms-marco-MiniLM-L-6-v2 (Apache-2.0, 22M params) for 1 epoch on validation half A; re-scores each entity's top-10 LightGBM candidates; blend 0.5 LightGBM + 0.5 cross-encoder, thresholds tuned on half B. Half-B F0.5: LightGBM alone 0.9379 -> 0.9522 (India 0.9339, US 0.9644). |
| 5 | 27 Sep 2026, 9:28 AM | 0.9580 (half B) | 0.944 | 0aca892 | Run 5b: cross-encoder (top 20 candidates) on run 4. Blend 0.6 LightGBM + 0.4 cross-encoder. Half-B: 0.9488 -> 0.9580 (India 0.9402, US 0.9699). |
| 6 | 27 Sep 2026, <time> | 0.9624 (half B) | 0.949 | 0aca892 | Run 7 + cross-encoder (top 20), blend 0.6 LightGBM + 0.4 cross-encoder. Run 7 = typo-tolerant blocking + similarity-to-best-candidate features + per-country/source thresholds + IDs/phone numbers stripped from names (val 0.9534). Half-B: 0.9532 -> 0.9624. |
