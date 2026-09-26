"""
cross_encoder.py
Re-scores the best LightGBM candidates with a small pretrained transformer
(a "cross-encoder") that reads both records as text, then blends its score
with the LightGBM score. Needs a GPU (Kaggle: Accelerator = GPU).

Model: cross-encoder/ms-marco-MiniLM-L-6-v2 (Apache-2.0, 22M parameters),
fine-tuned here on our own labelled validation pairs. No external data.

Inputs: the saved output of a finished LightGBM run (e.g. ber-run3), attached
as a Kaggle input:
    work/model/val_scores.parquet   validation pairs: score + true label
    work/model/test_scores.parquet  test pairs: score
    work/model/params.json          (only for reference)
    work/splits/gt_links.parquet    true links, to score validation exactly
    output/candidate_pairs.tsv      copied unchanged to the new output
plus the raw dataset (names/addresses are read from the source TSVs).

How the validation set is used (no leakage):
    validation entities are split in two halves by a fixed hash
      half A -> fine-tune the cross-encoder
      half B -> choose the blend weight and thresholds, and report F0.5
    the reported F0.5 on half B is compared with LightGBM alone on half B.

Usage (Kaggle GPU notebook):
    python src/cross_encoder.py \
        --run-dir /kaggle/input/notebooks/<user>/ber-run3 \
        --data-dir /kaggle/input/datasets/<user>/<slug>/student_resource/dataset \
        --resource-dir /kaggle/input/datasets/<user>/<slug>/student_resource

    --val-only      stop after the validation comparison (quick check, ~30 min)
    --topk 10       candidates per Source 1 entity that get re-scored

Outputs (in --out-dir, default /kaggle/working/output):
    matching_results.tsv, candidate_pairs.tsv
    ce_model/ (fine-tuned model), ce_val_scores.parquet, ce_test_scores.parquet,
    ce_params.json (blend weight, thresholds, validation F0.5)
"""

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from normalize import transliterate_indic  # noqa: E402
from postprocess import f05_macro, select_matches, tune, tune_per_country  # noqa: E402

START = time.time()


def log(msg):
    print(f"[{time.time() - START:7.1f}s] {msg}", flush=True)


def find(run_dir, name):
    hits = glob.glob(os.path.join(run_dir, "**", name), recursive=True)
    if not hits:
        raise FileNotFoundError(f"{name} not found under {run_dir}")
    return hits[0]


# ============================================================ candidates

def top_candidates(scores, k, min_score):
    """The k best-scored candidates of every Source 1 entity (score >= min)."""
    s = scores[scores["score"] >= min_score]
    s = s.sort_values(["s1_entity_id", "score"], ascending=[True, False])
    s = s[s.groupby("s1_entity_id").cumcount() < k]
    return s.reset_index(drop=True)


def half_of(ids):
    h = pd.util.hash_array(np.asarray(ids, dtype=object), hash_key="0123456789abcdef")
    return (h % 2).astype(np.int8)


# ================================================================== text

def load_text(data_dir, split, ids):
    """entity_id -> 'name | address' for the requested ids only."""
    ids = set(ids)
    parts = []
    for i in (1, 2, 3):
        path = os.path.join(data_dir, split, f"{split}_source{i}.tsv")
        for chunk in pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False,
                                 chunksize=1_000_000):
            chunk = chunk[chunk["entity_id"].isin(ids)]
            if len(chunk):
                parts.append(chunk)
    df = pd.concat(parts, ignore_index=True)
    text = (df["business_name"].map(transliterate_indic) + " | "
            + df["business_address"].map(transliterate_indic))
    return pd.Series(text.str.slice(0, 300).to_numpy(), index=df["entity_id"].to_numpy())


# ================================================================= model

def load_model(name, device):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    if name == "__tiny_test__":           # offline smoke test only
        from transformers import BertConfig, BertForSequenceClassification, BertTokenizerFast
        vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "|"] + \
            [chr(c) for c in range(97, 123)] + [str(d) for d in range(10)]
        path = "/tmp/tiny_vocab.txt"
        with open(path, "w") as fh:
            fh.write("\n".join(vocab))
        tok = BertTokenizerFast(vocab_file=path, do_lower_case=True)
        cfg = BertConfig(vocab_size=len(vocab), hidden_size=32, num_hidden_layers=2,
                         num_attention_heads=2, intermediate_size=64, num_labels=1)
        model = BertForSequenceClassification(cfg)
    else:
        tok = AutoTokenizer.from_pretrained(name)
        model = AutoModelForSequenceClassification.from_pretrained(
            name, num_labels=1, ignore_mismatched_sizes=True)
    return tok, model.to(device)


def batches(n, size):
    for i in range(0, n, size):
        yield slice(i, min(i + size, n))


def encode(tok, a, b, max_len, device):
    enc = tok(list(a), list(b), truncation=True, max_length=max_len,
              padding=True, return_tensors="pt")
    return {k: v.to(device) for k, v in enc.items()}


def fine_tune(tok, model, a, b, y, args, device):
    import torch
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    n = len(y)
    steps = args.epochs * ((n + args.batch - 1) // args.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps,
                                                pct_start=0.1)
    use_amp = device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    lossf = torch.nn.BCEWithLogitsLoss()
    rng = np.random.default_rng(2026)
    step = 0
    for ep in range(args.epochs):
        order = rng.permutation(n)
        for sl in batches(n, args.batch):
            idx = order[sl]
            enc = encode(tok, a[idx], b[idx], args.max_len, device)
            target = torch.tensor(y[idx], dtype=torch.float32, device=device)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                logits = model(**enc).logits.squeeze(-1)
                loss = lossf(logits.float(), target)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            step += 1
            if step % 500 == 0 or step == steps:
                log(f"  epoch {ep + 1} step {step}/{steps} loss {loss.item():.4f}")
    model.eval()


def predict(tok, model, a, b, args, device):
    import torch
    n = len(a)
    out = np.zeros(n, dtype=np.float32)
    # sort by length so each batch pads to a similar size (much faster)
    order = np.argsort([len(x) + len(y) for x, y in zip(a, b)])
    use_amp = device == "cuda"
    done = 0
    with torch.no_grad():
        for sl in batches(n, args.pred_batch):
            idx = order[sl]
            enc = encode(tok, a[idx], b[idx], args.max_len, device)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                logits = model(**enc).logits.squeeze(-1).float()
            out[idx] = torch.sigmoid(logits).cpu().numpy()
            done += len(idx)
            if done // args.pred_batch % 2000 == 0 or done == n:
                log(f"  scored {done:,}/{n:,} pairs")
    return out


# ================================================================ blending

def blend(scores, top, ce, w):
    """LightGBM score everywhere; for re-scored pairs, w*lgb + (1-w)*ce."""
    out = scores[["s1_entity_id", "candidate_entity_id", "score"]].copy()
    if "country" in scores.columns:
        out["country"] = scores["country"].to_numpy()
    mixed = pd.DataFrame({"s1_entity_id": top["s1_entity_id"].to_numpy(),
                          "candidate_entity_id": top["candidate_entity_id"].to_numpy(),
                          "new": w * top["score"].to_numpy() + (1 - w) * np.asarray(ce)})
    out = out.merge(mixed, on=["s1_entity_id", "candidate_entity_id"], how="left")
    out["score"] = out["new"].fillna(out["score"])
    return out.drop(columns="new")


def apply(scores, p):
    return select_matches(scores, p["threshold"], p["one_owner"], p["top1_threshold"],
                          p.get("country_thresholds"))


# =================================================================== main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, help="saved output of a LightGBM run")
    ap.add_argument("--data-dir", required=True, help="student_resource/dataset")
    ap.add_argument("--resource-dir", default=None, help="student_resource (validator)")
    ap.add_argument("--out-dir", default="/kaggle/working/output")
    ap.add_argument("--model-name", default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--min-score", type=float, default=0.02)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--pred-batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--val-only", action="store_true")
    ap.add_argument("--force-w-lgb", type=float, default=None,
                    help="use this blend weight instead of the tuned one (testing)")
    args = ap.parse_args()

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else
                               "  WARNING: no GPU, this will be very slow"))
    os.makedirs(args.out_dir, exist_ok=True)

    # ------------------------------------------------------------ validation
    val = pd.read_parquet(find(args.run_dir, "val_scores.parquet"))
    links = pd.read_parquet(find(args.run_dir, "gt_links.parquet"),
                            columns=["source1_entity_id", "matched_entity_id"])
    val_ids = pd.Series(pd.unique(val["s1_entity_id"]))
    ent_ids_path = glob.glob(os.path.join(args.run_dir, "**", "val_s1_ids.txt"), recursive=True)
    if ent_ids_path:   # include entities that had no candidates at all
        val_ids = pd.read_csv(ent_ids_path[0], header=None, dtype=str)[0]
    links = links[links["source1_entity_id"].isin(set(val_ids))]
    ent_half = pd.Series(half_of(val_ids), index=val_ids.to_numpy())
    country = val.drop_duplicates("s1_entity_id").set_index("s1_entity_id")["country"]

    top = top_candidates(val, args.topk, args.min_score)
    top["half"] = top["s1_entity_id"].map(ent_half).to_numpy()
    log(f"validation: {len(val):,} pairs, {len(top):,} re-scored "
        f"(top {args.topk}); positives in top: {top['label'].mean():.2%}; "
        f"true links covered by top: {top['label'].sum() / max(len(links), 1):.2%}")

    log("loading validation text ...")
    text = load_text(args.data_dir, "train",
                     pd.concat([top["s1_entity_id"], top["candidate_entity_id"]]).unique())
    a = top["s1_entity_id"].map(text).fillna("").to_numpy()
    b = top["candidate_entity_id"].map(text).fillna("").to_numpy()

    tok, model = load_model(args.model_name, device)
    tr = top["half"].to_numpy() == 0
    log(f"fine-tuning on half A: {int(tr.sum()):,} pairs ...")
    fine_tune(tok, model, a[tr], b[tr], top["label"].to_numpy()[tr].astype(np.float32),
              args, device)
    model.save_pretrained(os.path.join(args.out_dir, "ce_model"))
    tok.save_pretrained(os.path.join(args.out_dir, "ce_model"))

    ho = ~tr
    log(f"scoring half B: {int(ho.sum()):,} pairs ...")
    ce_b = predict(tok, model, a[ho], b[ho], args, device)
    top_b = top[ho].reset_index(drop=True)
    pd.DataFrame({"s1_entity_id": top_b["s1_entity_id"],
                  "candidate_entity_id": top_b["candidate_entity_id"],
                  "lgb": top_b["score"], "ce": ce_b, "label": top_b["label"]}) \
        .to_parquet(os.path.join(args.out_dir, "ce_val_scores.parquet"), index=False)

    # ---- choose blend weight + thresholds on half B
    ids_b = val_ids[val_ids.map(ent_half).to_numpy() == 1]
    set_b = set(ids_b)
    val_b = val[val["s1_entity_id"].isin(set_b)].reset_index(drop=True)
    links_b = links[links["source1_entity_id"].isin(set_b)]
    results = []
    for w in (1.0, 0.8, 0.6, 0.5, 0.4, 0.2, 0.0):
        sc = blend(val_b, top_b, ce_b, w)
        p, _ = tune(sc, links_b, ids_b, top1_options=(None, 0.2))
        f, pr, rc = f05_macro(apply(sc, p), links_b, ids_b)
        results.append((f, w, p, pr, rc))
        log(f"  blend w_lgb={w:.1f}: F0.5 {f:.4f}  precision {pr:.4f}  recall {rc:.4f}  "
            f"threshold {p['threshold']}")
    base = [r for r in results if r[1] == 1.0][0][0]
    f_best, w_best, p_best, _, _ = max(results, key=lambda r: r[0])
    sc = blend(val_b, top_b, ce_b, w_best)
    ct, _ = tune_per_country(sc, links_b, ids_b, country, p_best)
    p_c = dict(p_best, country_thresholds=ct)
    f_c, pr_c, rc_c = f05_macro(apply(sc, p_c), links_b, ids_b)
    if f_c > f_best:
        p_best, f_best = p_c, f_c
    print(f"\nHalf-B F0.5: LightGBM alone {base:.4f}  ->  with cross-encoder {f_best:.4f} "
          f"(w_lgb={w_best}, gain {f_best - base:+.4f})")
    for c in sorted(set(country.reindex(ids_b).dropna())):
        ids_c = ids_b[ids_b.map(country).to_numpy() == c]
        s_c = set(ids_c)
        f, pr, rc = f05_macro(apply(sc[sc["s1_entity_id"].isin(s_c)], p_best),
                              links_b[links_b["source1_entity_id"].isin(s_c)], ids_c)
        print(f"  {c:<8} F0.5 {f:.4f}  precision {pr:.4f}  recall {rc:.4f}")
    if args.force_w_lgb is not None:
        forced = [r for r in results if r[1] == args.force_w_lgb][0]
        f_best, w_best, p_best = forced[0], forced[1], forced[2]
    params = dict(p_best, w_lgb=w_best, half_b_f05=f_best, half_b_f05_lgb_only=base,
                  topk=args.topk, min_score=args.min_score, model=args.model_name)
    with open(os.path.join(args.out_dir, "ce_params.json"), "w") as fh:
        json.dump(params, fh, indent=2)

    if args.val_only:
        log("--val-only: stopping before the test set")
        return
    if w_best == 1.0:
        log("the cross-encoder did not help on validation; writing LightGBM-only results")

    # ------------------------------------------------------------------ test
    test = pd.read_parquet(find(args.run_dir, "test_scores.parquet"))
    top_t = top_candidates(test, args.topk, args.min_score)
    log(f"test: {len(test):,} saved pairs, {len(top_t):,} to re-score")
    ce_t = np.zeros(len(top_t), dtype=np.float32)
    if w_best < 1.0:
        log("loading test text ...")
        text = load_text(args.data_dir, "test",
                         pd.concat([top_t["s1_entity_id"], top_t["candidate_entity_id"]]).unique())
        a = top_t["s1_entity_id"].map(text).fillna("").to_numpy()
        b = top_t["candidate_entity_id"].map(text).fillna("").to_numpy()
        log("scoring test pairs with the cross-encoder ...")
        ce_t = predict(tok, model, a, b, args, device)
        pd.DataFrame({"s1_entity_id": top_t["s1_entity_id"],
                      "candidate_entity_id": top_t["candidate_entity_id"],
                      "lgb": top_t["score"], "ce": ce_t}) \
            .to_parquet(os.path.join(args.out_dir, "ce_test_scores.parquet"), index=False)

    final = blend(test, top_t, ce_t, w_best)
    matches = apply(final, p_best)
    all_s1 = pd.read_csv(os.path.join(args.data_dir, "test", "test_source1.tsv"), sep="\t",
                         dtype=str, keep_default_na=False, usecols=["entity_id"])["entity_id"]
    grouped = matches.groupby("s1_entity_id")["candidate_entity_id"].apply(",".join)
    out = pd.DataFrame({"source1_entity_id": all_s1})
    out["matched_entity_ids"] = out["source1_entity_id"].map(grouped).fillna("")
    match_path = os.path.join(args.out_dir, "matching_results.tsv")
    out.to_csv(match_path, sep="\t", index=False)
    has = out["matched_entity_ids"] != ""
    log(f"wrote {match_path}: {len(out):,} rows, {has.mean():.1%} with matches")

    cand_path = os.path.join(args.out_dir, "candidate_pairs.tsv")
    shutil.copyfile(find(args.run_dir, "candidate_pairs.tsv"), cand_path)
    log(f"copied {cand_path}")

    if args.resource_dir:
        subprocess.run([sys.executable,
                        os.path.join(args.resource_dir, "utils", "validate_submission.py"),
                        "--matching", match_path, "--candidate", cand_path,
                        "--test-dir", os.path.join(args.resource_dir, "dataset", "test")],
                       check=False)
    log("done")


if __name__ == "__main__":
    main()
