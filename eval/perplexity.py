#!/usr/bin/env python3
"""Held-out perplexity: the primary quantitative check that the fine-tune beat base.

Lower perplexity on HN text the model NEVER trained on = it learned the HN next-token
distribution (the exact objective it was trained on). Run it for BOTH the base model
and the fine-tuned checkpoint on the SAME held-out parquet, then diff:

    # base model (the "before")
    python eval/perplexity.py --model-dir /workspace/models/Qwen2.5-7B \
        --data /workspace/data/hn_prepared.holdout.parquet

    # fine-tuned checkpoint (the "after") — should report a clearly LOWER perplexity
    python eval/perplexity.py --model-dir /workspace/output/qwen2_5_7B_hn/epoch_0 \
        --data /workspace/data/hn_prepared.holdout.parquet

Make the held-out file with:  data/prepare.py ... --holdout-frac 0.02

Fairness: both checkpoints share the Qwen tokenizer, so token counts are identical and
the comparison is apples-to-apples. Point --data at a NON-HN corpus (e.g. a parquet of
Wikipedia paragraphs) to run the catastrophic-forgetting guardrail: that perplexity
should stay roughly flat between base and fine-tuned, not spike.
"""
import argparse
import math

import polars as pl
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-dir", required=True,
                    help="HF checkpoint dir: base model or a fine-tuned epoch_N")
    ap.add_argument("--data", required=True,
                    help="Held-out parquet with a text column (NOT training data)")
    ap.add_argument("--tokenizer-dir", default="/workspace/models/Qwen2.5-7B",
                    help="Tokenizer source (base model — fine-tuning doesn't change it, and "
                         "torchtune doesn't copy tokenizer files into the checkpoint dir).")
    ap.add_argument("--column", default="text", help="Text column name")
    ap.add_argument("--max-seq-len", type=int, default=4096,
                    help="Truncate each document to this many tokens")
    ap.add_argument("--limit", type=int, default=1000,
                    help="Score at most this many documents (eval speed)")
    args = ap.parse_args()

    texts = pl.read_parquet(args.data).get_column(args.column).drop_nulls().to_list()
    texts = texts[: args.limit]
    if not texts:
        raise SystemExit(f"[ppl] no rows in {args.data} column '{args.column}'")

    # Single GPU is plenty for eval (7B bf16 ~14 GB); avoid device_map so we don't
    # need `accelerate`. Falls back to CPU if no CUDA (slow, but runs anywhere).
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.tokenizer_dir)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir, dtype=torch.bfloat16
    ).to(device)
    model.eval()

    # Token-weighted corpus perplexity = exp(sum(NLL) / sum(predicted tokens)). HF returns
    # the MEAN cross-entropy over the (n-1) predicted positions per doc, so multiply back
    # out to a summed NLL before aggregating — otherwise long and short docs miscount.
    total_nll, total_tokens = 0.0, 0
    for i, text in enumerate(texts):
        ids = tok(text, return_tensors="pt", truncation=True,
                  max_length=args.max_seq_len).input_ids.to(device)
        n_pred = ids.size(1) - 1
        if n_pred < 1:
            continue
        with torch.no_grad():
            loss = model(ids, labels=ids).loss
        total_nll += loss.item() * n_pred
        total_tokens += n_pred
        if (i + 1) % 100 == 0:
            print(f"[ppl] {i + 1}/{len(texts)} docs | "
                  f"running ppl {math.exp(total_nll / total_tokens):.3f}")

    if total_tokens == 0:
        raise SystemExit("[ppl] every document was too short to score")

    mean_nll = total_nll / total_tokens
    print("=" * 64)
    print(f"[ppl] model      : {args.model_dir}")
    print(f"[ppl] data       : {args.data}")
    print(f"[ppl] tokens      : {total_tokens:,} over {len(texts):,} docs")
    print(f"[ppl] mean loss  : {mean_nll:.4f}")
    print(f"[ppl] PERPLEXITY : {math.exp(mean_nll):.3f}")
    print("=" * 64)


if __name__ == "__main__":
    main()
