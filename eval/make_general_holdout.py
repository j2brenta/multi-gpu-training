#!/usr/bin/env python3
"""Build a small NON-HN (general English) holdout for the catastrophic-forgetting check.

Run eval/perplexity.py on this for BOTH base and fine-tuned. HN perplexity should DROP
(it did: 22.2 -> 13.9) while THIS general-English perplexity stays roughly FLAT. A big
spike here = the fine-tune traded general ability for HN style (lr/steps too hot); a
roughly-flat number = it learned HN *without* getting globally dumber.

    python eval/make_general_holdout.py --out /workspace/data/general_holdout.parquet
    python eval/perplexity.py --model-dir /workspace/models/Qwen2.5-7B \
        --data /workspace/data/general_holdout.parquet
    python eval/perplexity.py --model-dir /workspace/output/qwen2_5_7B_hn/hf \
        --data /workspace/data/general_holdout.parquet
"""
import argparse

import polars as pl
from datasets import load_dataset


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True, help="Output parquet (single 'text' column)")
    ap.add_argument("--n", type=int, default=500, help="How many paragraphs to keep")
    ap.add_argument("--min-chars", type=int, default=200,
                    help="Skip short/blank lines and section headers")
    args = ap.parse_args()

    # wikitext-2-raw: tiny (~4 MB), ungated, one line per row with many blanks and
    # '= Heading =' rows. Keep only real prose paragraphs so perplexity is meaningful.
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    texts = [t.strip() for t in ds["text"]]
    texts = [t for t in texts if len(t) >= args.min_chars and not t.startswith("=")]
    texts = texts[: args.n]
    if not texts:
        raise SystemExit("[general] no paragraphs passed the filter — lower --min-chars")

    pl.DataFrame({"text": texts}).write_parquet(args.out)
    print(f"[general] wrote {len(texts)} general-English paragraphs -> {args.out}")


if __name__ == "__main__":
    main()
