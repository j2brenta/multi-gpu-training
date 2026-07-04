#!/usr/bin/env python3
"""Prepare the Hacker News full export for continued pre-training.

Input is the official HN API export (~38M items: stories, comments, polls) in one
parquet with the standard Firebase item schema (id, type, by, time, text, dead,
deleted, parent, title, url, score, ...).

Two modes:
  raw   : keep cleaned comment text, one comment per training document.
  reply : self-join each comment to its IMMEDIATE parent item and build a
          "(context) -> reply" document, so the fine-tuned model is steerable
          (prompt with a thread, get an HN-style reply). Context = the parent's
          title (if the parent is a story) or the parent's comment text.

Token budget is approximate (char/token heuristic) on purpose — an exact tokenizer
count would couple this step to the model and slow it down. The heuristic and budget
are logged decisions (see DECISIONS.md).

Usage:
    python data/prepare.py \
        --input /workspace/data/hn_raw.parquet \
        --output /workspace/data/hn_prepared.parquet \
        --mode reply \
        --target-tokens 200_000_000
"""
import argparse

import polars as pl

# Delimiter between context and reply in `reply` mode. At inference, prompt the model
# with:  <context><REPLY_SEP>  and let it complete the reply.
REPLY_SEP = "\n\n— reply —\n\n"


def clean(col: str) -> pl.Expr:
    """Undo HN's HTML: <p> -> blank line, strip tags, unescape common entities."""
    return (
        pl.col(col)
        .str.replace_all("<p>", "\n\n", literal=True)
        .str.replace_all(r"<[^>]+>", "")          # strip any remaining tags
        .str.replace_all("&#x27;", "'", literal=True)
        .str.replace_all("&#x2F;", "/", literal=True)
        .str.replace_all("&gt;", ">", literal=True)
        .str.replace_all("&lt;", "<", literal=True)
        .str.replace_all("&quot;", '"', literal=True)
        .str.replace_all("&amp;", "&", literal=True)   # keep &amp; last
        .str.strip_chars()
    )


def build_raw(comments: pl.LazyFrame, args) -> pl.LazyFrame:
    """One cleaned comment per document."""
    return comments.select(clean(args.text_column).alias("text"))


def build_reply(comments: pl.LazyFrame, all_items: pl.LazyFrame,
                schema: list[str], args) -> pl.LazyFrame:
    """Join each comment to its immediate parent -> '(context) REPLY_SEP (reply)'."""
    for needed in ("parent", "title"):
        if needed not in schema:
            raise SystemExit(
                f"[prepare] --mode reply needs a '{needed}' column; found {schema}"
            )

    # Parent lookup over ALL items (a parent may be a story w/ title or a comment w/ text).
    parents = all_items.select(
        pl.col("id").alias("parent"),
        clean("title").alias("_parent_title"),
        clean(args.text_column).alias("_parent_text"),
    )
    joined = comments.join(parents, on="parent", how="left")

    # Prefer the parent story's title; fall back to the parent comment's text.
    context = pl.coalesce(
        pl.when(pl.col("_parent_title").str.len_chars() > 0).then(pl.col("_parent_title")),
        pl.col("_parent_text"),
    )
    return (
        joined.with_columns(
            context.alias("_ctx"),
            clean(args.text_column).alias("_reply"),
        )
        .filter(pl.col("_ctx").is_not_null() & (pl.col("_ctx").str.len_chars() > 0))
        .select((pl.col("_ctx") + pl.lit(REPLY_SEP) + pl.col("_reply")).alias("text"))
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, help="Raw HN parquet path")
    ap.add_argument("--output", required=True, help="Output prepared parquet path")
    ap.add_argument("--mode", choices=("raw", "reply"), default="raw",
                    help="raw = comment text only; reply = (parent context -> reply)")
    ap.add_argument("--text-column", default="text", help="Column holding item text")
    ap.add_argument("--target-tokens", type=int, default=200_000_000,
                    help="Approx token budget to keep (subsample target)")
    ap.add_argument("--chars-per-token", type=float, default=4.0,
                    help="Heuristic for token estimate from char count")
    ap.add_argument("--min-chars", type=int, default=64,
                    help="Drop documents shorter than this (noise)")
    ap.add_argument("--max-chars", type=int, default=8000,
                    help="Drop documents longer than this (outliers)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    schema = pl.scan_parquet(args.input).collect_schema().names()
    print(f"[prepare] mode={args.mode} | input columns: {schema}")

    # Filter to real comments — the training signal in both modes.
    comments = pl.scan_parquet(args.input)
    if "type" in schema:
        comments = comments.filter(pl.col("type") == "comment")
    for flag in ("dead", "deleted"):
        if flag in schema:
            comments = comments.filter(pl.col(flag).fill_null(False) == False)  # noqa: E712
    comments = comments.filter(pl.col(args.text_column).is_not_null())

    if args.mode == "reply":
        # reply mode needs the FULL (unfiltered) item table for the parent lookup.
        built = build_reply(comments, pl.scan_parquet(args.input), schema, args)
    else:
        built = build_raw(comments, args)

    built = built.filter(
        pl.col("text").str.len_chars().is_between(args.min_chars, args.max_chars)
    )

    print("[prepare] scanning + cleaning + (self-)joining (streaming)...")
    df = built.collect(engine="streaming")
    print(f"[prepare] {df.height:,} documents after cleaning/filtering")

    # Shuffle, then keep rows until the running char budget hits the token target.
    char_budget = int(args.target_tokens * args.chars_per_token)
    df = (
        df.sample(fraction=1.0, shuffle=True, seed=args.seed)
        .with_columns(pl.col("text").str.len_chars().cum_sum().alias("_cum_chars"))
        .filter(pl.col("_cum_chars") <= char_budget)
        .drop("_cum_chars")
    )

    kept_chars = df.select(pl.col("text").str.len_chars().sum()).item() or 0
    est_tokens = int(kept_chars / args.chars_per_token)
    print(f"[prepare] kept {df.height:,} documents "
          f"(~{est_tokens:,} tokens by heuristic) -> {args.output}")

    df.write_parquet(args.output)
    print("[prepare] done.")


if __name__ == "__main__":
    main()
