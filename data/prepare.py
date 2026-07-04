#!/usr/bin/env python3
"""Prepare the Hacker News full export for continued pre-training.

Input is the official HN API export (~38M items: stories, comments, polls) in one
parquet with the standard Firebase item schema (id, type, by, time, text, dead,
deleted, parent, title, url, score, ...).

Three modes:
  raw        : keep cleaned comment text, one comment per training document.
  reply      : self-join each comment to its IMMEDIATE parent item and build a
               "(context) -> reply" document, so the fine-tuned model is steerable
               (prompt with a thread, get an HN-style reply). Context = the parent's
               title (if the parent is a story) or the parent's comment text.
  reply_root : like `reply` but anchored to the whole thread's ROOT STORY. Climbs
               the parent chain to the submitted story and prefixes the context with
               "[Story] <title> (<domain>)", then quotes the immediate parent comment
               (capped). Gives the model the topic + source the comment reacts to, so
               a comment deep in a thread is interpretable/promptable by article.

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
from pathlib import Path

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


def domain(col: str) -> pl.Expr:
    """Bare host from a URL: https://www.example.com/x?y -> example.com."""
    return pl.col(col).str.extract(r"https?://(?:www\.)?([^/?#]+)", 1)


def resolve_root_story(all_items: pl.LazyFrame, max_hops: int) -> pl.LazyFrame:
    """Map every item id -> its root story id by climbing the parent chain.

    Pointer-doubling: each pass replaces a node's pointer with its pointer's
    pointer, so the reachable depth doubles per pass (~log2(depth) passes, not
    one-per-level). A story points at itself, so chains converge on the root and
    we stop early at the fixpoint. A missing/deleted parent self-terminates too.
    Runs eagerly on a minimal (id, parent, type) projection — the graph walk is
    iterative, which the lazy/streaming engine can't express.
    """
    base = all_items.select("id", "parent", "type").collect(engine="streaming")
    ptr = base.select(
        "id",
        pl.when(pl.col("type") == "story")
          .then(pl.col("id"))               # root: self-loop
          .otherwise(pl.col("parent"))
          .fill_null(pl.col("id"))          # missing parent: stop here
          .alias("ptr"),
    )
    for _ in range(max_hops):
        stepped = ptr.join(
            ptr.rename({"id": "ptr", "ptr": "ptr_next"}),  # node -> its pointer
            on="ptr", how="left",
        ).with_columns(pl.coalesce("ptr_next", "ptr").alias("ptr_new"))
        changed = stepped.select((pl.col("ptr_new") != pl.col("ptr")).sum()).item()
        ptr = stepped.select("id", pl.col("ptr_new").alias("ptr"))
        if not changed:
            break
    return ptr.rename({"ptr": "root_id"}).lazy()


def build_reply_root(comments: pl.LazyFrame, all_items: pl.LazyFrame,
                     schema: list[str], args) -> pl.LazyFrame:
    """Root-story-anchored context: '[Story] title (domain)' + immediate parent."""
    for needed in ("parent", "title"):
        if needed not in schema:
            raise SystemExit(
                f"[prepare] --mode reply_root needs a '{needed}' column; found {schema}"
            )
    if "url" not in schema:
        print("[prepare] no 'url' column — anchoring on story title only (no domain).")

    print(f"[prepare] resolving root story per comment (<= {args.max_hops} hops)...")
    roots = resolve_root_story(all_items, args.max_hops)

    root_domain = domain("url") if "url" in schema else pl.lit(None, dtype=pl.String)
    stories = all_items.select(
        pl.col("id").alias("root_id"),
        clean("title").alias("_root_title"),
        root_domain.alias("_root_domain"),
    )
    parents = all_items.select(
        pl.col("id").alias("parent"),
        clean(args.text_column).alias("_parent_text"),
        pl.col("type").alias("_parent_type"),
    )
    joined = (
        comments
        .join(roots, on="id", how="left")
        .join(stories, on="root_id", how="left")
        .join(parents, on="parent", how="left")
    )

    # Topic anchor "[Story] <title> (<domain>)"; domain dropped if unknown, whole
    # line dropped if the root title is missing (broken/deleted chain).
    anchor = pl.when(pl.col("_root_title").str.len_chars() > 0).then(
        pl.lit("[Story] ") + pl.col("_root_title")
        + pl.when(pl.col("_root_domain").is_not_null())
            .then(pl.lit(" (") + pl.col("_root_domain") + pl.lit(")"))
            .otherwise(pl.lit(""))
    )
    # Immediate parent comment, quoted and length-capped — only when the parent is
    # itself a comment. If the parent IS the story, the anchor already carries it.
    parent_ctx = pl.when(pl.col("_parent_type") == "comment").then(
        pl.lit("> ") + pl.col("_parent_text").str.slice(0, args.max_context_chars)
    )
    context = pl.concat_str([anchor, parent_ctx], separator="\n\n", ignore_nulls=True)

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
    ap.add_argument("--mode", choices=("raw", "reply", "reply_root"), default="raw",
                    help="raw = comment text only; reply = (immediate parent -> reply); "
                         "reply_root = (root story + parent -> reply)")
    ap.add_argument("--text-column", default="text", help="Column holding item text")
    ap.add_argument("--max-context-chars", type=int, default=1000,
                    help="reply_root: cap the quoted parent-comment context length")
    ap.add_argument("--max-hops", type=int, default=16,
                    help="reply_root: max pointer-doubling passes to reach the root "
                         "story (covers ~2^hops thread depth; stops early at fixpoint)")
    ap.add_argument("--target-tokens", type=int, default=200_000_000,
                    help="Approx token budget to keep (subsample target)")
    ap.add_argument("--chars-per-token", type=float, default=4.0,
                    help="Heuristic for token estimate from char count")
    ap.add_argument("--min-chars", type=int, default=64,
                    help="Drop documents shorter than this (noise)")
    ap.add_argument("--max-chars", type=int, default=8000,
                    help="Drop documents longer than this (outliers)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--holdout-frac", type=float, default=0.0,
                    help="Split off this fraction as a disjoint held-out set for "
                         "base-vs-finetuned perplexity (eval/perplexity.py). Written "
                         "next to --output as <stem>.holdout.parquet. 0 = off.")
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
    elif args.mode == "reply_root":
        # reply_root also needs the full item table (parent chain + story metadata).
        built = build_reply_root(comments, pl.scan_parquet(args.input), schema, args)
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

    # Optional held-out split for honest base-vs-finetuned perplexity. df is already
    # shuffled above, so the head is a random sample disjoint from the training tail.
    if args.holdout_frac > 0:
        n_hold = int(df.height * args.holdout_frac)
        if n_hold > 0:
            holdout, df = df.head(n_hold), df.tail(df.height - n_hold)
            out = Path(args.output)
            hold_path = str(out.with_name(out.stem + ".holdout" + out.suffix))
            holdout.write_parquet(hold_path)
            print(f"[prepare] held out {holdout.height:,} docs -> {hold_path} "
                  f"(NOT in training set; feed to eval/perplexity.py)")

    kept_chars = df.select(pl.col("text").str.len_chars().sum()).item() or 0
    est_tokens = int(kept_chars / args.chars_per_token)
    print(f"[prepare] kept {df.height:,} documents "
          f"(~{est_tokens:,} tokens by heuristic) -> {args.output}")

    df.write_parquet(args.output)
    print("[prepare] done.")


if __name__ == "__main__":
    main()
