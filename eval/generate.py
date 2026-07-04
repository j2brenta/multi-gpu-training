#!/usr/bin/env python3
"""Sanity-check the fine-tuned model: load the HF checkpoint torchtune wrote and
generate a few completions. Look for HN-flavoured tone/vocabulary vs the base model.

Usage:
    python eval/generate.py --model-dir /workspace/output/qwen2_5_7B_hn/epoch_0
"""
import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Must match REPLY_SEP in data/prepare.py so reply-mode prompting matches training.
REPLY_SEP = "\n\n— reply —\n\n"

# For a `raw`-mode model: free-running comment continuations.
DEFAULT_PROMPTS = [
    "The real problem with modern web development is",
    "I've been using Rust in production for two years and",
    "Ask HN: how do you stay productive when",
    "Honestly, the startup advice everyone repeats is",
]

# For a `reply`-mode model: a thread context; the model completes the reply. These get
# REPLY_SEP appended automatically when --reply-mode is set.
DEFAULT_REPLY_CONTEXTS = [
    "Show HN: I built a self-hosted alternative to Notion",
    "The company mandated return-to-office five days a week starting next month.",
    "Why do so many developers dislike writing tests?",
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-dir", required=True,
                    help="Dir with the fine-tuned weights + config.json. NOTE: torchtune "
                         "0.4.0 writes these to the output_dir ROOT (hf_model_*.pt), not an "
                         "epoch_N/ subfolder — point here at /workspace/output/qwen2_5_7B_hn")
    ap.add_argument("--tokenizer-dir", default="/workspace/models/Qwen2.5-7B",
                    help="Where to load the tokenizer. Fine-tuning doesn't change it, so the "
                         "base model dir is always correct — and torchtune does NOT copy the "
                         "tokenizer files into the checkpoint dir.")
    ap.add_argument("--max-new-tokens", type=int, default=120)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--reply-mode", action="store_true",
                    help="Model trained with --mode reply: append REPLY_SEP to each prompt")
    ap.add_argument("--prompts", nargs="*", default=None,
                    help="Override the default prompts/contexts")
    args = ap.parse_args()

    if args.prompts is not None:
        prompts = args.prompts
    elif args.reply_mode:
        prompts = DEFAULT_REPLY_CONTEXTS
    else:
        prompts = DEFAULT_PROMPTS

    # Tokenizer from the base model (torchtune doesn't copy it into the checkpoint);
    # weights from the fine-tuned dir. Single-device load avoids needing `accelerate`.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.tokenizer_dir)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir, dtype=torch.bfloat16
    ).to(device)
    model.eval()

    for prompt in prompts:
        text_in = prompt + REPLY_SEP if args.reply_mode else prompt
        inputs = tok(text_in, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
                pad_token_id=tok.eos_token_id,
            )
        text = tok.decode(out[0], skip_special_tokens=True)
        print("=" * 80)
        print(text)
    print("=" * 80)


if __name__ == "__main__":
    main()
