#!/usr/bin/env python3
"""Convert a torchtune checkpoint dump into a standard, HF-loadable model dir.

torchtune 0.4.0's FullModelHFCheckpointer writes the fine-tuned weights as
`hf_model_0001_0.pt ...` in the output ROOT (not an epoch_N/ subfolder), which
`transformers.from_pretrained` cannot auto-discover — it wants `model*.safetensors`
or `pytorch_model*.bin` + an index. It also does NOT copy the tokenizer. This merges
those shards and writes one clean dir (safetensors + config + tokenizer) that
generate.py / perplexity.py / any HF tool load directly.

    python eval/to_hf.py \
        --ckpt-dir /workspace/output/qwen2_5_7B_hn \
        --base-dir /workspace/models/Qwen2.5-7B \
        --out      /workspace/output/qwen2_5_7B_hn/hf

Config + tokenizer are taken from --base-dir (fine-tuning changes neither), so we don't
depend on torchtune's minimal config.json. Then:  generate.py --model-dir <--out>
"""
import argparse
import glob
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt-dir", required=True,
                    help="Dir with torchtune's hf_model_*.pt shards")
    ap.add_argument("--base-dir", default="/workspace/models/Qwen2.5-7B",
                    help="Original base model dir — source of the unchanged config + tokenizer")
    ap.add_argument("--out", required=True, help="Output HF model dir to create")
    args = ap.parse_args()

    shards = sorted(glob.glob(str(Path(args.ckpt_dir) / "hf_model_*.pt")))
    if not shards:
        raise SystemExit(f"[to_hf] no hf_model_*.pt found in {args.ckpt_dir}")

    print(f"[to_hf] merging {len(shards)} shard(s) into one state dict...")
    state = {}
    for s in shards:
        state.update(torch.load(s, map_location="cpu", weights_only=True))
    print(f"[to_hf] {len(state)} tensors loaded")

    print(f"[to_hf] building model from {args.base_dir} config and loading weights...")
    config = AutoConfig.from_pretrained(args.base_dir)
    model = AutoModelForCausalLM.from_config(config, dtype=torch.bfloat16)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[to_hf] NOTE {len(missing)} missing key(s) (often just tied lm_head): "
              f"{missing[:3]}{'...' if len(missing) > 3 else ''}")
    if unexpected:
        print(f"[to_hf] WARNING {len(unexpected)} unexpected key(s): "
              f"{unexpected[:3]}{'...' if len(unexpected) > 3 else ''}")

    print(f"[to_hf] writing HF model (safetensors) -> {args.out}")
    model.save_pretrained(args.out, safe_serialization=True)
    # Bundle the tokenizer so the dir is self-contained.
    AutoTokenizer.from_pretrained(args.base_dir).save_pretrained(args.out)

    print(f"[to_hf] done. Sanity-check with:\n"
          f"    python eval/generate.py --model-dir {args.out} --reply-mode")


if __name__ == "__main__":
    main()
