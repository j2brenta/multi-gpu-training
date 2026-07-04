#!/usr/bin/env bash
# Download Qwen2.5-7B (base) weights into /workspace/models. Apache-2.0, not gated.
set -euo pipefail

DEST=/workspace/models/Qwen2.5-7B
mkdir -p "$DEST"

# torchtune's downloader (wraps huggingface_hub). Qwen2.5-7B ships 4 safetensors
# shards + vocab.json + merges.txt — exactly what the config references.
tune download Qwen/Qwen2.5-7B \
    --output-dir "$DEST" \
    --ignore-patterns "original/consolidated*"

echo "[download] contents:"
ls -lh "$DEST"
echo "[download] verify these safetensors shard names match configs/qwen2_5_7B_full_fsdp.yaml"
