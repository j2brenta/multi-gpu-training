#!/usr/bin/env bash
# THE "AFTER" SHOT: same config, sharded across all visible GPUs via FSDP2.
set -euo pipefail

NGPU="${NGPU:-$(python -c 'import torch; print(torch.cuda.device_count())')}"
echo "[train] launching FSDP full finetune on ${NGPU} GPUs"

# NCCL niceties for single-node multi-GPU; uncomment to debug hangs.
# export NCCL_DEBUG=INFO
# export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

tune run --nproc_per_node "${NGPU}" full_finetune_distributed \
    --config configs/qwen2_5_7B_full_fsdp.yaml \
    "$@"

echo "[train] done. Final HF checkpoint under /workspace/output/qwen2_5_7B_hn/epoch_*"
echo "[train] sanity check: python eval/generate.py --model-dir <that epoch dir>"
