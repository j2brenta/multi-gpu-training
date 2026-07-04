#!/usr/bin/env bash
# THE "BEFORE" SHOT for the article.
# Run the SAME full-finetune config on a SINGLE rank. With world_size=1, FULL_SHARD
# shards across 1 GPU = no sharding = the full ~120 GB of param+grad+optimizer state
# lands on one card. On a single 80 GB GPU this is expected to OOM.
#
# Capture the traceback (torch.cuda.OutOfMemoryError, with the reserved/allocated
# numbers) and log it in CHALLENGES.md. Then run launch_train.sh (2 GPUs) to show the
# same config succeeding once the state is sharded.
set -uo pipefail

echo "[baseline] Expect CUDA OOM below — that is the POINT (single GPU, no sharding)."
echo "[baseline] limiting to 1 GPU and a few steps..."

CUDA_VISIBLE_DEVICES=0 tune run --nproc_per_node 1 full_finetune_distributed \
    --config configs/qwen2_5_7B_full_fsdp.yaml \
    max_steps_per_epoch=2 \
    2>&1 | tee /workspace/output/baseline_oom.log

echo "[baseline] If it did NOT OOM, note that in DECISIONS.md (card had enough memory)"
echo "[baseline] and consider a bigger model or reduced offload to make the point."
