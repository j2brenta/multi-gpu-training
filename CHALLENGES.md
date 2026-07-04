# Challenge / Gotcha Log

Pre-seeded with the FSDP pain points to *watch for*. As each one bites (or doesn't),
fill in what happened, the symptom, and the fix. Unchecked = anticipated, not yet hit.
The ones that actually bite are the best article material — capture the exact error text.

Template for a hit:

```
### <title>  [HIT 2026-07-05]
**Symptom:** <exact error / behaviour>
**Cause:** ...
**Fix:** ...
**Article angle:** ...
```

---

## Anticipated (checklist)

- [ ] **Single-GPU OOM baseline** — confirm full 7B FT OOMs on 1×80 GB (the "before" shot).
      Capture the exact `torch.cuda.OutOfMemoryError` and the reserved/allocated numbers.
- [ ] **FSDP checkpoint format** — `FULL_STATE_DICT` (rank-0 + CPU offload, gathers whole
      model → slow, big RAM spike) vs `SHARDED_STATE_DICT` (fast, but needs re-consolidation
      for inference). torchtune's `FullModelHFCheckpointer` writes HF format — confirm the
      final `epoch_N/` folder loads in `transformers`. #1 classic footgun.
- [ ] **Auto-wrap policy** — must wrap the *transformer decoder layer*, not the whole model.
      Wrong policy = one giant shard = no memory win = still OOMs. Verify per-GPU memory drops
      ~2× vs single-rank.
- [ ] **Activation checkpointing on/off** — off may OOM at seq 4096; on costs ~20–30% speed.
      Log peak memory with `log_peak_memory_stats: True` both ways.
- [ ] **bf16 vs fp16** — use bf16 on A100/H100; fp16 needs loss scaling and can diverge.
- [ ] **Gradient clipping under FSDP** — must use FSDP-aware clip (torchtune handles via
      `clip_grad_norm`); naive `clip_grad_norm_` on sharded params is wrong.
- [ ] **Data loader starving GPUs** — if GPU util sawtooths, tokenization/IO is the bottleneck.
      Pre-clean in `prepare.py`; watch first-epoch packing cost.
- [ ] **NCCL / torchrun init** — `RANK`/`WORLD_SIZE`/`MASTER_ADDR` env, NCCL timeouts, and
      whether the pod actually has NVLink (`nvidia-smi topo -m`) vs PCIe.
- [ ] **FSDP1 vs FSDP2** — torchtune uses FSDP2 (per-parameter sharding). Note any API/behaviour
      differences if cross-referencing older Llama-recipes tutorials.
- [ ] **Resume from checkpoint** — `resume_from_checkpoint: True` with a mid-run recipe state;
      test that it actually restores optimizer state, not just weights.
- [ ] **Throughput / MFU** — record tokens/s and (optionally) MFU; needed for any 2→N scaling
      chart. Note comms overhead as world size grows.
- [ ] **Tokenizer / packing correctness** — spot-check that packed sequences have correct EOS
      boundaries and no cross-document attention leakage assumptions break the loss.
- [ ] **`checkpoint_files` mismatch** — the safetensors shard list in the config must match what
      actually downloaded (4 shards for Qwen2.5-7B). Wrong list → silent partial load / crash.
- [ ] **`fused=True` AdamW** — fused optimizer + FSDP can interact badly on some torch versions;
      fall back to non-fused if you see NaNs or a fused-kernel error.

## Hit
<!-- move items here with a filled-in entry as they actually occur -->
