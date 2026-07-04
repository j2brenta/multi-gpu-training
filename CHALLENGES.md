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

### torchtune HF checkpoint isn't actually HF-loadable  [HIT 2026-07-04]
**Symptom:** `eval/generate.py --model-dir /workspace/output/qwen2_5_7B_hn/epoch_0` →
```
OSError: Repo id must be in the form 'repo_name' or 'namespace/repo_name':
'/workspace/output/qwen2_5_7B_hn/epoch_0'. Use `repo_type` argument if needed.
```
**Cause:** Two things, both the classic FSDP-checkpoint footgun:
1. **Wrong path.** torchtune 0.4.0's `FullModelHFCheckpointer` writes to the output_dir
   ROOT, not an `epoch_N/` subfolder. The README assumed `epoch_0`; that dir doesn't
   exist, so `from_pretrained` fell back to treating the path as a Hub repo id → the
   confusing error above.
2. **Not HF-loadable even at the right path.** The dir has `config.json` + `hf_model_
   0001_0.pt ... 0004` (torchtune's own shard naming) but NO `model*.safetensors` /
   `pytorch_model*.bin` + index, and NO tokenizer files. `transformers.from_pretrained`
   can't auto-discover `hf_model_*.pt`, and the tokenizer is missing entirely.
**Fix:** One-time conversion `eval/to_hf.py`: merge the `.pt` shards, load into an HF
model built from the base config, `save_pretrained` as safetensors, and bundle the base
tokenizer → a clean `.../hf` dir that loads directly. Also hardened `generate.py` /
`perplexity.py` to load the tokenizer from the base model (`--tokenizer-dir`, default
`/workspace/models/Qwen2.5-7B`) and drop `device_map="auto"` (no `accelerate` needed).
**Article angle:** "torchtune wrote an HF checkpoint" is a half-truth — it's HF *format*
(converted keys) but not an HF *directory* (wrong filenames, no index, no tokenizer). The
save→load boundary is where FSDP recipes bite; always convert + reload before trusting a
run. This is exactly the anticipated "FSDP checkpoint format" item, now confirmed.

### Unpinned torchao pulled a torch>=2.11 build onto a torch 2.4.1 base  [HIT 2026-07-04]
**Symptom:** `tune` won't even start:
```
Skipping import of cpp extensions due to incompatible torch version. Please upgrade to
torch >= 2.11.0 (found 2.4.1+cu124).
...
File ".../torchao/quantization/quant_primitives.py", line 191, in <module>
    torch.int1: (-(2**0), 2**0 - 1),
AttributeError: module 'torch' has no attribute 'int1'
```
**Cause:** `torchtune`/`torchao` don't pin torch — they build against whatever the base
image ships. `requirements.txt` had them as unpinned `>=`, so uv installed the *latest*
`torchao` (0.17.0), whose module-load references `torch.int1` (a dtype only in very new
torch) and demands torch >= 2.11. The RunPod base image had torch 2.4.1 → import crash.
**Fix:** Pin the validated combo `torchtune==0.4.0` + `torchao==0.7.0` (has the config's
`qwen2_5_7b_base` builder, neither touches `torch.int1`), and add a torch-version guard in
`setup_pod.sh` that installs `torch==2.5.1+cu124` if the base image torch is < 2.5 (never
downgrades a newer one). Re-run `bash scripts/setup_pod.sh`.
**Article angle:** "torch + CUDA already present" base images are convenient but make you
the version-solver: libraries that float their torch requirement will happily install a
build the base image can't run. Pin the ML stack; treat the base image's torch as a fixed
constraint you resolve *around*, not a suggestion.
