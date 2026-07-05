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

> **RETRO 2026-07-05:** After a full successful run, the verdict is clear — **FSDP itself
> was the easy part.** Nearly every anticipated FSDP challenge was handled cleanly by the
> torchtune FSDP2 recipe (✅ below). The only anticipated item that actually bit was the
> checkpoint *format* — and even that as a loadability/tooling problem, not sharding. All
> the real pain lived in **infrastructure** (see the Hit section: dependency pinning,
> ephemeral env, wedged GPU, slow data-center storage). ✅ = went smoothly · ⚠️ = hit,
> see Hit section · ⏳ = still pending.

- [x] ⚠️ **Single-GPU "OOM" baseline — thesis DISPROVEN: it FIT on 1× A100 80 GB.** The
      punchline flipped. Two captures:
      1. **20 GB RTX 4000 Ada (2026-07-04):** OOM'd as expected — `torch.OutOfMemoryError:
         Tried to allocate 2.03 GiB. GPU 0 total 19.67 GiB, 1.36 GiB free, 18.31 GiB in use`,
         inside FSDP `all_gather_copy_in` at the first step. But "20 GB is too small" is
         unsurprising, so this was never the real baseline.
      2. **1× A100 80 GB (2026-07-05):** re-captured on the actual training card — and it
         **did NOT OOM.** `world_size=1` FULL_SHARD (= no sharding, whole model on one card):
         model init 17.80 GiB, optimizer + loss initialized, ran both steps (loss 2.97),
         checkpoint saved successfully. `nvidia-smi` peaked at **80,691 MiB / 81,920 MiB =
         98.5% full** — it fit with ~1.2 GB to spare (activation checkpointing ON, bf16).
      **So the back-of-envelope "~120 GB, won't fit on 80 GB" is wrong in practice:** with
      activation checkpointing + pure-bf16 the real footprint lands at ~80.7 GB, right at the
      ceiling. torchtune even hinted at the headroom: *"enable_activation_offloading isn't
      [on]. Enabling activation offloading should reduce memory further."*
      **Article angle (stronger than the OOM would've been):** the case for FSDP here is NOT
      "it doesn't fit" — it's "it fits so tightly you can't *train* on it." At 98.5% util,
      activation checkpointing is mandatory (not a choice), the batch can't grow, and the run
      survives only because it's 2 steps long. Sharded across 2 cards the same state is
      ~10 GiB each → checkpointing becomes optional, batch can grow, GPUs do real work. FSDP
      crosses from "fits, barely, uselessly" to "fits with room to actually work," not from
      "OOM" to "fits." `baseline_oom_test.sh` is misnamed for the 80 GB case — it completes.
- [x] ⚠️ **FSDP checkpoint format** — HIT, but not as a sharding problem: torchtune's
      `FullModelHFCheckpointer` writes `hf_model_*.pt` to the output ROOT (not `epoch_N/`,
      not HF-loadable, no tokenizer). Fixed with `eval/to_hf.py`. See the Hit entry. The
      "#1 classic footgun" prediction was right — just for a different reason (format/
      loadability, not `FULL_` vs `SHARDED_STATE_DICT`).
- [x] ✅ **Auto-wrap policy** — SMOOTH. torchtune wrapped decoder layers correctly; per-GPU
      memory was ~9–10 GiB (vs 17.8 GiB single-rank) → sharding confirmed working.
- [x] ✅ **Activation checkpointing** — SMOOTH. Default on, no OOM at seq 4096. (The only
      wobble was *suggesting* `=False` for speed, which risked OOM and wasn't needed.)
- [x] ✅ **bf16 vs fp16** — SMOOTH. bf16, no divergence.
- [x] ✅ **Gradient clipping under FSDP** — SMOOTH. `clip_grad_norm` handled it.
- [x] ✅ **Data loader starving GPUs** — SMOOTH during training (~50% MFU, no sawtooth). The
      slow I/O was one-time packing (~45 min) + the slow-DC storage stall, not train-time
      dataloader starvation.
- [x] ✅ **NCCL / torchrun init** — SMOOTH. Both ranks synced and sharded the model over the
      node's GPU interconnect. (The "process group NOT destroyed" warnings were symptoms of
      the wedged-GPU crashes, not an NCCL config problem — see Hit section.) NOTE: don't
      assume NVLink from "SXM" — verify with `nvidia-smi topo -m` (NV# = NVLink; PIX/PHB/SYS
      = PCIe/host bridge). SXM is a form factor; NVLink is a separate interconnect.
- [x] ✅ **FSDP1 vs FSDP2** — SMOOTH. FSDP2 per-parameter sharding, no API surprises.
- [ ] ⏳ **Resume from checkpoint** — PENDING/untested (optional; single-epoch run didn't
      need it).
- [x] ✅ **Throughput / MFU** — MEASURED: ~7.9k tok/s aggregate, ~50% MFU on 2× A100
      (recorded in DECISIONS.md). Healthy, not a fight.
- [x] ✅ **Tokenizer / packing correctness** — SMOOTH. Loss trended sanely (2.88→2.62) and
      generations are coherent on-format HN replies → no packing/boundary bug.
- [x] ✅ **`checkpoint_files` mismatch** — SMOOTH. The 4-shard list matched the download.
- [x] ✅ **`fused=True` AdamW** — SMOOTH. No NaNs / fused-kernel errors.

## Hit
<!-- move items here with a filled-in entry as they actually occur -->

### One slow data center masqueraded as TWO different bugs (co-location was NOT the issue)  [HIT 2026-07-05]
**Symptom:** On EU pods the run was (a) painfully slow to load the checkpoint —
`folio_wait_bit_common` in D-state, `read_bytes: 0` (mmap page-ins don't count there),
GPU memory crawling up — and (b) `torch.compile` appeared to *hang*: dozens of
`torch/_inductor/compile_worker` processes parked in `futex_`/`pipe_r`, recipe ranks in
`Dsl`, CPU ~94% idle, load average ~16 and climbing. The same code on a **US pod** was fast
AND `torch.compile` completed normally.
**Cause:** One root cause — slow network-volume storage — wearing two masks. It was NOT a
region-mismatch: the EU volume was **co-located with the GPU (both in EU-IS-1 / Iceland)**.
That data center's network-volume storage was simply slow. The slow mmap-backed reads
throttled the 15 GB checkpoint load, AND they starved Inductor, whose compile workers
read/write compilation artifacts to disk — so `torch.compile` stalled on I/O and looked
like a compile/FSDP2 deadlock. It wasn't: the same pod stack on fast US storage compiled
fine. Co-location is necessary but NOT sufficient — the DC's storage has to be fast.
**Fix:** Network-volume throughput varies a LOT by RunPod data center (EU-IS-1 was slow;
US was fast). **Benchmark storage before committing a multi-hour run** — e.g. time a cold
read of the model dir (`time cat /workspace/models/Qwen2.5-7B/*.safetensors > /dev/null`)
or a `dd` write test. If it's slow, pick a different DC, or stage model + data on the
pod's LOCAL NVMe and write outputs locally (`checkpointer.checkpoint_dir=/root/...
dataset.data_files=/root/... output_dir=/root/...`), copying the final checkpoint back.
**Article angle:** Two "bugs" — a slow load and a `torch.compile` hang — collapsed into one
infra fact: the volume was slow. Co-locating compute and storage isn't enough; some data
centers' network volumes are just slow, and that starves both the checkpoint load and
compile's on-disk artifact I/O. Diagnostic signature of storage-bound (not a framework
bug): idle CPU + high load average + D-state workers + `folio_wait`/`read_bytes: 0`.
Benchmark volume throughput first. And `torch.compile` + FSDP2 is fine — the "compile
deadlock" hypothesis was wrong; it was I/O all along.

### GPU wedged: "device(s) is/are busy or unavailable" with a clean nvidia-smi  [HIT 2026-07-05]
**Symptom:** After fixing the env (torch saw both cards), `tune run` still died at rank 0.
The elastic wrapper hides the real error — the actual worker traceback is ABOVE the
`closing signal SIGTERM` lines (or `grep` the tee'd log). It was:
```
torch.empty(0, device=device)
RuntimeError: CUDA error: CUDA-capable device(s) is/are busy or unavailable
```
`torch.cuda.is_available()` returned True and `device_count()==2`, but a bare
`torch.empty(1, device='cuda:0')` failed on BOTH cards — while `nvidia-smi` showed them
idle (0 MiB, no processes, Default compute mode).
**Cause:** The device was wedged at the driver/host level (CUDA error 46,
`cudaErrorDevicesUnavailable`), most likely from repeated ungraceful NCCL exits — every
failed launch logged "process group has NOT been destroyed." Key insight:
`is_available()`/`device_count()` only *enumerate* devices; they don't create a context,
so they pass while the first real allocation fails.
**Fix:** Not clearable from inside the container (no `nvidia-smi --gpu-reset` privilege).
`pkill` + `fuser -k /dev/nvidia*` did NOT help — nothing was holding the cards. **Restart
the pod** (`/workspace` persists) → re-run `setup_pod.sh` → gate on a bare allocation
before launching. If a freshly restarted pod fails the bare-alloc test with nothing else
running, it's a **bad host GPU** — reprovision (likely a different host).
**Isolating diagnostic:** `python -c "import torch; torch.empty(1, device='cuda:0'); \
torch.empty(1, device='cuda:1'); print('OK')"` — a bare context/alloc, no torchtune.
Passing = env/process issue; failing with a clean `nvidia-smi` = wedged host, restart.
**Article angle:** `torch.cuda.is_available()==True` is necessary but NOT sufficient — it
enumerates, it doesn't allocate. The real liveness check is a zero-size allocation.
"Kill the process" only helps if a process is holding it; a driver-level wedge needs a
pod restart — and stop hammering the launcher, since each ungraceful exit deepens the hole.

### Reusing a network volume on a new pod: env is gone, "cuda:0 is not available"  [HIT 2026-07-05]
**Symptom:** Swapped to a fresh GPU pod backed by the same RunPod network volume, launched
training, and:
```
RuntimeError: The device cuda:0 is not available on this machine.
  ... torchtune/utils/_device.py, _validate_device_from_env(device)
```
even though `nvidia-smi` clearly showed 2× A100-SXM4-80GB attached and idle.
**Cause:** A RunPod **network volume only persists `/workspace`** — model weights, data, and
output checkpoints survived, but the Python environment did NOT. Everything installed with
`uv pip install --system` lives in the container image at `/usr/local/lib/python3.11/dist-
packages`, which is ephemeral and reset per pod. So the new pod was running the base image's
default (wrong/old) torch, which couldn't bind the GPU. After re-running `setup_pod.sh`,
`python -c "import torch; ...` reported `avail True | count 2` — torch could see the cards
again (a *separate* device-wedge then surfaced; see the entry above, which needed a pod
restart, not an env fix).
**Fix:** Re-run `bash scripts/setup_pod.sh` on **every** new pod, even when the volume is
reused. Added a GPU-visibility gate at the end of setup that asserts
`torch.cuda.is_available()` and `device_count() > 0`, so a bad env/pod fails loudly at setup
with the actual numbers instead of deep inside torchtune. Red herring along the way:
`echo $CUDA_VISIBLE_DEVICES` printed `[]` — that's an *unset* var (harmless); an empty-string
value would hide all GPUs, but torch reporting `count 2` proved it was unset.
**Article angle:** "The volume persisted, so my setup persisted" is the trap. Persistent
storage ≠ persistent environment — `/workspace` is durable, the container's site-packages are
cattle. Treat `setup_pod.sh` as mandatory per-pod, and gate on GPU visibility so the failure
is one clear line at setup, not a cryptic device error mid-recipe.

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
