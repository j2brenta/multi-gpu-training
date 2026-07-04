# Multi-GPU FSDP Fine-Tuning: Hacker News LM

A weekend proof-of-capability: full-parameter fine-tune (continued pre-training)
of a 7–8B base LLM on Hacker News comments, sharded across 2+ GPUs with **PyTorch
FSDP**, rented on RunPod. Every non-obvious decision and every challenge is logged
in [`DECISIONS.md`](DECISIONS.md) and [`CHALLENGES.md`](CHALLENGES.md) so the
article can be structured afterwards.

## The one-sentence thesis

A 7–8B model fine-tuned with **full parameters** (params + grads + Adam optimizer
states ≈ 16 bytes/param ≈ ~120 GB) **does not fit on a single GPU**. FSDP shards
that state across ranks so it trains anyway. That's the whole point — and it's why
this repo deliberately does **not** use LoRA/QLoRA (which fits on one 24 GB card and
would make the multi-GPU setup pointless).

## Stack decisions (see DECISIONS.md for the why)

| Decision            | Choice                                              |
|---------------------|-----------------------------------------------------|
| Framework           | **PyTorch** (FSDP is PyTorch-native; TF has no FSDP) |
| Trainer             | **torchtune** `full_finetune_distributed` (FSDP2)   |
| Model               | **Qwen2.5-7B (base)** — Apache-2.0, no gated wait   |
| Fine-tune type      | **Full-parameter** (not LoRA), to justify sharding  |
| Hardware            | 2× A100/H100 80 GB (NVLink), single RunPod node     |
| Task                | Continued pre-training (next-token) on raw comments |

## Quickstart on a fresh RunPod pod

```bash
# 0. Clone this repo onto the pod, then:
bash scripts/setup_pod.sh            # installs uv, then torchtune, polars, etc.
bash scripts/download_model.sh       # pulls Qwen2.5-7B base into /workspace/models

# 1. Prepare data (put the HN full-export parquet at /workspace/data/hn_raw.parquet first)
#    --mode raw        : one comment per document (simplest continued pre-training)
#    --mode reply      : (immediate parent -> reply) pairs, so the model is promptable
#    --mode reply_root : (root story "[Story] title (domain)" + parent -> reply) —
#                        anchors each comment to the article it reacts to (topic + source)
#    --holdout-frac 0.02 : split off 2% as disjoint held-out text for perplexity eval
python data/prepare.py \
    --input /workspace/data/hn_raw.parquet \
    --output /workspace/data/hn_prepared.parquet \
    --mode reply_root \
    --target-tokens 200_000_000 \
    --holdout-frac 0.02

# 2. (Article narrative) Show it OOMs on ONE GPU — same config, 1 rank, no sharding:
bash scripts/baseline_oom_test.sh    # expected: CUDA OOM

# 3. Now shard across 2 GPUs and actually train:
bash scripts/launch_train.sh         # tune run --nproc_per_node 2 ...

# 4. Sanity-check the fine-tuned model (add --reply-mode if you trained with --mode reply):
python eval/generate.py --model-dir /workspace/output/qwen2_5_7B_hn/epoch_0 --reply-mode

# 5. PROVE it beat base: held-out perplexity, base vs fine-tuned (lower = learned HN).
python eval/perplexity.py --model-dir /workspace/models/Qwen2.5-7B \
    --data /workspace/data/hn_prepared.holdout.parquet          # the "before"
python eval/perplexity.py --model-dir /workspace/output/qwen2_5_7B_hn/epoch_0 \
    --data /workspace/data/hn_prepared.holdout.parquet          # the "after" (lower)
```

The **step 2 → step 3 before/after** (OOM on one card, works when sharded) is the
single most convincing beat for the article — run it and screenshot both. **Step 5**
is the quantitative payoff: same held-out HN text, fine-tuned perplexity clearly below
base. (Point `--data` at a non-HN parquet to check the fine-tune didn't *forget* general
English — that perplexity should stay roughly flat, not spike.)

## Cost / time budget

- 2× A100 80 GB community cloud ≈ $1.5–2/hr each → **~$20–40** for a 5–10 h run.
- Throughput on 2× A100 ≈ 5–10k tokens/s aggregate → **~150–300M tokens** in ~8 h,
  so we **subsample** the 5.5 GB down to a fixed token budget (see `prepare.py`).
  You are *not* doing a full epoch over 5.5 GB; that's a deliberate decision.

## Layout

```
configs/qwen2_5_7B_full_fsdp.yaml   torchtune FSDP full-finetune config
data/prepare.py                     parquet → cleaned, subsampled text parquet
scripts/setup_pod.sh                one-shot env install on the pod
scripts/download_model.sh           fetch Qwen2.5-7B base weights
scripts/baseline_oom_test.sh        single-rank run that should OOM (the "before")
scripts/launch_train.sh             2-GPU FSDP training launch (the "after")
eval/generate.py                    load fine-tuned HF checkpoint, generate samples
eval/perplexity.py                  held-out perplexity, base vs fine-tuned (the proof)
DECISIONS.md                        running decision log (pre-seeded)
CHALLENGES.md                       running challenge/gotcha log (pre-seeded)
```

## Alternative trainers (if torchtune fights you)

- **HF Transformers `Trainer` + Accelerate** with an FSDP config (`accelerate config`)
  — more familiar to many readers; expose the same FSDP knobs.
- **Axolotl** — YAML-driven, batteries included.

torchtune is the primary path here because its recipe/config *is* the teaching
material for an FSDP article: the sharding, wrapping, and checkpointing are all
visible and editable.

## Future work (phase 2)

**Autoregressive → diffusion-LM conversion** (à la Sber's GFusion, open-sourced
2026-07-02) is explicitly **out of scope for this weekend** and parked as a follow-up.
It's a different axis, not an upgrade: it changes the training objective, sampling, and
serving stack all at once, and each reusable piece needs a real training run — not a
config change:

- **The AR→diffusion recipe** — possible on an open base (precedent: Dream / Dream-Coder
  initialized from Qwen), but GFusion's recipe is tuned to GigaChat's MoE architecture and
  tokenizer, so porting to dense Qwen/Llama is genuine adaptation.
- **Optimized attention kernels** (+60% vs Flex-Attention) — the one model-agnostic piece,
  but only pays off once you're already training a text-diffusion model.
- **The SGLang sampling algorithm** — accelerates *diffusion* LLMs only; does nothing for a
  stock autoregressive Qwen/Llama.

The bridge worth noting: the FSDP harness in this repo is exactly what such a run would
reuse (you still shard a 7–8B training job the same way). Phase 1 is the foundation for
phase 2, not throwaway setup. See `DECISIONS.md` for the full reasoning.
