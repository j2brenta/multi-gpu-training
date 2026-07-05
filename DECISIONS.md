# Decision Log

Append-only. Newest at top. Each entry: date, decision, the alternatives, and the
*why* — that "why" is what turns into article paragraphs. Keep it honest; record the
ones that turned out wrong too.

Template:

```
## YYYY-MM-DD — <short title>
**Decision:** ...
**Alternatives considered:** ...
**Why:** ...
**Revisit if:** ...
```

---

## 2026-07-05 — RESULT: fine-tune beat base, held-out perplexity 22.2 → 13.9 (−37%), no overfit
**Outcome:** First full run complete. Full-parameter continued pre-training of Qwen2.5-7B
(base) on HN `reply_root` data, FSDP2 via torchtune, 2× A100-SXM 80 GB in a **US** data
center (after EU-IS-1's slow storage forced a move — see CHALLENGES).
**Run:** 760 optimizer steps ≈ **199M tokens** (batch 4 × grad_accum 8 × 2 GPUs × 4096
seq); **~7 h** wall; **~7.9k tok/s** aggregate (**~50% MFU**); **~$21** (7 h × $2.98/hr).
`compile=False` (FSDP2 is fine — the earlier "compile hang" was slow-DC I/O, not compile).
Packing the oversized 6.35M-doc parquet cost ~45 min one-time; token budget enforced with
`max_steps_per_epoch=760` rather than re-preparing.
**The money number** — held-out perplexity (`eval/perplexity.py` on
`hn_prepared.holdout.parquet`; **167,446 tokens / 1,000 docs, identical for both**, shared
Qwen tokenizer → apples-to-apples):

| Model | Perplexity | Mean loss |
|-------|-----------|-----------|
| Base Qwen2.5-7B | 22.155 | 3.0981 |
| Fine-tuned      | **13.882** | **2.6306** |

→ **37% lower perplexity** on unseen HN text.
**No overfitting:** fine-tuned held-out loss (2.6306) ≈ final training loss (~2.616) — they
match, so it's generalization, not memorization (memorized weights would show held-out ≫
train).
**Still to confirm:** catastrophic-forgetting guardrail — general-English (wikitext)
perplexity should stay ~flat base vs fine-tuned (`eval/make_general_holdout.py` +
`perplexity.py`). Plus qualitative before/after via `generate.py --base-dir`.
**Why it matters:** This is the quantitative half of the thesis payoff (the other half:
OOM on 1 GPU vs works when sharded). A full-parameter FSDP fine-tune of a model too big for
one card produced a real, measurable, non-overfit domain gain.

## 2026-07-04 — Verify "it worked" with held-out perplexity, not just vibes
**Decision:** Make held-out **perplexity, base vs fine-tuned** the primary success metric.
Add `--holdout-frac` to `prepare.py` (splits a disjoint sample off the already-shuffled,
budget-trimmed docs into `<stem>.holdout.parquet`) and `eval/perplexity.py` (token-weighted
corpus perplexity for any HF checkpoint on that file). `eval/generate.py` (side-by-side
samples) stays as the qualitative/screenshot signal.
**Alternatives considered:** Qualitative generation only; a downstream benchmark (MMLU etc.);
train-loss curve as the headline number.
**Why:** The objective *is* next-token prediction on HN, so perplexity on HN text the model
never saw is the honest, directly-aligned measure — and it's cheap (forward passes only).
Generation alone is unfalsifiable cherry-picking; train loss falling doesn't prove it beat
base; a general benchmark measures the wrong thing (we didn't train for MMLU). Fairness holds
because base and fine-tuned share the Qwen tokenizer (identical token counts). Held-out (not
train) is essential — train-set perplexity is inflated by memorization. Same script on a
non-HN parquet doubles as the catastrophic-forgetting guardrail: HN perplexity should drop
while general-English perplexity stays ~flat; a spike there means lr/steps were too hot.
**Verified:** `--holdout-frac 0.05` on the local sample produced a disjoint split (train
3,154 / holdout 165, overlap 0); `perplexity.py` ran end-to-end under uv on a tiny model
(token-weighted NLL→exp path confirmed). Real base-vs-finetuned numbers pending the pod run.
**Revisit if:** perplexity drops but samples look worse (over-fit to formatting) → add a
small held-out generation set scored by an LLM judge, or a cloze/idiom-completion probe.

## 2026-07-04 — Add `reply_root`: anchor comments to the root story (topic + source)
**Decision:** Add a third `prepare.py` mode, `reply_root`, alongside `raw`/`reply`. It
climbs each comment's parent chain to the submitted **root story** and builds
`"[Story] <title> (<domain>)" [+ "> <immediate parent comment>"] — reply — <comment>"`.
The root walk is pointer-doubling over the item table (each pass doubles reachable
depth, ~log2(depth) passes, early-stop at fixpoint), run eagerly on an (id, parent,
type) projection. Parent context is length-capped (`--max-context-chars`, default 1000).
**Alternatives considered:** Leave `reply` (immediate-parent-only) as the steerable mode;
a recursive per-row walk; a second dataset for topic labels.
**Why:** `reply` gives conversational context but a comment deep in a thread loses the
*article it reacts to* — the topic and source that make it interpretable/promptable.
`reply_root` restores that (prompt "here's an article about X → what would HN say?").
This reverses the 2026-07-04 call to skip the root walk as "recursive, fiddly": pointer-
doubling makes it a handful of set-based self-joins, not per-row recursion — cheap enough
to justify the richer artifact. `raw`/`reply` are unchanged and remain the low-friction
defaults; `reply_root` is opt-in.
**Verified:** ran all three modes on a 200k-item local sample. `reply_root`: 6,635/6,644
docs carry the `[Story]` anchor (broken/deleted chains fall back to parent-only context),
3,640 also quote the immediate parent; direct-to-story replies show anchor only (no dup).
**Revisit if:** root titles/domains add too little signal vs. the extra full-table walk on
the 5.5 GB input (walk is eager — watch peak RAM on a laptop; fine on an 80 GB pod).

## 2026-07-04 — Dataset confirmed: HN full API export; add `reply` framing mode
**Decision:** Use the Kaggle "official HN API export" (~38M items: stories + comments +
polls, one parquet, standard Firebase item schema). Keep HN over swapping datasets. Add a
`--mode reply` to `prepare.py` alongside `--mode raw`.
**Alternatives considered:** Switch to an instruction dataset (Tulu/OpenHermes) for a more
"normal" SFT story; BigQuery HN export instead of Kaggle.
**Why:** For an FSDP article the dataset is a vehicle — judge it by low pipeline friction +
a crisp screenshottable result, both of which HN wins. The export being *full* (stories and
comments in one file) means the steerable framing is a single self-join, not a second
dataset: `reply` mode joins each comment to its immediate parent → "(context) — reply —
(reply)" documents, so the model becomes promptable at inference (context + `REPLY_SEP`).
Chose the immediate-parent join over walking to the root story title (recursive, fiddly);
immediate parent is conversational and one join. `raw` mode stays the default/simplest.
**Verified:** smoke-tested both modes on synthetic HN-schema data — dead/deleted dropped,
HTML unescaped, story-parent uses `title`, comment-parent uses parent `text`.
**Open:** confirm the Kaggle dataset's **license** before publishing generated samples or
weights (HN content is user-generated; Kaggle re-host terms vary). BigQuery export is the
cleaner-provenance fallback.
**Revisit if:** reply-mode context bloats sequences (parent story bodies can be long) →
cap parent context length or fall back to title-only context.

## 2026-07-03 — Parked: GFusion / diffusion-LM conversion is phase 2, not this weekend
**Decision:** Do NOT attempt an autoregressive→diffusion conversion (à la Sber's GFusion,
open-sourced 2026-07-02) in this run. Log it as future work.
**Alternatives considered:** Fold a diffusion-conversion experiment into the weekend run.
**Why:** It's a different axis, not an upgrade. Phase 1 has one sharp thesis — full-param
FSDP fine-tuning of an AR model too big for one GPU. Diffusion conversion changes the
objective, sampling, and serving stack at once; it would split the article into two
half-told stories and blow the 5–10 h budget. The reusable GFusion pieces each require a
real training run, not a config change: (1) the AR→diffusion recipe (precedent: Dream /
Dream-Coder from Qwen — possible, but their recipe is tuned to GigaChat's MoE arch +
tokenizer, so porting to dense Qwen/Llama is real adaptation); (2) optimized attention
kernels (+60% vs Flex-Attention — model-agnostic, but only pays off once you're already
training a diffusion model); (3) the SGLang sampler (accelerates *diffusion* LLMs only —
does nothing for a stock AR Qwen).
**Bridge to phase 2:** The FSDP harness built here is exactly what a diffusion-conversion
run would reuse (still sharding a 7–8B training job the same way). Phase 1 is the
foundation for phase 2, not throwaway setup — a good closing hook for the article.
**Note:** GFusion postdates this author's/assistant's knowledge cutoff; the above is from
a secondhand summary, verify specifics before acting.
**Revisit if:** doing a follow-up post on open-base diffusion LLMs.

## 2026-07-03 — Framework: PyTorch, not TensorFlow
**Decision:** PyTorch.
**Alternatives considered:** TensorFlow (`tf.distribute`, DTensor).
**Why:** FSDP is a PyTorch-native API (`torch.distributed.fsdp`, now FSDP2). TensorFlow
has no FSDP equivalent, and the entire LLM fine-tuning ecosystem (HF, torchtune,
Axolotl, DeepSpeed) is PyTorch. An FSDP article is a PyTorch article by definition.
**Revisit if:** never, for this project.

## 2026-07-03 — Trainer: torchtune `full_finetune_distributed`
**Decision:** torchtune, FSDP2 recipe.
**Alternatives considered:** HF Trainer + Accelerate FSDP; Axolotl; raw torch FSDP loop.
**Why:** torchtune is PyTorch-native and its recipe/config exposes sharding strategy,
auto-wrap, activation checkpointing, and checkpoint format directly — i.e. the config
*is* the teaching material. Raw FSDP loop is more educational but more weekend-risk.
**Revisit if:** torchtune's Qwen2.5 builders/checkpointer fight us → fall back to HF
Accelerate FSDP (already noted as plan B in README).

## 2026-07-03 — Model: Qwen2.5-7B (base)
**Decision:** Qwen2.5-7B, the **base** (not -instruct) checkpoint.
**Alternatives considered:** Llama 3.1 8B (gated HF access, canonical FSDP demo);
Qwen3-8B (newer, less trodden); smaller 1–3B models.
**Why:** Apache-2.0 → no gated-access wait on a weekend. 7B is the sweet spot: full FT
overflows one GPU (forces FSDP) but still finishes in 5–10 h. Base model because raw HN
comments = next-token continued pre-training, and base is the honest starting point.
**Revisit if:** want a bigger "wow" or NVLink headroom → swap to Llama 3.1 8B (request
gated access the day before).

## 2026-07-03 — Full-parameter fine-tune, NOT LoRA/QLoRA
**Decision:** Full-parameter fine-tune.
**Alternatives considered:** LoRA, QLoRA.
**Why:** This is the crux of the article. QLoRA of a 7B fits on one 24 GB 4090 — if we
used it, the multi-GPU story collapses ("why 2 GPUs?"). Full FT needs ~16 bytes/param of
state (~120 GB for 8B), which genuinely requires sharding. FSDP earns its keep only here.
**Revisit if:** OOM even at 2×80 GB with offload → drop to LoRA and pivot the article to
"FSDP + LoRA at larger scale", but that's a different piece.

## 2026-07-03 — Hardware: 2× A100/H100 80 GB, single node, NVLink
**Decision:** 2× 80 GB on one RunPod node, prefer NVLink/SXM.
**Alternatives considered:** 2× 48 GB (A6000/A40) with CPU offload; multi-node.
**Why:** 80 GB cards make full 7B FT comfortable without heavy offload. FSDP is
communication-heavy (all-gather / reduce-scatter each layer) so interconnect matters;
NVLink >> PCIe. Single node avoids multi-node NCCL setup pain on a weekend. ~$20–40 total.
**Revisit if:** want to demonstrate scaling 2→4→8 GPUs for the article's throughput chart.

## 2026-07-03 — Data: subsample to a fixed token budget, not a full epoch
**Decision:** Subsample the 5.5 GB parquet to ~200M tokens (configurable) via `prepare.py`.
**Alternatives considered:** Full epoch over all 5.5 GB.
**Why:** At ~5–10k tokens/s aggregate, ~8 h buys ~150–300M tokens — far less than the full
corpus. A fixed token budget makes the run bounded and reproducible. Sampling method and
budget are themselves a logged decision.
**Revisit if:** throughput is higher than expected → raise `--target-tokens`.

## 2026-07-03 — Sequence packing at max_seq_len 4096
**Decision:** Pack cleaned comments into 4096-token sequences (`packed: True`).
**Alternatives considered:** No packing (pad to longest); shorter 2048 context.
**Why:** HN comments are short; without packing most of each sequence is padding and GPUs
waste FLOPs. Packing maximizes tokens/s. 4096 balances context vs activation memory.
**Revisit if:** activation memory OOM → drop to 2048.
