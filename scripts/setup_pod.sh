#!/usr/bin/env bash
# One-shot environment setup on a fresh RunPod pod (PyTorch base image assumed:
# torch + CUDA already present). Run once after cloning the repo onto the pod.
set -euo pipefail

echo "[setup] GPU / interconnect topology:"
nvidia-smi --query-gpu=name,memory.total --format=csv || true
nvidia-smi topo -m || true   # look for 'NV#' (NVLink) vs 'PIX/PHB' (PCIe) between GPUs

echo "[setup] torch version:"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'gpus', torch.cuda.device_count())"

echo "[setup] installing uv..."
if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

# Ensure torch is new enough for pinned torchtune 0.4.0 / torchao 0.7.0 (need >= 2.5).
# torchtune/torchao don't pin torch, so an old base-image torch (e.g. 2.4.1) otherwise
# slips through and blows up at import. Only install if too OLD — never downgrade a
# newer base image. Matched CUDA build from the PyTorch cu124 index.
if python -c "import sys,torch; v=tuple(map(int,torch.__version__.split('+')[0].split('.')[:2])); sys.exit(0 if v>=(2,5) else 1)"; then
    echo "[setup] torch OK (>= 2.5) for the pinned torchtune/torchao."
else
    echo "[setup] torch too old for pinned deps; installing torch 2.5.1+cu124..."
    uv pip install --system torch==2.5.1 torchvision==0.20.1 \
        --index-url https://download.pytorch.org/whl/cu124
fi

# --system installs on top of the base image's Python (where torch + CUDA already
# live), instead of a fresh venv that would shadow the preinstalled CUDA torch.
echo "[setup] installing python deps with uv..."
uv pip install --system -r requirements.txt

# Optional: live training charts. Export WANDB_API_KEY and flip the metric_logger
# in the config to WandBLogger to use it.
#   export WANDB_API_KEY=...
#   wandb login

echo "[setup] mkdir workspace dirs..."
mkdir -p /workspace/models /workspace/data /workspace/output

# GPU-visibility gate. A RunPod network volume carries /workspace but NOT the Python
# env, so every new pod must re-run this script — and a pod can come up with a driver
# mismatch, no GPU attached, or CUDA_VISIBLE_DEVICES clobbered. Fail LOUDLY here with the
# actual numbers instead of deep inside torchtune ("cuda:0 is not available"). If this
# fires: check `nvidia-smi`, `echo $CUDA_VISIBLE_DEVICES` (empty-string hides all GPUs),
# and that torch's cuda build matches the driver.
echo "[setup] verifying torch can see the GPU(s)..."
python - <<'PY'
import sys, torch
avail, n = torch.cuda.is_available(), torch.cuda.device_count()
print(f"[setup]   torch {torch.__version__} | cuda {torch.version.cuda} | "
      f"available={avail} | device_count={n}")
if not avail or n == 0:
    sys.exit("[setup] FATAL: torch cannot see a GPU. nvidia-smi should list the cards; "
             "if it does, suspect a CPU-only torch, a driver older than the cuda build, "
             "or CUDA_VISIBLE_DEVICES set to an empty string.")
PY

echo "[setup] done. Next: bash scripts/download_model.sh"
