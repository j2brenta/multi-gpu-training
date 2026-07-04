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

echo "[setup] done. Next: bash scripts/download_model.sh"
