#!/usr/bin/env bash
# Depth Anything V2 — Mac (MPS) and Linux (CUDA). Windows: use setup_depth_anything_v2.ps1
# Usage: bash scripts/setup_depth_anything_v2.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
THIRD_PARTY="${REPO_ROOT}/third_party"
DA_DIR="${THIRD_PARTY}/Depth-Anything-V2"
ENV_NAME="${DEPTH_ANYTHING_V2_ENV:-depth_anything_v2}"
CKPT_DIR="${DA_DIR}/checkpoints"
CKPT="${CKPT_DIR}/depth_anything_v2_vits.pth"
VITS_URL="https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth"

mkdir -p "${THIRD_PARTY}"

if [[ ! -d "${DA_DIR}/.git" ]]; then
  echo "Cloning Depth-Anything-V2 into ${DA_DIR} ..."
  git clone --depth 1 https://github.com/DepthAnything/Depth-Anything-V2.git "${DA_DIR}"
else
  echo "Depth-Anything-V2 already at ${DA_DIR}"
fi

if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: conda not found. Install Miniconda first."
  exit 1
fi

if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "Conda env '${ENV_NAME}' already exists."
else
  echo "Creating conda env '${ENV_NAME}' (Python 3.11) ..."
  conda create -y -n "${ENV_NAME}" python=3.11 pip

  if [[ "$(uname -s)" == "Darwin" ]]; then
    echo "Installing PyTorch (Mac CPU/MPS) ..."
    conda run -n "${ENV_NAME}" pip install torch torchvision
  else
    CUDA_INDEX="${PYTORCH_CUDA_INDEX:-https://download.pytorch.org/whl/cu124}"
    echo "Installing PyTorch with CUDA (${CUDA_INDEX}) ..."
    conda run -n "${ENV_NAME}" pip install torch torchvision --index-url "${CUDA_INDEX}"
  fi

  echo "Installing runtime deps ..."
  conda run -n "${ENV_NAME}" pip install opencv-python numpy matplotlib
fi

mkdir -p "${CKPT_DIR}"
if [[ ! -f "${CKPT}" ]]; then
  echo "Downloading Depth-Anything-V2-Small checkpoint ..."
  curl -L --fail -o "${CKPT}" "${VITS_URL}"
else
  echo "Checkpoint: ${CKPT}"
fi

echo ""
echo "Verify device:"
echo "  conda run -n ${ENV_NAME} python -c \"import torch; print('cuda', torch.cuda.is_available(), 'mps', getattr(torch.backends,'mps',None) and torch.backends.mps.is_available())\""
echo ""
echo "Then:"
echo "  cd stereo_lidar_pointcloud"
echo "  python 02_make_stereo_pointcloud.py --run latest"
echo "  python 02_make_depth_anything_pointcloud.py --run latest --reuse-rectified"
