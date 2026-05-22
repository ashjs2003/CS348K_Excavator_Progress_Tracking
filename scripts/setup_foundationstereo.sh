#!/usr/bin/env bash
# Clone FoundationStereo and create the conda env (CUDA GPU machine required).
# Usage: bash scripts/setup_foundationstereo.sh
#
# On macOS this script exits successfully WITHOUT creating the env (so Mac stays clean).
# Use scripts/setup_foundationstereo.ps1 on Windows with an NVIDIA GPU.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
THIRD_PARTY="${REPO_ROOT}/third_party"
FS_DIR="${THIRD_PARTY}/FoundationStereo"
ENV_NAME="${FOUNDATION_STEREO_ENV:-foundation_stereo}"

mkdir -p "${THIRD_PARTY}"

if [[ ! -d "${FS_DIR}/.git" ]]; then
  echo "Cloning FoundationStereo into ${FS_DIR} ..."
  git clone --depth 1 https://github.com/NVlabs/FoundationStereo.git "${FS_DIR}"
else
  echo "FoundationStereo already cloned at ${FS_DIR}"
fi

if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: conda not found. Install Miniconda, then re-run this script."
  exit 1
fi

if [[ "$(uname -s)" == "Darwin" && "${SETUP_FOUNDATION_ON_MAC:-0}" != "1" ]]; then
  echo "macOS detected — skipping foundation_stereo conda env (no NVIDIA CUDA here)."
  echo "  Your Mac pipeline is unchanged: python 02_make_stereo_pointcloud.py (OpenCV)."
  echo "  On Windows + NVIDIA GPU, run:"
  echo "    powershell -ExecutionPolicy Bypass -File scripts\\setup_foundationstereo.ps1"
  echo "  Optional: clone weights on Windows only; third_party/ is gitignored."
  exit 0
fi

if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "Conda env '${ENV_NAME}' already exists."
  echo "  To recreate: conda env remove -n ${ENV_NAME} && bash scripts/setup_foundationstereo.sh"
else
  echo "Creating conda env '${ENV_NAME}' (Python 3.11) ..."
  conda create -y -n "${ENV_NAME}" python=3.11 pip

  echo "Installing PyTorch first (xformers needs torch at build time) ..."
  conda run -n "${ENV_NAME}" pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1

  echo "Installing FoundationStereo Python deps (without xformers) ..."
  conda run -n "${ENV_NAME}" pip install \
    scikit-image omegaconf opencv-contrib-python imgaug ninja timm albumentations \
    jupyterlab scipy joblib scikit-learn ruamel.yaml trimesh pyyaml imageio open3d \
    transformations einops gdown nodejs huggingface-hub

  echo "Installing xformers (optional; skip if this fails on Mac) ..."
  conda run -n "${ENV_NAME}" pip install xformers==0.0.28.post1 || {
    echo "WARNING: xformers install failed. Inference may still work with XFORMERS_DISABLED=1."
  }

  echo "Installing flash-attn (optional; CUDA Linux only) ..."
  conda run -n "${ENV_NAME}" pip install flash-attn || {
    echo "WARNING: flash-attn install failed. See FoundationStereo FAQ if inference fails."
  }
fi

CKPT_DIR="${FS_DIR}/pretrained_models/23-51-11"
CKPT="${CKPT_DIR}/model_best_bp2.pth"
if [[ ! -f "${CKPT}" ]]; then
  echo ""
  echo "=== Download model weights (manual step) ==="
  echo "1. Open: https://github.com/NVlabs/FoundationStereo#model-weights"
  echo "2. Download folder 23-51-11 from Google Drive"
  echo "3. Place it at: ${CKPT_DIR}/"
  echo "   (must contain model_best_bp2.pth and cfg.yaml)"
  echo ""
else
  echo "Checkpoint found: ${CKPT}"
fi

echo ""
echo "Setup complete. Test CUDA:"
echo "  conda run -n ${ENV_NAME} python -c \"import torch; print('cuda', torch.cuda.is_available())\""
echo ""
echo "Then from stereo_lidar_pointcloud/:"
echo "  python 02_make_stereo_pointcloud.py --run latest"
echo "  python 02_make_stereo_pointcloud_foundation.py --run latest --reuse-rectified"
echo "  python compare_stereo_methods.py --run latest"
