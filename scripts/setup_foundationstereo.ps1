# FoundationStereo setup for Windows + NVIDIA CUDA.
# Run from repo root in PowerShell:
#   powershell -ExecutionPolicy Bypass -File scripts\setup_foundationstereo.ps1
#
# Does NOT affect your normal Python env — only conda env "foundation_stereo".

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$ThirdParty = Join-Path $RepoRoot "third_party"
$FsDir = Join-Path $ThirdParty "FoundationStereo"
$EnvName = if ($env:FOUNDATION_STEREO_ENV) { $env:FOUNDATION_STEREO_ENV } else { "foundation_stereo" }
$CudaIndex = if ($env:PYTORCH_CUDA_INDEX) { $env:PYTORCH_CUDA_INDEX } else { "https://download.pytorch.org/whl/cu124" }

function Test-Conda {
    $conda = Get-Command conda -ErrorAction SilentlyContinue
    if (-not $conda) {
        Write-Host "ERROR: conda not found. Install Miniconda for Windows first."
        exit 1
    }
}

function Test-NvidiaGpu {
    $smi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
    if (-not $smi) {
        Write-Host "WARNING: nvidia-smi not found. Install NVIDIA drivers + CUDA toolkit for GPU inference."
    } else {
        & nvidia-smi
    }
}

New-Item -ItemType Directory -Force -Path $ThirdParty | Out-Null

if (-not (Test-Path (Join-Path $FsDir ".git"))) {
    Write-Host "Cloning FoundationStereo into $FsDir ..."
    git clone --depth 1 https://github.com/NVlabs/FoundationStereo.git $FsDir
} else {
    Write-Host "FoundationStereo already cloned at $FsDir"
}

Test-Conda
Test-NvidiaGpu

$envList = conda env list 2>&1 | Out-String
if ($envList -match "(?m)^$EnvName\s") {
    Write-Host "Conda env '$EnvName' already exists."
    Write-Host "  Remove with: conda env remove -n $EnvName"
} else {
    Write-Host "Creating conda env '$EnvName' (Python 3.11) ..."
    conda create -y -n $EnvName python=3.11 pip

    Write-Host "Installing PyTorch with CUDA ($CudaIndex) ..."
    conda run -n $EnvName pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url $CudaIndex

    Write-Host "Installing FoundationStereo dependencies ..."
    conda run -n $EnvName pip install `
        scikit-image omegaconf opencv-contrib-python imgaug ninja timm albumentations `
        jupyterlab scipy joblib scikit-learn ruamel.yaml trimesh pyyaml imageio open3d `
        transformations einops gdown nodejs huggingface-hub

    Write-Host "Installing xformers ..."
    conda run -n $EnvName pip install xformers==0.0.28.post1

    Write-Host "Installing flash-attn (may take several minutes) ..."
    conda run -n $EnvName pip install flash-attn
    if ($LASTEXITCODE -ne 0) {
        Write-Host "WARNING: flash-attn failed. Try FoundationStereo FAQ or set XFORMERS_DISABLED=1."
    }
}

$CkptDir = Join-Path $FsDir "pretrained_models\23-51-11"
$Ckpt = Join-Path $CkptDir "model_best_bp2.pth"
if (-not (Test-Path $Ckpt)) {
    Write-Host ""
    Write-Host "=== Download model weights (manual) ==="
    Write-Host "https://github.com/NVlabs/FoundationStereo#model-weights"
    Write-Host "Place folder 23-51-11 at: $CkptDir"
    Write-Host "  (needs model_best_bp2.pth and cfg.yaml)"
} else {
    Write-Host "Checkpoint found: $Ckpt"
}

Write-Host ""
Write-Host "Verify CUDA:"
Write-Host "  conda run -n $EnvName python -c `"import torch; print('cuda', torch.cuda.is_available())`""
Write-Host ""
Write-Host "From stereo_lidar_pointcloud\:"
Write-Host "  python 02_make_stereo_pointcloud.py --run latest"
Write-Host "  python 02_make_stereo_pointcloud_foundation.py --run latest --reuse-rectified"
