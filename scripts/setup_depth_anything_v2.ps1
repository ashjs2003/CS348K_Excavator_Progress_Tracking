# Depth Anything V2 setup for Windows (CUDA) or CPU fallback.
#   powershell -ExecutionPolicy Bypass -File scripts\setup_depth_anything_v2.ps1

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$ThirdParty = Join-Path $RepoRoot "third_party"
$DaDir = Join-Path $ThirdParty "Depth-Anything-V2"
$EnvName = if ($env:DEPTH_ANYTHING_V2_ENV) { $env:DEPTH_ANYTHING_V2_ENV } else { "depth_anything_v2" }
$CudaIndex = if ($env:PYTORCH_CUDA_INDEX) { $env:PYTORCH_CUDA_INDEX } else { "https://download.pytorch.org/whl/cu124" }
$CkptDir = Join-Path $DaDir "checkpoints"
$Ckpt = Join-Path $CkptDir "depth_anything_v2_vits.pth"
$VitsUrl = "https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth"

New-Item -ItemType Directory -Force -Path $ThirdParty | Out-Null

if (-not (Test-Path (Join-Path $DaDir ".git"))) {
    Write-Host "Cloning Depth-Anything-V2 ..."
    git clone --depth 1 https://github.com/DepthAnything/Depth-Anything-V2.git $DaDir
} else {
    Write-Host "Depth-Anything-V2 at $DaDir"
}

if (-not (Get-Command conda -ErrorAction SilentlyContinue)) {
    Write-Host "ERROR: conda not found."
    exit 1
}

$envList = conda env list 2>&1 | Out-String
if ($envList -match "(?m)^$EnvName\s") {
    Write-Host "Conda env '$EnvName' exists."
} else {
    conda create -y -n $EnvName python=3.11 pip
    Write-Host "Installing PyTorch ($CudaIndex) ..."
    conda run -n $EnvName pip install torch torchvision --index-url $CudaIndex
    conda run -n $EnvName pip install opencv-python numpy matplotlib
}

New-Item -ItemType Directory -Force -Path $CkptDir | Out-Null
if (-not (Test-Path $Ckpt)) {
    Write-Host "Downloading vits checkpoint ..."
    Invoke-WebRequest -Uri $VitsUrl -OutFile $Ckpt
} else {
    Write-Host "Checkpoint: $Ckpt"
}

Write-Host ""
Write-Host "Verify:"
Write-Host "  conda run -n $EnvName python -c `"import torch; print('cuda', torch.cuda.is_available())`""
Write-Host ""
Write-Host "  python 02_make_stereo_pointcloud.py --run latest"
Write-Host "  python 02_make_depth_anything_pointcloud.py --run latest --reuse-rectified"
