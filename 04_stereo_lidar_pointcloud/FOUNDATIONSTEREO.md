# FoundationStereo

**Full instructions are in [README.md](README.md)** (setup, run commands, comparison, Mac/Windows).

Quick run on **Windows + NVIDIA GPU** (after `scripts\setup_foundationstereo.ps1`):

```powershell
python 02_make_stereo_pointcloud.py --run latest
python 02_make_stereo_pointcloud_foundation.py --run latest --reuse-rectified
```

Not supported on macOS (CUDA required). Use Depth Anything V2 on Mac instead.
