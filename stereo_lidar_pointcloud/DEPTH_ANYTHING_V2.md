# Depth Anything V2

**Full instructions are in [README.md](README.md)** (setup, run commands, comparison, Mac/Windows).

Quick run (after one-time `bash scripts/setup_depth_anything_v2.sh` or Windows `.ps1`):

```bash
python 02_make_stereo_pointcloud.py --run latest
python 02_make_depth_anything_pointcloud.py --run latest --reuse-rectified
```
