# Depth Anything V2

**Full instructions are in [README.md](README.md)** (setup, run commands, comparison, Mac/Windows).

Quick run (after one-time `bash scripts/setup_depth_anything_v2.sh` or Windows `.ps1`):

```bash
python 02_make_stereo_pointcloud.py --run latest
python 02_make_depth_anything_pointcloud.py --run latest --reuse-rectified
```

Batch all `data/<scene>/` folders (each pair → `outputs/runs/<scene>/pair_<id>/`):

```bash
python batch_dav2_data_folders.py
python batch_dav2_data_folders.py --folders checkerboard_data --skip-existing
python 06_evaluate_run.py --run checkerboard_data/pair_000
```
