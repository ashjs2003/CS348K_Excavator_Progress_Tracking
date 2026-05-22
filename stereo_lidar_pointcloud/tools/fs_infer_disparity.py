"""
Run FoundationStereo inference and save only disparity (float32 .npy).

Must be executed inside the FoundationStereo conda env on a CUDA machine:

    export FOUNDATION_STEREO_ROOT=/path/to/FoundationStereo
    python fs_infer_disparity.py --left ... --right ... --ckpt ... --out_disp ...

This script is invoked by 02_make_stereo_pointcloud_foundation.py via conda run.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--left", required=True, type=str)
    parser.add_argument("--right", required=True, type=str)
    parser.add_argument("--ckpt", required=True, type=str, help="Path to model_best_bp2.pth")
    parser.add_argument("--out_disp", required=True, type=str)
    parser.add_argument("--scale", type=float, default=1.0, help="Resize factor (<=1)")
    parser.add_argument("--valid_iters", type=int, default=16)
    parser.add_argument("--hiera", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    fs_root = os.environ.get("FOUNDATION_STEREO_ROOT")
    if not fs_root:
        raise RuntimeError("Set FOUNDATION_STEREO_ROOT to the cloned FoundationStereo repo.")
    fs_root = Path(fs_root).resolve()
    if not fs_root.is_dir():
        raise RuntimeError(f"FOUNDATION_STEREO_ROOT is not a directory: {fs_root}")

    sys.path.insert(0, str(fs_root))
    from omegaconf import OmegaConf  # noqa: WPS433

    from core.foundation_stereo import FoundationStereo  # noqa: WPS433
    from core.utils.utils import InputPadder  # noqa: WPS433

    if not torch.cuda.is_available():
        raise RuntimeError(
            "FoundationStereo requires CUDA (NVIDIA GPU). "
            "On macOS, use a Linux/Windows machine with NVIDIA drivers, or run OpenCV stereo only."
        )

    ckpt_dir = Path(args.ckpt)
    cfg = OmegaConf.load(ckpt_dir.parent / "cfg.yaml")
    if "vit_size" not in cfg:
        cfg["vit_size"] = "vitl"
    cfg["scale"] = args.scale
    cfg["hiera"] = args.hiera
    cfg["valid_iters"] = args.valid_iters
    cfg = OmegaConf.create(cfg)

    model = FoundationStereo(cfg)
    ckpt = torch.load(str(ckpt_dir), map_location="cuda")
    model.load_state_dict(ckpt["model"])
    model.cuda()
    model.eval()

    img0 = cv2.cvtColor(cv2.imread(args.left), cv2.COLOR_BGR2RGB)
    img1 = cv2.cvtColor(cv2.imread(args.right), cv2.COLOR_BGR2RGB)
    if img0 is None or img1 is None:
        raise RuntimeError("Could not read left/right rectified images.")

    scale = float(args.scale)
    if scale < 1.0:
        img0 = cv2.resize(img0, fx=scale, fy=scale, dsize=None)
        img1 = cv2.resize(img1, fx=scale, fy=scale, dsize=None)

    height, width = img0.shape[:2]
    tensor0 = torch.as_tensor(img0).cuda().float()[None].permute(0, 3, 1, 2)
    tensor1 = torch.as_tensor(img1).cuda().float()[None].permute(0, 3, 1, 2)
    padder = InputPadder(tensor0.shape, divis_by=32, force_square=False)
    tensor0, tensor1 = padder.pad(tensor0, tensor1)

    with torch.cuda.amp.autocast(True):
        if args.hiera:
            disp = model.run_hierachical(
                tensor0, tensor1, iters=args.valid_iters, test_mode=True, small_ratio=0.5
            )
        else:
            disp = model.forward(tensor0, tensor1, iters=args.valid_iters, test_mode=True)
    disp = padder.unpad(disp.float()).data.cpu().numpy().reshape(height, width).astype(np.float32)

    if scale < 1.0:
        full_h = int(round(height / scale))
        full_w = int(round(width / scale))
        disp = cv2.resize(disp, (full_w, full_h), interpolation=cv2.INTER_LINEAR)

    out_path = Path(args.out_disp)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, disp)
    valid = disp[disp > 0]
    print(
        f"Saved {out_path} shape={disp.shape} "
        f"coverage={100.0 * np.count_nonzero(disp > 0) / disp.size:.1f}% "
        f"range=({valid.min():.2f}, {valid.max():.2f})" if len(valid) else f"Saved {out_path} (empty)"
    )


if __name__ == "__main__":
    main()
