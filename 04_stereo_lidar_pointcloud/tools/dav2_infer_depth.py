"""
Depth Anything V2 inference — run inside conda env depth_anything_v2.

    export DEPTH_ANYTHING_V2_ROOT=/path/to/Depth-Anything-V2
    python dav2_infer_depth.py --image rgb1_rectified.png --ckpt ... --out depth_relative.npy
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
    parser.add_argument("--image", required=True, type=str)
    parser.add_argument("--ckpt", required=True, type=str)
    parser.add_argument("--out", required=True, type=str, help="Output .npy relative depth")
    parser.add_argument("--encoder", default="vits", choices=["vits", "vitb", "vitl"])
    parser.add_argument("--input-size", type=int, default=518)
    return parser.parse_args()


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    args = parse_args()
    root = os.environ.get("DEPTH_ANYTHING_V2_ROOT")
    if not root:
        raise RuntimeError("Set DEPTH_ANYTHING_V2_ROOT to the cloned Depth-Anything-V2 repo.")
    root = Path(root).resolve()
    if not root.is_dir():
        raise RuntimeError(f"DEPTH_ANYTHING_V2_ROOT is not a directory: {root}")

    sys.path.insert(0, str(root))
    from depth_anything_v2.dpt import DepthAnythingV2  # noqa: WPS433

    model_configs = {
        "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
        "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
        "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
    }
    device = pick_device()
    print(f"Device: {device}")

    model = DepthAnythingV2(**model_configs[args.encoder])
    state = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(state)
    model = model.to(device).eval()

    image = cv2.imread(args.image)
    if image is None:
        raise RuntimeError(f"Could not read {args.image}")

    depth = model.infer_image(image, args.input_size)
    depth = np.asarray(depth, dtype=np.float32)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, depth)
    print(
        f"Saved {out_path} shape={depth.shape} "
        f"range=({float(np.min(depth)):.4f}, {float(np.max(depth)):.4f})"
    )


if __name__ == "__main__":
    main()
