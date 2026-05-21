"""
Copy RGB1 intrinsics as an approximate starting calibration for RGB2.

Run:
    python 01_copy_rgb1_intrinsics_to_rgb2.py
"""

from pathlib import Path

import numpy as np


RGB1_CALIBRATION_FILE = Path("../config/camera_calibration_rgb1.npz")
RGB2_APPROX_CALIBRATION_FILE = Path("camera_calibration_rgb2_approx.npz")


def main():
    if not RGB1_CALIBRATION_FILE.exists():
        raise FileNotFoundError(f"Missing {RGB1_CALIBRATION_FILE}")

    rgb1 = np.load(RGB1_CALIBRATION_FILE)
    required_keys = [
        "camera_matrix",
        "dist_coeffs",
        "image_size",
        "checkerboard_size",
        "square_size",
    ]
    missing = [key for key in required_keys if key not in rgb1.files]
    if missing:
        raise KeyError(f"{RGB1_CALIBRATION_FILE} is missing keys: {missing}")

    np.savez(
        RGB2_APPROX_CALIBRATION_FILE,
        camera_matrix=rgb1["camera_matrix"],
        dist_coeffs=rgb1["dist_coeffs"],
        image_size=rgb1["image_size"],
        checkerboard_size=rgb1["checkerboard_size"],
        square_size=rgb1["square_size"],
    )

    print(f"Saved approximate RGB2 calibration: {RGB2_APPROX_CALIBRATION_FILE.resolve()}")
    print()
    print("WARNING: RGB2 intrinsics were copied from RGB1.")
    print("This is only an approximation for a quick prototype.")
    print("Even same-model cameras can have different intrinsics and distortion,")
    print("so replace this with a real RGB2 calibration when accuracy matters.")


if __name__ == "__main__":
    main()
