"""Save distorted vs undistorted comparison images for RGB calibration.

Run from this folder:
    python 05_save_distorted_vs_undistorted.py
    python 05_save_distorted_vs_undistorted.py --camera L --image-id 005
"""

import argparse
import os
from pathlib import Path

os.environ["OPENCV_OPENCL_RUNTIME"] = "disabled"

import cv2
import numpy as np

from calib_targets import resolve_camera


cv2.ocl.setUseOpenCL(False)
SCRIPT_DIR = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(description="Save distorted vs undistorted RGB calibration previews.")
    parser.add_argument(
        "--camera",
        choices=["L", "R", "l", "r", "left", "right", "rgb1", "rgb2", "both"],
        default="both",
        help="Which RGB calibration set to render. Default: both.",
    )
    parser.add_argument("--image-id", default=None, help="Calibration image id, for example 005.")
    parser.add_argument("--image", type=Path, default=None, help="Use one explicit image path instead of image-id.")
    parser.add_argument("--alpha", type=float, default=0.0, help="OpenCV undistortion alpha: 0 crops, 1 keeps FOV.")
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR / "outputs")
    return parser.parse_args()


def default_calibration_file(target):
    return target["outlier_npz"] if target["outlier_npz"].exists() else target["normal_npz"]


def load_image_paths(image_dir):
    paths = []
    for pattern in ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff"):
        paths.extend(sorted(image_dir.glob(pattern)))
    return sorted(set(paths))


def choose_image(target, image_id, explicit_image):
    if explicit_image is not None:
        return explicit_image
    if image_id is not None:
        return target["image_dir"] / f"calib_{image_id}.png"
    paths = load_image_paths(target["image_dir"])
    if not paths:
        raise RuntimeError(f"No calibration images found in {target['image_dir']}")
    return paths[0]


def label_image(image, text):
    labeled = image.copy()
    cv2.rectangle(labeled, (0, 0), (labeled.shape[1], 44), (0, 0, 0), -1)
    cv2.putText(labeled, text, (16, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
    return labeled


def crop_to_roi(image, roi):
    x, y, w, h = roi
    if w <= 0 or h <= 0:
        return image
    return image[y : y + h, x : x + w]


def save_comparison(camera_name, args):
    target = resolve_camera(camera_name)
    calibration_file = default_calibration_file(target)
    if not calibration_file.exists():
        raise FileNotFoundError(f"Missing calibration file: {calibration_file}")

    image_path = choose_image(target, args.image_id, args.image)
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    data = np.load(calibration_file)
    camera_matrix = data["camera_matrix"]
    dist_coeffs = data["dist_coeffs"]
    frame_size = (image.shape[1], image.shape[0])

    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix,
        dist_coeffs,
        frame_size,
        args.alpha,
        frame_size,
    )
    undistorted = cv2.undistort(image, camera_matrix, dist_coeffs, None, new_camera_matrix)
    if args.alpha == 0:
        undistorted = crop_to_roi(undistorted, roi)
        undistorted = cv2.resize(undistorted, frame_size, interpolation=cv2.INTER_AREA)

    distorted_label = f"{target['label']} distorted: {image_path.name}"
    undistorted_label = f"{target['label']} undistorted: alpha={args.alpha:g}"
    combined = np.hstack([label_image(image, distorted_label), label_image(undistorted, undistorted_label)])

    output_path = args.output_dir / target["label"] / "distorted_vs_undistorted.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), combined):
        raise RuntimeError(f"Could not write {output_path}")
    print(f"Saved {output_path.resolve()}")


def main():
    args = parse_args()
    cameras = ("L", "R") if args.camera == "both" else (args.camera,)
    for camera in cameras:
        save_comparison(camera, args)


if __name__ == "__main__":
    main()
