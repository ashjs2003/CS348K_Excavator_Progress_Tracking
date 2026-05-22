"""
Capture one synchronized-ish RGB1 + RGB2 + 2D LiDAR set.

Run:
    python 01_capture_one_set.py
    python 01_capture_one_set.py --label carpet

Each save creates a new folder under outputs/runs/<timestamp>_<label>/capture/.

Controls:
    s - save one set
    q - quit
"""

import argparse
from pathlib import Path
import json
import sys
import time

import cv2
import numpy as np
from rplidar import RPLidar, RPLidarException

from calib_utils import load_camera_calibration, open_camera

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_STEREO_CALIB = _REPO_ROOT / "stereo_calibration"
if str(_STEREO_CALIB) not in sys.path:
    sys.path.insert(0, str(_STEREO_CALIB))
from hardware_settings import BAUDRATE, LIDAR_PORT, RGB1_CAMERA_INDEX, RGB2_CAMERA_INDEX
from output_runs import create_run_dir, write_run_info

WINDOW = "Capture RGB1 + RGB2"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--label",
        default="capture",
        help="Tag for this run folder, e.g. carpet, cardboard (default: capture)",
    )
    return parser.parse_args()


def clean_lidar_startup(lidar):
    """Reset common RPLidar stream states before reading scans."""
    clean_input = getattr(lidar, "clean_input", None) or getattr(lidar, "clear_input", None)
    for action in [lidar.stop, lidar.stop_motor, clean_input]:
        if action is None:
            continue
        try:
            action()
            time.sleep(0.3)
        except Exception:
            pass
    lidar.start_motor()
    time.sleep(2.0)


def scan_to_rows(scan):
    """Convert an RPLidar scan to angle_degrees, distance_meters, quality rows."""
    rows = []
    for quality, angle_deg, distance_mm in scan:
        distance_m = distance_mm / 1000.0
        if quality > 0 and distance_m > 0:
            rows.append([angle_deg, distance_m, quality])
    return np.array(rows, dtype=float)


def side_by_side(left, right):
    if left.shape[0] != right.shape[0]:
        scale = left.shape[0] / right.shape[0]
        right = cv2.resize(right, (int(right.shape[1] * scale), left.shape[0]))
    return np.hstack([left, right])


def save_capture(capture_dir, frame1, frame2, lidar_rows, metadata):
    capture_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(capture_dir / "rgb1.png"), frame1)
    cv2.imwrite(str(capture_dir / "rgb2.png"), frame2)
    np.savetxt(
        capture_dir / "lidar_scan.csv",
        lidar_rows,
        delimiter=",",
        header="angle_degrees,distance_meters,quality",
        comments="",
    )
    with open(capture_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved capture to {capture_dir.resolve()}")


def main():
    args = parse_args()
    run_dir = None
    capture_dir = None

    rgb1_calib = load_camera_calibration("camera_calibration_rgb1.npz")
    rgb2_calib = load_camera_calibration("camera_calibration_rgb2.npz", "camera_calibration_rgb2_approx.npz")
    image_size = rgb1_calib["image_size"]

    cap1 = open_camera(RGB1_CAMERA_INDEX, image_size)
    cap2 = open_camera(RGB2_CAMERA_INDEX, rgb2_calib["image_size"])
    lidar = RPLidar(LIDAR_PORT, baudrate=BAUDRATE, timeout=3)

    actual1 = (int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    actual2 = (int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    print(f"RGB1 actual resolution: {actual1[0]}x{actual1[1]}")
    print(f"RGB2 actual resolution: {actual2[0]}x{actual2[1]}")
    print(f"Label for next save: {args.label} (new folder under outputs/runs/)")
    print("Press s to save one set, q to quit.")
    last_lidar_rows = np.empty((0, 3), dtype=float)

    try:
        clean_lidar_startup(lidar)
        should_quit = False
        while not should_quit:
            try:
                for scan in lidar.iter_scans(max_buf_meas=5000):
                    last_lidar_rows = scan_to_rows(scan)
                    ret1, frame1 = cap1.read()
                    ret2, frame2 = cap2.read()
                    if not ret1 or not ret2:
                        print("Warning: failed to read one or both cameras.")
                        continue

                    preview1 = frame1.copy()
                    preview2 = frame2.copy()
                    cv2.putText(preview1, "RGB1", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                    cv2.putText(preview2, "RGB2", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                    preview = side_by_side(preview1, preview2)
                    cv2.putText(preview, f"s save | q quit | lidar points={len(last_lidar_rows)}", (20, preview.shape[0] - 24), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 4, cv2.LINE_AA)
                    cv2.putText(preview, f"s save | q quit | lidar points={len(last_lidar_rows)}", (20, preview.shape[0] - 24), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
                    cv2.imshow(WINDOW, preview)

                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        should_quit = True
                        break
                    if key == ord("s"):
                        if run_dir is None:
                            run_dir = create_run_dir(args.label)
                            capture_dir = run_dir / "capture"
                            write_run_info(run_dir, label=args.label, step="capture")
                            print(f"Run folder: {run_dir.resolve()}")
                        metadata = {
                            "timestamp_unix_sec": time.time(),
                            "rgb1_camera_index": RGB1_CAMERA_INDEX,
                            "rgb2_camera_index": RGB2_CAMERA_INDEX,
                            "lidar_port": LIDAR_PORT,
                            "baudrate": BAUDRATE,
                            "rgb1_shape": list(frame1.shape),
                            "rgb2_shape": list(frame2.shape),
                            "lidar_columns": ["angle_degrees", "distance_meters", "quality"],
                            "lidar_points": int(len(last_lidar_rows)),
                        }
                        save_capture(capture_dir, frame1, frame2, last_lidar_rows, metadata)
                        write_run_info(
                            run_dir,
                            last_capture_unix_sec=metadata["timestamp_unix_sec"],
                            lidar_points=metadata["lidar_points"],
                        )
            except RPLidarException as exc:
                print(f"LiDAR stream error: {exc}")
                print("Resetting LiDAR stream...")
                clean_lidar_startup(lidar)
    finally:
        cap1.release()
        cap2.release()
        cv2.destroyAllWindows()
        try:
            lidar.stop()
            lidar.stop_motor()
        except Exception:
            pass
        lidar.disconnect()


if __name__ == "__main__":
    main()
