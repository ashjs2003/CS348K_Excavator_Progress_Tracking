"""
Step 1: capture synchronized-ish RGB image + 2D LiDAR scan pairs.

Run:
    python 01_capture_rgb_lidar_pairs.py

Controls:
    s - save current RGB frame and latest LiDAR scan
    p - capture a fresh RGB frame and full LiDAR scan, then select board LiDAR points
    q - quit
"""

from pathlib import Path
import sys
import time

import cv2
import numpy as np
from rplidar import RPLidar, RPLidarException

from calibration_settings import (
    CHECKERBOARD_INNER_CORNERS,
    EXPECTED_SEGMENT_LENGTH_M,
    GOOD_LENGTH_TOLERANCE_M,
    GOOD_LINE_RMS_M,
    MAX_CAPTURE_DISTANCE_M,
    MIN_CAPTURE_DISTANCE_M,
)

_STEREO_CALIB_DIR = Path(__file__).resolve().parents[1] / "stereo_calibration"
if str(_STEREO_CALIB_DIR) not in sys.path:
    sys.path.insert(0, str(_STEREO_CALIB_DIR))
from hardware_settings import BAUDRATE, LIDAR_PORT, RGB1_CAMERA_INDEX

CAMERA_INDEX = RGB1_CAMERA_INDEX
CHECKERBOARD = CHECKERBOARD_INNER_CORNERS
MIN_DISTANCE_M = MIN_CAPTURE_DISTANCE_M
MAX_DISTANCE_M = MAX_CAPTURE_DISTANCE_M
RGB_CALIBRATION_FILE = Path("../rgb_calibration/camera_calibration_normal.npz")
OUT_DIR = Path("pairs")
SELECTED_DIR = Path("selected_lidar_points")
LIDAR_VIEW_SIZE = 800
MIN_SAVE_POINTS = 10
FRESH_CAPTURE_SCANS = 3


def load_calibrated_image_size():
    data = np.load(RGB_CALIBRATION_FILE)
    return tuple(data["image_size"].astype(int))


def next_pair_index():
    existing = sorted(OUT_DIR.glob("pair_*_image.png"))
    indices = []
    for path in existing:
        try:
            indices.append(int(path.stem.split("_")[1]))
        except (IndexError, ValueError):
            pass
    return max(indices, default=-1) + 1


def open_camera(image_size):
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open RGB camera index {CAMERA_INDEX}.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(image_size[0]))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(image_size[1]))
    return cap


def clean_lidar_startup(lidar):
    """Reset common RPLidar stream states before reading scans."""
    clean_input = getattr(lidar, "clean_input", None)
    if clean_input is None:
        clean_input = getattr(lidar, "clear_input", None)

    actions = [lidar.stop, lidar.stop_motor]
    if clean_input is not None:
        actions.append(clean_input)

    for action in actions:
        try:
            action()
            time.sleep(0.3)
        except Exception:
            pass
    lidar.start_motor()
    time.sleep(2.0)


def scan_to_array(scan):
    """Convert an RPLidar scan to angle_degrees, distance_meters, quality."""
    rows = []
    for quality, angle_deg, distance_mm in scan:
        distance_m = distance_mm / 1000.0
        if quality > 0 and MIN_DISTANCE_M <= distance_m <= MAX_DISTANCE_M:
            rows.append([angle_deg, distance_m, quality])
    return np.array(rows, dtype=float)


def capture_fresh_pair(cap, scan_iterator):
    """Wait for new LiDAR scans, keep the fullest one, then grab a fresh RGB frame."""
    print("Capturing a fresh full LiDAR scan...")
    best_lidar_points = None

    for _ in range(FRESH_CAPTURE_SCANS):
        scan = next(scan_iterator)
        lidar_points = scan_to_array(scan)
        if best_lidar_points is None or len(lidar_points) > len(best_lidar_points):
            best_lidar_points = lidar_points

    ret, frame = cap.read()
    if not ret:
        print("Not saved: failed to read a fresh RGB frame.")
        return None, None, False

    if best_lidar_points is None or len(best_lidar_points) < MIN_SAVE_POINTS:
        print("Not saved: fresh LiDAR scan has too few points.")
        return frame, best_lidar_points, False

    checkerboard_found, _ = detect_checkerboard(frame)
    if not checkerboard_found:
        print("Not saved: checkerboard was not detected in fresh RGB image.")
        return frame, best_lidar_points, False

    print(f"Fresh capture ready: {len(best_lidar_points)} LiDAR points.")
    return frame, best_lidar_points, True


def detect_checkerboard(frame):
    """Detect checkerboard in the live RGB frame so bad pairs are obvious."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if hasattr(cv2, "findChessboardCornersSB"):
        flags = cv2.CALIB_CB_EXHAUSTIVE + cv2.CALIB_CB_ACCURACY
        found, corners = cv2.findChessboardCornersSB(gray, CHECKERBOARD, flags)
        if found:
            return True, corners.astype(np.float32)

    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    found, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, flags)
    return found, corners


def polar_to_xy(lidar_points):
    theta = np.deg2rad(lidar_points[:, 0])
    x = lidar_points[:, 1] * np.cos(theta)
    y = lidar_points[:, 1] * np.sin(theta)
    return x, y


def lidar_to_xyz(lidar_points):
    x, y = polar_to_xy(lidar_points)
    return np.column_stack([x, y, np.zeros_like(x)])


def lidar_view_coordinates(lidar_points):
    x, y = polar_to_xy(lidar_points)
    max_range = max(1.0, float(np.percentile(lidar_points[:, 1], 98)))
    scale = LIDAR_VIEW_SIZE * 0.45 / max_range
    center = LIDAR_VIEW_SIZE // 2
    px = np.rint(center + x * scale).astype(int)
    py = np.rint(center - y * scale).astype(int)
    return px, py, center, scale


def fit_line_quality(selected_xy):
    """Return line length and RMS line residual for selected 2D LiDAR points."""
    if len(selected_xy) < 2:
        return 0.0, np.inf

    centered = selected_xy - np.mean(selected_xy, axis=0)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    axis = vh[0]
    along = centered @ axis
    across = centered @ np.array([-axis[1], axis[0]])
    length = float(np.max(along) - np.min(along))
    rms = float(np.sqrt(np.mean(across**2)))
    return length, rms


def fit_segment_endpoints_lidar(selected_lidar):
    """Estimate the two LiDAR-frame endpoints of the selected straight segment."""
    selected_xy = np.column_stack(polar_to_xy(selected_lidar))
    center = np.mean(selected_xy, axis=0)
    centered = selected_xy - center
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    axis = vh[0]
    along = centered @ axis
    endpoints_xy = np.vstack([
        center + axis * np.min(along),
        center + axis * np.max(along),
    ])
    endpoints_lidar = np.column_stack([endpoints_xy, np.zeros(2)])
    length, rms = fit_line_quality(selected_xy)
    return endpoints_lidar, length, rms


def render_lidar_preview(lidar_points, selected_mask=None):
    """Draw a top-down LiDAR preview. The board should appear as a straight segment."""
    view = np.full((LIDAR_VIEW_SIZE, LIDAR_VIEW_SIZE, 3), 255, dtype=np.uint8)
    if lidar_points is None or len(lidar_points) == 0:
        return view

    px, py, center, scale = lidar_view_coordinates(lidar_points)
    in_view = (0 <= px) & (px < LIDAR_VIEW_SIZE) & (0 <= py) & (py < LIDAR_VIEW_SIZE)

    for point_x, point_y in zip(px[in_view], py[in_view]):
        cv2.circle(view, (point_x, point_y), 2, (170, 170, 170), -1)

    if selected_mask is not None:
        selected = selected_mask & in_view
        for point_x, point_y in zip(px[selected], py[selected]):
            cv2.circle(view, (point_x, point_y), 4, (0, 0, 255), -1)

    cv2.circle(view, (center, center), 5, (255, 0, 0), -1)
    cv2.circle(view, (center, center), int(0.5 * scale), (225, 225, 225), 1)
    cv2.circle(view, (center, center), int(1.0 * scale), (210, 210, 210), 1)
    cv2.putText(view, "LiDAR top-down: board should be one clean edge-to-edge segment", (16, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 2)
    cv2.putText(view, f"expected length={EXPECTED_SEGMENT_LENGTH_M:.3f}m, range <= {MAX_DISTANCE_M:.1f}m", (16, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 2)
    cv2.putText(view, "s save pair | p fresh-save + select board points | q quit", (16, 86), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 2)
    return view


def select_board_points(pair_index, lidar_points):
    """Let the user drag a box around board returns immediately after capture."""
    SELECTED_DIR.mkdir(exist_ok=True)
    state = {"dragging": False, "start": None, "current": None, "box": None}

    def current_box():
        if state["dragging"] and state["start"] and state["current"]:
            return (*state["start"], *state["current"])
        return state["box"]

    def selected_mask_from_box(box):
        if box is None:
            return None
        px, py, _, _ = lidar_view_coordinates(lidar_points)
        x0, y0, x1, y1 = box
        return (
            (px >= min(x0, x1))
            & (px <= max(x0, x1))
            & (py >= min(y0, y1))
            & (py <= max(y0, y1))
        )

    def on_mouse(event, x, y, flags, userdata):
        if event == cv2.EVENT_LBUTTONDOWN:
            state["dragging"] = True
            state["start"] = (x, y)
            state["current"] = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and state["dragging"]:
            state["current"] = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            state["dragging"] = False
            state["current"] = (x, y)
            if state["start"] is not None:
                state["box"] = (*state["start"], *state["current"])

    window = f"Select board LiDAR points - pair_{pair_index:03d}"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, LIDAR_VIEW_SIZE, LIDAR_VIEW_SIZE)
    cv2.setMouseCallback(window, on_mouse)

    while True:
        box = current_box()
        selected_mask = selected_mask_from_box(box)
        view = render_lidar_preview(lidar_points, selected_mask)
        if box is not None:
            x0, y0, x1, y1 = box
            cv2.rectangle(view, (x0, y0), (x1, y1), (255, 0, 255), 2)

        message = "Drag board line, Enter accept, r reset, q skip"
        if selected_mask is not None and np.count_nonzero(selected_mask) >= 2:
            selected = lidar_points[selected_mask]
            selected_xy = np.column_stack(polar_to_xy(selected))
            length, rms = fit_line_quality(selected_xy)
            length_error = abs(length - EXPECTED_SEGMENT_LENGTH_M)
            quality = "GOOD" if rms <= GOOD_LINE_RMS_M and length_error <= GOOD_LENGTH_TOLERANCE_M else "check"
            message = (
                f"{quality}: selected={len(selected)} length={length:.3f}m "
                f"target={EXPECTED_SEGMENT_LENGTH_M:.3f}m err={length_error:.3f}m rms={rms:.3f}m"
            )
        cv2.putText(view, message, (16, LIDAR_VIEW_SIZE - 22), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 2)
        cv2.imshow(window, view)

        key = cv2.waitKey(30) & 0xFF
        if key in (ord("q"), 27):
            cv2.destroyWindow(window)
            print("Skipped LiDAR board-point selection for this pair.")
            return
        if key == ord("r"):
            state["box"] = None
        if key in (13, 10, ord(" ")):
            if selected_mask is None or np.count_nonzero(selected_mask) < 2:
                print("Select at least 2 points first.")
                continue

            selected_lidar = lidar_points[selected_mask]
            selected_xyz = lidar_to_xyz(selected_lidar)
            endpoints_lidar, length, rms = fit_segment_endpoints_lidar(selected_lidar)
            length_error = abs(length - EXPECTED_SEGMENT_LENGTH_M)
            np.savetxt(
                SELECTED_DIR / f"pair_{pair_index:03d}_selected_lidar.csv",
                selected_lidar,
                delimiter=",",
                header="angle_degrees,distance_meters,quality",
                comments="",
            )
            np.save(SELECTED_DIR / f"pair_{pair_index:03d}_selected_points_lidar.npy", selected_xyz)
            np.savez(
                SELECTED_DIR / f"pair_{pair_index:03d}_segment_lidar.npz",
                endpoints_lidar=endpoints_lidar,
                estimated_length_m=np.array(length),
                expected_length_m=np.array(EXPECTED_SEGMENT_LENGTH_M),
                length_error_m=np.array(length_error),
                line_rms_m=np.array(rms),
            )
            cv2.destroyWindow(window)
            print(
                f"Saved {len(selected_xyz)} selected board points for pair {pair_index:03d}; "
                f"length={length:.3f}m, target={EXPECTED_SEGMENT_LENGTH_M:.3f}m, "
                f"error={length_error:.3f}m, line RMS={rms:.3f}m"
            )
            return


def save_pair(pair_index, frame, lidar_points):
    image_path = OUT_DIR / f"pair_{pair_index:03d}_image.png"
    lidar_path = OUT_DIR / f"pair_{pair_index:03d}_lidar.csv"

    cv2.imwrite(str(image_path), frame)
    np.savetxt(
        lidar_path,
        lidar_points,
        delimiter=",",
        header="angle_degrees,distance_meters,quality",
        comments="",
    )
    print(f"Saved pair {pair_index:03d}: {image_path}, {lidar_path}")


def main():
    OUT_DIR.mkdir(exist_ok=True)

    try:
        image_size = load_calibrated_image_size()
        cap = open_camera(image_size)
    except Exception as exc:
        print(f"Error: {exc}")
        return

    actual_size = (
        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )
    print(f"RGB camera resolution: {actual_size[0]}x{actual_size[1]}")
    print(f"LiDAR: {LIDAR_PORT} at {BAUDRATE}")
    print("Good pair checklist:")
    print("- RGB preview says checkerboard FOUND and corners are drawn correctly.")
    print(f"- LiDAR preview shows a clean straight segment near {EXPECTED_SEGMENT_LENGTH_M:.3f} m.")
    print(f"- Board is within {MAX_DISTANCE_M:.1f} m so farther background points are filtered out.")
    print("- Board is at varied distance/angle from previous captures.")
    print("Press 's' to save, 'p' to fresh-capture + select LiDAR board points, 'q' to quit.")

    lidar = RPLidar(LIDAR_PORT, baudrate=BAUDRATE, timeout=3)
    pair_index = next_pair_index()
    last_lidar_points = None

    try:
        clean_lidar_startup(lidar)
        should_quit = False

        while not should_quit:
            try:
                scan_iterator = lidar.iter_scans(max_buf_meas=5000)
                for scan in scan_iterator:
                    last_lidar_points = scan_to_array(scan)
                    ret, frame = cap.read()
                    if not ret:
                        print("Warning: failed to read RGB frame.")
                        continue

                    checkerboard_found, corners = detect_checkerboard(frame)
                    preview = frame.copy()
                    if checkerboard_found:
                        cv2.drawChessboardCorners(preview, CHECKERBOARD, corners, checkerboard_found)

                    status = "FOUND" if checkerboard_found else "not found"
                    status_color = (0, 255, 0) if checkerboard_found else (0, 0, 255)
                    cv2.putText(
                        preview,
                        f"checkerboard: {status} | next pair_{pair_index:03d} | lidar points: {len(last_lidar_points)}",
                        (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.75,
                        status_color,
                        2,
                        cv2.LINE_AA,
                    )
                    cv2.putText(
                        preview,
                        "s save | p fresh-save + select LiDAR board points | q quit",
                        (20, 68),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.75,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )
                    cv2.imshow("RGB Camera", preview)
                    cv2.imshow("LiDAR Top-Down", render_lidar_preview(last_lidar_points))

                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("s"):
                        if last_lidar_points is None or len(last_lidar_points) < MIN_SAVE_POINTS:
                            print("Not saved: LiDAR scan has too few points.")
                            continue
                        if not checkerboard_found:
                            print("Not saved: checkerboard was not detected in RGB image.")
                            continue
                        save_pair(pair_index, frame, last_lidar_points)
                        pair_index += 1
                    elif key == ord("p"):
                        capture_frame, capture_lidar_points, ok = capture_fresh_pair(cap, scan_iterator)
                        if not ok:
                            continue
                        save_pair(pair_index, capture_frame, capture_lidar_points)
                        select_board_points(pair_index, capture_lidar_points)
                        pair_index += 1
                    elif key == ord("q"):
                        should_quit = True
                        break
            except RPLidarException as exc:
                print(f"LiDAR stream error: {exc}")
                print("Resetting LiDAR stream...")
                clean_lidar_startup(lidar)

    finally:
        cap.release()
        cv2.destroyAllWindows()
        try:
            lidar.stop()
            lidar.stop_motor()
        except Exception:
            pass
        lidar.disconnect()


if __name__ == "__main__":
    main()
