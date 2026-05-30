"""
Step 3: manually select LiDAR points that hit the checkerboard board.

Run:
    python 03_select_lidar_board_points.py

Controls:
    drag mouse - draw a selection box around board points
    Enter/Space - accept selection for current pair
    r - reset current selection
    q/Esc - quit
"""

from pathlib import Path

import cv2
import numpy as np

from calibration_settings import (
    EXPECTED_SEGMENT_LENGTH_M,
    GOOD_LENGTH_TOLERANCE_M,
    GOOD_LINE_RMS_M,
)

PAIR_DIR = Path("data")
OUT_DIR = Path("selected_lidar_points")
OUT_FILE = OUT_DIR / "selected_lidar_board_points.npz"
VIEW_SIZE = 850


def load_lidar_csv(path):
    data = np.genfromtxt(path, delimiter=",", names=True)
    if data.ndim == 0:
        data = np.array([data])
    return np.column_stack(
        [data["angle_degrees"], data["distance_meters"], data["quality"]]
    )


def polar_to_xy(lidar):
    theta = np.deg2rad(lidar[:, 0])
    x = lidar[:, 1] * np.cos(theta)
    y = lidar[:, 1] * np.sin(theta)
    return x, y


def lidar_to_points_3d(lidar):
    x, y = polar_to_xy(lidar)
    z = np.zeros_like(x)
    return np.column_stack([x, y, z])


def fit_line_quality(selected_xy):
    if len(selected_xy) < 2:
        return 0.0, np.inf
    center = np.mean(selected_xy, axis=0)
    centered = selected_xy - center
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    axis = vh[0]
    along = centered @ axis
    across = centered @ np.array([-axis[1], axis[0]])
    return float(np.max(along) - np.min(along)), float(np.sqrt(np.mean(across**2)))


def fit_segment_endpoints_lidar(selected_lidar):
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
    length, rms = fit_line_quality(selected_xy)
    return np.column_stack([endpoints_xy, np.zeros(2)]), length, rms


def render_lidar(lidar, box=None, selected_mask=None):
    view = np.full((VIEW_SIZE, VIEW_SIZE, 3), 255, dtype=np.uint8)
    x, y = polar_to_xy(lidar)
    max_range = max(1.0, float(np.percentile(lidar[:, 1], 98)))
    scale = VIEW_SIZE * 0.45 / max_range
    center = VIEW_SIZE // 2
    px = np.rint(center + x * scale).astype(int)
    py = np.rint(center - y * scale).astype(int)

    in_view = (0 <= px) & (px < VIEW_SIZE) & (0 <= py) & (py < VIEW_SIZE)
    view[py[in_view], px[in_view]] = (180, 180, 180)

    if selected_mask is not None:
        sel = selected_mask & in_view
        for point_x, point_y in zip(px[sel], py[sel]):
            cv2.circle(view, (point_x, point_y), 3, (0, 0, 255), -1)

    if box is not None:
        x0, y0, x1, y1 = box
        cv2.rectangle(view, (x0, y0), (x1, y1), (255, 0, 255), 2)

    cv2.circle(view, (center, center), 5, (255, 0, 0), -1)
    cv2.putText(view, "Drag box around edge-to-edge checkerboard board returns", (18, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2)
    cv2.putText(view, f"Expected segment length: {EXPECTED_SEGMENT_LENGTH_M:.3f} m", (18, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2)
    cv2.putText(view, "Enter/Space accept | r reset | q quit", (18, 86), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2)
    return view, px, py


def select_points_for_pair(pair_id, lidar):
    state = {"dragging": False, "start": None, "current": None, "box": None}

    def current_box():
        if state["dragging"] and state["start"] and state["current"]:
            return (*state["start"], *state["current"])
        return state["box"]

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

    window = f"Select LiDAR Board Points - pair {pair_id}"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, VIEW_SIZE, VIEW_SIZE)
    cv2.setMouseCallback(window, on_mouse)

    while True:
        box = current_box()
        selected_mask = None
        if box is not None:
            _, px, py = render_lidar(lidar)
            x0, y0, x1, y1 = box
            selected_mask = (
                (px >= min(x0, x1))
                & (px <= max(x0, x1))
                & (py >= min(y0, y1))
                & (py <= max(y0, y1))
            )

        view, _, _ = render_lidar(lidar, box=box, selected_mask=selected_mask)
        if selected_mask is not None and np.count_nonzero(selected_mask) >= 2:
            selected = lidar[selected_mask]
            selected_xy = np.column_stack(polar_to_xy(selected))
            length, rms = fit_line_quality(selected_xy)
            length_error = abs(length - EXPECTED_SEGMENT_LENGTH_M)
            quality = "GOOD" if rms <= GOOD_LINE_RMS_M and length_error <= GOOD_LENGTH_TOLERANCE_M else "check"
            message = (
                f"{quality}: length={length:.3f}m target={EXPECTED_SEGMENT_LENGTH_M:.3f}m "
                f"err={length_error:.3f}m rms={rms:.3f}m"
            )
            cv2.putText(view, message, (18, VIEW_SIZE - 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2)
        cv2.imshow(window, view)

        key = cv2.waitKey(30) & 0xFF
        if key in (ord("q"), 27):
            cv2.destroyWindow(window)
            raise RuntimeError("Selection cancelled.")
        if key == ord("r"):
            state["box"] = None
        if key in (13, 10, ord(" ")):
            if selected_mask is None or np.count_nonzero(selected_mask) < 2:
                print("Select at least 2 LiDAR points.")
                continue
            selected_lidar = lidar[selected_mask]
            endpoints_lidar, length, rms = fit_segment_endpoints_lidar(selected_lidar)
            cv2.destroyWindow(window)
            return selected_lidar, lidar_to_points_3d(selected_lidar), endpoints_lidar, length, rms


def main():
    OUT_DIR.mkdir(exist_ok=True)
    pair_ids = []
    selected_points = []

    for lidar_path in sorted(PAIR_DIR.glob("pair_*_lidar.csv")):
        pair_id = lidar_path.stem.split("_")[1]
        lidar = load_lidar_csv(lidar_path)
        print(f"Selecting board points for pair {pair_id}")

        try:
            selected_lidar, points_3d, endpoints_lidar, length, rms = select_points_for_pair(pair_id, lidar)
        except RuntimeError as exc:
            print(exc)
            break

        pair_ids.append(pair_id)
        selected_points.append(points_3d)

        np.savetxt(
            OUT_DIR / f"pair_{pair_id}_selected_lidar.csv",
            selected_lidar,
            delimiter=",",
            header="angle_degrees,distance_meters,quality",
            comments="",
        )
        np.save(OUT_DIR / f"pair_{pair_id}_selected_points_lidar.npy", points_3d)
        length_error = abs(length - EXPECTED_SEGMENT_LENGTH_M)
        np.savez(
            OUT_DIR / f"pair_{pair_id}_segment_lidar.npz",
            endpoints_lidar=endpoints_lidar,
            estimated_length_m=np.array(length),
            expected_length_m=np.array(EXPECTED_SEGMENT_LENGTH_M),
            length_error_m=np.array(length_error),
            line_rms_m=np.array(rms),
        )
        print(
            f"Saved {len(points_3d)} selected points for pair {pair_id}; "
            f"length={length:.3f}m, target={EXPECTED_SEGMENT_LENGTH_M:.3f}m, "
            f"error={length_error:.3f}m, line RMS={rms:.3f}m"
        )

    if pair_ids:
        np.savez(
            OUT_FILE,
            pair_ids=np.array(pair_ids),
            points_lidar=np.array(selected_points, dtype=object),
        )
        print(f"Saved {OUT_FILE.resolve()}")


if __name__ == "__main__":
    main()
