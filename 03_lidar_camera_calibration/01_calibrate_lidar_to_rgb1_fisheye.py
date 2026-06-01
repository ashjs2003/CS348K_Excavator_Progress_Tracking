"""Calibrate 2D LiDAR to RGB1 using checkerboard planes and fisheye cameras.

Inputs:
- data/checkerboard_data*, or 00_data_capture/checkerboard_*
- L/R fisheye no-outlier intrinsics
- fixed fisheye RGB1->RGB2 stereo extrinsics

Outputs:
- config/lidar_rgb1_extrinsics.npz
- outputs/lidar_rgb1_calibration_eval.json
- outputs/lidar_rgb1_calibration_eval_per_capture.csv
- outputs/lidar_rgb1_leave_one_distance_out.csv
- outputs/rgb1_overlays/*.png
- outputs/rgb2_overlays/*.png
"""

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.project_config import calibration_file


CHECKERBOARD = (9, 6)
BOARD_SQUARES = (10, 7)
BOARD_BOUNDS_WEIGHT = 0.8
IMAGE_LINE_WEIGHT = 0.002
BOARD_STRIP_WEIGHT = 1.0


def parse_args():
    parser = argparse.ArgumentParser(description="Calibrate LiDAR to RGB1 from checkerboard plane observations.")
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--left-calibration", type=Path, default=calibration_file("left_fisheye_intrinsics_no_outliers"))
    parser.add_argument("--right-calibration", type=Path, default=calibration_file("right_fisheye_intrinsics_no_outliers"))
    parser.add_argument("--stereo-calibration", type=Path, default=calibration_file("stereo_rgb1_rgb2_extrinsics"))
    parser.add_argument("--output-file", type=Path, default=SCRIPT_DIR / "config" / "lidar_rgb1_extrinsics.npz")
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR / "outputs")
    parser.add_argument("--distance-window", type=float, default=0.06)
    parser.add_argument(
        "--segment-axis",
        choices=["width", "height"],
        default="height",
        help="Expected checkerboard dimension crossed by the LiDAR scan. This dataset uses height, the shorter side.",
    )
    parser.add_argument("--segment-length-tolerance", type=float, default=0.08)
    parser.add_argument("--annotations", type=Path, default=SCRIPT_DIR / "outputs" / "manual_board_corners.json")
    parser.add_argument("--annotate-missing", action="store_true")
    parser.add_argument(
        "--annotation-mode",
        choices=["line", "outer", "partial-grid"],
        default="line",
        help="Manual fallback mode. line uses two clicked endpoints of the visible LiDAR-hit line.",
    )
    return parser.parse_args()


def load_intrinsics(path):
    data = np.load(path)
    return {
        "K": data["camera_matrix"].astype(np.float64),
        "D": data["dist_coeffs"].reshape(-1, 1).astype(np.float64),
        "image_size": tuple(int(v) for v in data["image_size"].astype(int)),
        "square_size": float(np.asarray(data["square_size"]).reshape(())),
    }


def load_stereo(path):
    data = np.load(path)
    return {
        "R": data["R_rgb1_to_rgb2"].astype(np.float64),
        "t": data["t_rgb1_to_rgb2"].reshape(3, 1).astype(np.float64),
    }


def checkerboard_object_points(square_size):
    cols, rows = CHECKERBOARD
    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp *= square_size
    return objp


def board_outer_object_points(square_size):
    width = BOARD_SQUARES[0] * square_size
    height = BOARD_SQUARES[1] * square_size
    return np.array(
        [
            [0.0, 0.0, 0.0],
            [width, 0.0, 0.0],
            [width, height, 0.0],
            [0.0, height, 0.0],
        ],
        dtype=np.float32,
    )


def detect_corners(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    variants = (gray, cv2.equalizeHist(gray), cv2.GaussianBlur(cv2.equalizeHist(gray), (3, 3), 0))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.001)
    if hasattr(cv2, "findChessboardCornersSB"):
        for candidate in variants:
            found, corners = cv2.findChessboardCornersSB(
                candidate, CHECKERBOARD, cv2.CALIB_CB_EXHAUSTIVE + cv2.CALIB_CB_ACCURACY
            )
            if found:
                corners = cv2.cornerSubPix(candidate, corners.astype(np.float32), (11, 11), (-1, -1), criteria)
                return True, corners
    for candidate in variants:
        found, corners = cv2.findChessboardCorners(
            candidate, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        )
        if found:
            corners = cv2.cornerSubPix(candidate, corners.astype(np.float32), (11, 11), (-1, -1), criteria)
            return True, corners
    return False, None


def pose_to_plane(rvec, tvec):
    R, _ = cv2.Rodrigues(rvec)
    normal = R @ np.array([0.0, 0.0, 1.0])
    normal = normal / np.linalg.norm(normal)
    d = -float(normal @ tvec.reshape(3))
    return normal, d


def solve_board_plane_from_points(image_points, object_points, K, D):
    if len(object_points) < 4 or np.linalg.matrix_rank(object_points[:, :2] - object_points[:1, :2]) < 2:
        return None
    undistorted = cv2.fisheye.undistortPoints(
        image_points.reshape(-1, 1, 2).astype(np.float64),
        K,
        D,
        P=K,
    ).astype(np.float32)
    ok, rvec, tvec = cv2.solvePnP(object_points, undistorted, K, None)
    if not ok:
        return None
    normal, d = pose_to_plane(rvec, tvec)
    return {"corners": image_points.reshape(-1, 1, 2).astype(np.float32), "rvec": rvec, "tvec": tvec, "normal": normal, "d": d}


def checkerboard_axis_line(corners, axis):
    pts = corners.reshape(CHECKERBOARD[1], CHECKERBOARD[0], 2)
    if axis == "width":
        p0 = np.mean(pts[:, 0, :], axis=0)
        p1 = np.mean(pts[:, -1, :], axis=0)
    else:
        p0 = np.mean(pts[0, :, :], axis=0)
        p1 = np.mean(pts[-1, :, :], axis=0)
    return np.asarray([p0, p1], dtype=np.float64)


def solve_board_plane_rgb1(image, K, D, objp):
    found, corners = detect_corners(image)
    if not found:
        return None
    pose = solve_board_plane_from_points(corners.reshape(-1, 2), objp, K, D)
    if pose is not None:
        pose["full_corners"] = corners
    return pose


def load_annotations(path):
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_annotations(path, annotations):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(annotations, f, indent=2)


def click_four_corners(image, title):
    points = []
    preview = image.copy()

    def on_mouse(event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN or len(points) >= 4:
            return
        points.append([float(x), float(y)])
        cv2.circle(preview, (x, y), 5, (0, 255, 255), -1, cv2.LINE_AA)
        cv2.putText(preview, str(len(points)), (x + 8, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(title, on_mouse)
    while True:
        shown = preview.copy()
        cv2.putText(
            shown,
            "Click 4 OUTER board corners: top-left, top-right, bottom-right, bottom-left. Enter=save, q=skip",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.imshow(title, shown)
        key = cv2.waitKey(20) & 0xFF
        if key in (13, 10) and len(points) == 4:
            cv2.destroyWindow(title)
            return points
        if key == ord("q"):
            cv2.destroyWindow(title)
            return None


def click_partial_grid_points(image, title, square_size):
    points = []
    preview = image.copy()

    def on_mouse(event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        points.append([float(x), float(y)])
        cv2.circle(preview, (x, y), 5, (0, 255, 255), -1, cv2.LINE_AA)
        cv2.putText(preview, str(len(points)), (x + 8, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(title, on_mouse)
    while True:
        shown = preview.copy()
        cv2.putText(
            shown,
            "Click visible INTERNAL checkerboard intersections. Need 4+ non-collinear. Enter=done, u=undo, q=skip",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.imshow(title, shown)
        key = cv2.waitKey(20) & 0xFF
        if key == ord("u") and points:
            points.pop()
            preview = image.copy()
            for i, (px, py) in enumerate(points, start=1):
                cv2.circle(preview, (int(px), int(py)), 5, (0, 255, 255), -1, cv2.LINE_AA)
                cv2.putText(preview, str(i), (int(px) + 8, int(py) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        elif key in (13, 10) and len(points) >= 4:
            cv2.destroyWindow(title)
            break
        elif key == ord("q"):
            cv2.destroyWindow(title)
            return None

    object_points = []
    print(f"\nManual partial-grid annotation for {title}")
    print("For each clicked INTERNAL corner, enter its inner-corner col,row.")
    print("Valid cols: 0-8, valid rows: 0-5. Points must not all be on one line.")
    for i, point in enumerate(points, start=1):
        while True:
            text = input(f"Point {i} at ({point[0]:.1f}, {point[1]:.1f}) col,row: ").strip()
            try:
                col_text, row_text = text.split(",", 1)
                col = int(col_text)
                row = int(row_text)
                if not (0 <= col < CHECKERBOARD[0] and 0 <= row < CHECKERBOARD[1]):
                    raise ValueError
                object_points.append([col * square_size, row * square_size, 0.0])
                break
            except ValueError:
                print("Enter as col,row, for example 3,2.")

    object_points = np.asarray(object_points, dtype=np.float32)
    if np.linalg.matrix_rank(object_points[:, :2] - object_points[:1, :2]) < 2:
        print("Skipped: clicked points are collinear. Need points from at least two rows/columns.")
        return None
    return {
        "type": "partial-grid",
        "image_points": points,
        "object_points": object_points.tolist(),
    }


def click_line_segment(image, title):
    points = []
    preview = image.copy()

    def on_mouse(event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN or len(points) >= 2:
            return
        points.append([float(x), float(y)])
        cv2.circle(preview, (x, y), 7, (0, 255, 255), -1, cv2.LINE_AA)
        cv2.putText(preview, str(len(points)), (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        if len(points) == 2:
            p0 = tuple(int(v) for v in points[0])
            p1 = tuple(int(v) for v in points[1])
            cv2.line(preview, p0, p1, (0, 255, 255), 3, cv2.LINE_AA)

    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(title, on_mouse)
    while True:
        shown = preview.copy()
        cv2.putText(
            shown,
            "Click the two endpoints of the visible LiDAR-hit line on the board. Enter=save, u=undo, q=skip",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.imshow(title, shown)
        key = cv2.waitKey(20) & 0xFF
        if key == ord("u") and points:
            points.pop()
            preview = image.copy()
        elif key in (13, 10) and len(points) == 2:
            cv2.destroyWindow(title)
            return {"type": "line", "image_points": points}
        elif key == ord("q"):
            cv2.destroyWindow(title)
            return None


def annotation_to_pose(annotation, outer_objp, K, D):
    if isinstance(annotation, list):
        return solve_board_plane_from_points(np.asarray(annotation, dtype=np.float32), outer_objp, K, D)
    if annotation.get("type") == "outer":
        return solve_board_plane_from_points(np.asarray(annotation["image_points"], dtype=np.float32), outer_objp, K, D)
    if annotation.get("type") == "partial-grid":
        return solve_board_plane_from_points(
            np.asarray(annotation["image_points"], dtype=np.float32),
            np.asarray(annotation["object_points"], dtype=np.float32),
            K,
            D,
        )
    return None


def parse_angle(folder):
    match = re.search(r"_(\d+)$", folder.name)
    return 0.0 if match is None else float(match.group(1))


def parse_distance_m(path):
    text = path.read_text(encoding="utf-8").strip()
    value = float(text)
    return value / 100.0 if value > 2.0 else value


def lidar_points_from_csv(path):
    data = np.genfromtxt(path, delimiter=",", names=True)
    if data.size == 0:
        return np.empty((0, 3)), np.empty((0,))
    angles = np.deg2rad(np.asarray(data["angle_degrees"], dtype=float))
    ranges = np.asarray(data["distance_meters"], dtype=float)
    x = ranges * np.cos(angles)
    y = ranges * np.sin(angles)
    points = np.column_stack([x, y, np.zeros_like(x)])
    return points, ranges


def largest_angle_cluster(points, ranges, window_center, window, expected_length, length_tolerance):
    mask = np.abs(ranges - window_center) <= window
    if np.count_nonzero(mask) < 4:
        nearest = np.argsort(np.abs(ranges - window_center))[: max(4, min(20, len(ranges)))]
        mask = np.zeros_like(ranges, dtype=bool)
        mask[nearest] = True
    candidates = points[mask]
    if len(candidates) <= 2:
        return candidates
    angles = np.unwrap(np.arctan2(candidates[:, 1], candidates[:, 0]))
    order = np.argsort(angles)
    candidates = candidates[order]
    angles = angles[order]
    splits = np.where(np.diff(angles) > np.deg2rad(4.0))[0] + 1
    clusters = np.split(candidates, splits)
    scored = []
    for cluster in clusters:
        if len(cluster) < 3:
            continue
        length = float(np.linalg.norm(cluster[-1] - cluster[0]))
        length_error = abs(length - expected_length)
        if length_error <= length_tolerance:
            scored.append((length_error, -len(cluster), cluster))
    if scored:
        return sorted(scored, key=lambda item: (item[0], item[1]))[0][2]
    return max(clusters, key=len)


def discover_capture_folders(data_root):
    roots = []
    if data_root is not None:
        roots.append(data_root)
    roots.extend([REPO_ROOT / "00_data_capture", REPO_ROOT / "data"])
    folders = []
    for root in roots:
        if not root.exists():
            continue
        folders.extend([p for p in root.iterdir() if p.is_dir() and p.name.startswith("checkerboard_")])
    return sorted(set(folders), key=lambda p: p.name)


def load_observations(args, left_calib):
    objp = checkerboard_object_points(left_calib["square_size"])
    outer_objp = board_outer_object_points(left_calib["square_size"])
    segment_squares = BOARD_SQUARES[0] if args.segment_axis == "width" else BOARD_SQUARES[1]
    expected_segment_length = segment_squares * left_calib["square_size"]
    annotations = load_annotations(args.annotations)
    observations = []
    skipped = []
    for folder in discover_capture_folders(args.data_root):
        angle = parse_angle(folder)
        for txt_path in sorted(folder.glob("pair_*.txt")):
            pair_id = txt_path.stem.split("_")[1]
            left_path = folder / f"pair_{pair_id}_rgb_L.png"
            right_path = folder / f"pair_{pair_id}_rgb_R.png"
            lidar_path = folder / f"pair_{pair_id}_lidar.csv"
            if not (left_path.exists() and right_path.exists() and lidar_path.exists()):
                skipped.append((folder.name, pair_id, "missing files"))
                continue
            image_l = cv2.imread(str(left_path))
            image_r = cv2.imread(str(right_path))
            if image_l is None or image_r is None:
                skipped.append((folder.name, pair_id, "unreadable image"))
                continue
            pose = solve_board_plane_rgb1(image_l, left_calib["K"], left_calib["D"], objp)
            annotation_key = f"{folder.name}/{pair_id}"
            annotation = None
            if pose is None:
                annotation = annotations.get(annotation_key)
                if annotation is None and args.annotate_missing:
                    if args.annotation_mode == "outer":
                        points = click_four_corners(image_l, annotation_key)
                        annotation = None if points is None else {"type": "outer", "image_points": points}
                    elif args.annotation_mode == "line":
                        annotation = click_line_segment(image_l, annotation_key)
                    else:
                        annotation = click_partial_grid_points(image_l, annotation_key, left_calib["square_size"])
                    if annotation is not None:
                        annotations[annotation_key] = annotation
                        save_annotations(args.annotations, annotations)
                if annotation is None:
                    skipped.append((folder.name, pair_id, "checkerboard pose failed; no manual annotation"))
                    continue
                if annotation.get("type") != "line":
                    pose = annotation_to_pose(annotation, outer_objp, left_calib["K"], left_calib["D"])
                    if pose is None:
                        skipped.append((folder.name, pair_id, "manual annotation pose failed"))
                        continue
            try:
                distance_m = parse_distance_m(txt_path)
            except ValueError:
                skipped.append((folder.name, pair_id, "non-numeric distance metadata"))
                continue
            lidar_points, ranges = lidar_points_from_csv(lidar_path)
            board_points = largest_angle_cluster(
                lidar_points,
                ranges,
                distance_m,
                args.distance_window,
                expected_segment_length,
                args.segment_length_tolerance,
            )
            if len(board_points) < 3:
                skipped.append((folder.name, pair_id, "too few lidar board points"))
                continue
            obs = {
                "id": f"{folder.name}_{pair_id}",
                "folder": folder.name,
                "pair_id": pair_id,
                "angle_deg": angle,
                "distance_m": distance_m,
                "image_l": image_l,
                "image_r": image_r,
                "left_path": left_path,
                "right_path": right_path,
                "lidar_points": board_points,
                "K": left_calib["K"],
                "D": left_calib["D"],
            }
            if pose is not None:
                obs.update(
                    {
                        "constraint_type": "plane",
                        "plane_n": pose["normal"],
                        "plane_d": pose["d"],
                        "board_R": cv2.Rodrigues(pose["rvec"])[0],
                        "board_t": pose["tvec"].reshape(3),
                        "board_width_m": BOARD_SQUARES[0] * left_calib["square_size"],
                        "board_height_m": BOARD_SQUARES[1] * left_calib["square_size"],
                        "segment_axis": args.segment_axis,
                        "corners_l": pose["corners"],
                        "line_uv": checkerboard_axis_line(pose["full_corners"], args.segment_axis)
                        if "full_corners" in pose
                        else None,
                    }
                )
                if obs["line_uv"] is None:
                    del obs["line_uv"]
            else:
                obs.update(
                    {
                        "constraint_type": "line",
                        "line_uv": np.asarray(annotation["image_points"], dtype=np.float64),
                    }
                )
            observations.append(obs)
    return observations, skipped


def transform_points(points, params):
    R, _ = cv2.Rodrigues(params[:3].reshape(3, 1))
    return (R @ points.T).T + params[3:6]


def point_plane_residuals(params, observations):
    residuals = []
    for obs_index, obs in enumerate(observations):
        transformed = transform_points(obs["lidar_points"], params)
        if obs["constraint_type"] == "plane":
            board_points = (obs["board_R"].T @ (transformed - obs["board_t"]).T).T
            x = board_points[:, 0]
            y = board_points[:, 1]
            z = board_points[:, 2]

            # The LiDAR scan is a line segment on the finite checkerboard, not
            # just any set of points on the infinite board plane.
            strip_position = params[6 + obs_index]
            if obs["segment_axis"] == "height":
                const_coord = x
                span_coord = y
                span_length = obs["board_height_m"]
            else:
                const_coord = y
                span_coord = x
                span_length = obs["board_width_m"]

            residuals.extend(BOARD_STRIP_WEIGHT * z)
            residuals.extend(BOARD_STRIP_WEIGHT * (const_coord - strip_position))
            residuals.append(BOARD_STRIP_WEIGHT * (float(np.min(span_coord)) - 0.0))
            residuals.append(BOARD_STRIP_WEIGHT * (float(np.max(span_coord)) - span_length))

            outside_x = np.maximum(0.0, np.maximum(-x, x - obs["board_width_m"]))
            outside_y = np.maximum(0.0, np.maximum(-y, y - obs["board_height_m"]))
            residuals.extend(BOARD_BOUNDS_WEIGHT * outside_x)
            residuals.extend(BOARD_BOUNDS_WEIGHT * outside_y)
            if "line_uv" in obs:
                uv = project_fisheye(transformed, obs["K"], obs["D"])
                p0, p1 = obs["line_uv"]
                direction = p1 - p0
                length = np.linalg.norm(direction)
                if length >= 1.0:
                    direction = direction / length
                    normal = np.array([-direction[1], direction[0]])
                    rel = uv - p0
                    perp_px = rel @ normal
                    along_px = rel @ direction
                    outside_start = np.maximum(0.0, -along_px)
                    outside_end = np.maximum(0.0, along_px - length)
                    residuals.extend(IMAGE_LINE_WEIGHT * perp_px)
                    residuals.extend(IMAGE_LINE_WEIGHT * outside_start)
                    residuals.extend(IMAGE_LINE_WEIGHT * outside_end)
        else:
            uv = project_fisheye(transformed, obs["K"], obs["D"])
            p0, p1 = obs["line_uv"]
            direction = p1 - p0
            length = np.linalg.norm(direction)
            if length < 1.0:
                continue
            direction = direction / length
            normal = np.array([-direction[1], direction[0]])
            rel = uv - p0
            perp_px = rel @ normal
            along_px = rel @ direction
            residuals.extend(0.001 * perp_px)
            residuals.append(0.001 * (float(np.min(along_px)) - 0.0))
            residuals.append(0.001 * (float(np.max(along_px)) - length))
    return np.asarray(residuals)


def fit_transform(observations):
    strip_initial = []
    lower_strip = []
    upper_strip = []
    for obs_index, obs in enumerate(observations):
        if obs.get("constraint_type") == "plane":
            if obs["segment_axis"] == "height":
                strip_initial.append(0.5 * obs["board_width_m"])
                lower_strip.append(0.0)
                upper_strip.append(obs["board_width_m"])
            else:
                strip_initial.append(0.5 * obs["board_height_m"])
                lower_strip.append(0.0)
                upper_strip.append(obs["board_height_m"])
        else:
            strip_initial.append(0.0)
            lower_strip.append(-1.0)
            upper_strip.append(1.0)
    strip_initial = np.asarray(strip_initial, dtype=float)

    seeds = [
        np.array([0, 0, 0, 0, 0, 0.4], dtype=float),
        np.array([np.pi / 2, 0, 0, 0, 0, 0.4], dtype=float),
        np.array([-np.pi / 2, 0, 0, 0, 0, 0.4], dtype=float),
        np.array([0, np.pi / 2, 0, 0, 0, 0.4], dtype=float),
        np.array([0, -np.pi / 2, 0, 0, 0, 0.4], dtype=float),
    ]
    lower = np.r_[[-np.inf] * 6, lower_strip]
    upper = np.r_[[np.inf] * 6, upper_strip]
    best = None
    for seed in seeds:
        full_seed = np.r_[seed, strip_initial]
        result = least_squares(
            point_plane_residuals,
            full_seed,
            args=(observations,),
            bounds=(lower, upper),
            loss="soft_l1",
            f_scale=0.02,
            max_nfev=3000,
        )
        if best is None or result.cost < best.cost:
            best = result
    return best.x, best


def summarize(values):
    arr = np.abs(np.asarray(values, dtype=float))
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p90": float(np.percentile(arr, 90)),
        "max": float(np.max(arr)),
    }


def project_fisheye(points_cam, K, D):
    rvec = np.zeros((3, 1), dtype=np.float64)
    tvec = np.zeros((3, 1), dtype=np.float64)
    projected, _ = cv2.fisheye.projectPoints(points_cam.reshape(1, -1, 3).astype(np.float64), rvec, tvec, K, D)
    return projected.reshape(-1, 2)


def in_frame_count(points_cam, uv, image_shape):
    h, w = image_shape[:2]
    mask = (
        (points_cam[:, 2] > 0)
        & (uv[:, 0] >= 0)
        & (uv[:, 0] < w)
        & (uv[:, 1] >= 0)
        & (uv[:, 1] < h)
    )
    return int(np.count_nonzero(mask)), mask


def draw_overlay(path, image, points_cam, uv, color, label):
    overlay = image.copy()
    h, w = overlay.shape[:2]
    count, mask = in_frame_count(points_cam, uv, image.shape)
    for x, y in uv[mask]:
        cv2.circle(overlay, (int(round(x)), int(round(y))), 9, color, -1, cv2.LINE_AA)
        cv2.circle(overlay, (int(round(x)), int(round(y))), 11, (0, 0, 0), 2, cv2.LINE_AA)
    for x, y in uv[~mask]:
        x_clip = int(np.clip(round(x), 0, w - 1))
        y_clip = int(np.clip(round(y), 43, h - 1))
        cv2.drawMarker(
            overlay,
            (x_clip, y_clip),
            (0, 0, 255),
            markerType=cv2.MARKER_TILTED_CROSS,
            markerSize=18,
            thickness=2,
            line_type=cv2.LINE_AA,
        )
    cv2.rectangle(overlay, (0, 0), (w, 42), (0, 0, 0), -1)
    cv2.putText(
        overlay,
        f"{label}: {count}/{len(uv)} projected LiDAR board points in frame",
        (14, 29),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), overlay)
    return count


def evaluate_and_save(args, observations, params, left_calib, right_calib, stereo, skipped):
    R_lidar, _ = cv2.Rodrigues(params[:3].reshape(3, 1))
    t_lidar = params[3:6].reshape(3, 1)
    rows = []
    all_residuals = []
    by_distance = {}
    rgb1_dir = args.output_dir / "rgb1_overlays"
    rgb2_dir = args.output_dir / "rgb2_overlays"
    R12, t12 = stereo["R"], stereo["t"]

    for obs_index, obs in enumerate(observations):
        points_rgb1 = transform_points(obs["lidar_points"], params)
        if obs["constraint_type"] == "plane":
            residuals = points_rgb1 @ obs["plane_n"] + obs["plane_d"]
            all_residuals.extend(residuals.tolist())
            by_distance.setdefault(obs["distance_m"], []).extend(np.abs(residuals).tolist())
            stats = summarize(residuals)
            if "line_uv" in obs:
                p0, p1 = obs["line_uv"]
                direction = p1 - p0
                direction = direction / np.linalg.norm(direction)
                normal = np.array([-direction[1], direction[0]])
                uv_line_probe = project_fisheye(points_rgb1, left_calib["K"], left_calib["D"])
                line_errors = np.abs((uv_line_probe - p0) @ normal)
                line_mean_px = float(np.mean(line_errors))
                line_max_px = float(np.max(line_errors))
            else:
                line_mean_px = ""
                line_max_px = ""
        else:
            residuals = np.asarray([], dtype=float)
            p0, p1 = obs["line_uv"]
            direction = p1 - p0
            direction = direction / np.linalg.norm(direction)
            normal = np.array([-direction[1], direction[0]])
            uv_line_probe = project_fisheye(points_rgb1, left_calib["K"], left_calib["D"])
            line_errors = np.abs((uv_line_probe - p0) @ normal)
            stats = {"mean": "", "median": "", "p90": "", "max": ""}
            line_mean_px = float(np.mean(line_errors))
            line_max_px = float(np.max(line_errors))

        uv1 = project_fisheye(points_rgb1, left_calib["K"], left_calib["D"])
        points_rgb2 = (R12 @ points_rgb1.T).T + t12.reshape(3)
        uv2 = project_fisheye(points_rgb2, right_calib["K"], right_calib["D"])
        rgb1_in_frame = draw_overlay(rgb1_dir / f"{obs['id']}.png", obs["image_l"], points_rgb1, uv1, (0, 255, 255), "RGB1")
        rgb2_in_frame = draw_overlay(rgb2_dir / f"{obs['id']}.png", obs["image_r"], points_rgb2, uv2, (0, 128, 255), "RGB2")

        rows.append(
            {
                "capture_id": obs["id"],
                "folder": obs["folder"],
                "pair_id": obs["pair_id"],
                "constraint_type": obs["constraint_type"],
                "angle_deg": obs["angle_deg"],
                "distance_m": obs["distance_m"],
                "lidar_board_points": len(obs["lidar_points"]),
                "rgb1_projected_in_frame": rgb1_in_frame,
                "rgb2_projected_in_frame": rgb2_in_frame,
                "rgb1_line_mean_px": line_mean_px,
                "rgb1_line_max_px": line_max_px,
                "board_strip_position_m": params[6 + obs_index] if obs["constraint_type"] == "plane" else "",
                "mean_abs_point_to_plane_m": stats["mean"],
                "median_abs_point_to_plane_m": stats["median"],
                "p90_abs_point_to_plane_m": stats["p90"],
                "max_abs_point_to_plane_m": stats["max"],
            }
        )

    per_distance = []
    for cm in range(10, 101, 10):
        distance = cm / 100.0
        vals = by_distance.get(distance, [])
        if vals:
            per_distance.append({"distance_m": distance, **summarize(vals), "count": len(vals)})
        else:
            per_distance.append({"distance_m": distance, "mean": None, "median": None, "p90": None, "max": None, "count": 0})
    summary = summarize(all_residuals) if all_residuals else {"mean": None, "median": None, "p90": None, "max": None}
    result = {
        "summary_m": summary,
        "per_distance_m": per_distance,
        "observation_count": len(observations),
        "point_count": len(all_residuals),
        "skipped_observations": [
            {"folder": folder, "pair_id": pair_id, "reason": reason}
            for folder, pair_id, reason in skipped
        ],
        "R_lidar_to_rgb1": R_lidar.tolist(),
        "t_lidar_to_rgb1": t_lidar.reshape(3).tolist(),
        "T_rgb1_lidar": np.vstack([np.hstack([R_lidar, t_lidar]), [0.0, 0.0, 0.0, 1.0]]).tolist(),
        "board_strip_positions_m": {
            obs["id"]: float(params[6 + obs_index])
            for obs_index, obs in enumerate(observations)
            if obs["constraint_type"] == "plane"
        },
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "lidar_rgb1_calibration_eval.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    with (args.output_dir / "lidar_rgb1_calibration_eval_per_capture.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return result


def leave_one_distance_out(args, observations, left_calib, right_calib, stereo):
    rows = []
    distances = sorted({obs["distance_m"] for obs in observations})
    for distance in distances:
        train = [obs for obs in observations if obs["distance_m"] != distance]
        test = [obs for obs in observations if obs["distance_m"] == distance]
        if len(train) < 3:
            continue
        params, _ = fit_transform(train)
        residuals = point_plane_residuals(params, test)
        stats = summarize(residuals)
        rows.append({"held_out_distance_m": distance, "test_captures": len(test), **stats})
    if rows:
        with (args.output_dir / "lidar_rgb1_leave_one_distance_out.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)


def main():
    args = parse_args()
    left_calib = load_intrinsics(args.left_calibration)
    right_calib = load_intrinsics(args.right_calibration)
    stereo = load_stereo(args.stereo_calibration)
    observations, skipped = load_observations(args, left_calib)
    if len(observations) < 3:
        raise RuntimeError(f"Need at least 3 usable observations; found {len(observations)}. Skipped: {skipped}")

    params, result = fit_transform(observations)
    R_lidar, _ = cv2.Rodrigues(params[:3].reshape(3, 1))
    t_lidar = params[3:6].reshape(3)
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output_file,
        R_lidar_to_rgb1=R_lidar,
        t_lidar_to_rgb1=t_lidar,
        T_rgb1_lidar=np.vstack([np.hstack([R_lidar, t_lidar.reshape(3, 1)]), [0.0, 0.0, 0.0, 1.0]]),
        rvec_lidar_to_rgb1=params[:3],
        source=np.array("checkerboard point-to-plane fisheye RGB1"),
    )
    eval_result = evaluate_and_save(args, observations, params, left_calib, right_calib, stereo, skipped)
    leave_one_distance_out(args, observations, left_calib, right_calib, stereo)

    print("\nLiDAR to RGB1 calibration complete")
    print("----------------------------------")
    print(f"Usable observations: {len(observations)}")
    print(f"Skipped observations: {len(skipped)}")
    print(f"Mean/median/p90/max point-to-plane: {eval_result['summary_m']['mean']:.4f} / {eval_result['summary_m']['median']:.4f} / {eval_result['summary_m']['p90']:.4f} / {eval_result['summary_m']['max']:.4f} m")
    print(f"Saved {args.output_file.resolve()}")
    print(f"Saved outputs in {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
