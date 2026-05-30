"""
Live camera index viewer.

Run from this folder:
    python 00_live_camera_index_viewer.py
    python 00_live_camera_index_viewer.py --indices 0 1 2 3

This opens live previews for available camera indices and overlays the index on
each stream. Use it to confirm which physical camera should be L or R in
../config.yaml.

Controls:
    q / Esc - quit
"""

import argparse
import sys
from pathlib import Path

import cv2

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from utils.project_config import camera_index, frame_size

FRAME_SIZE = frame_size()
LEFT_CAMERA_INDEX = camera_index("L")
RIGHT_CAMERA_INDEX = camera_index("R")


def parse_args():
    parser = argparse.ArgumentParser(description="Show live camera streams labeled by camera index.")
    parser.add_argument(
        "--indices",
        nargs="+",
        type=int,
        default=[0, 1, 2, 3, 4, 5],
        help="Camera indices to try opening.",
    )
    return parser.parse_args()


def open_index(index):
    """Open a camera index with the same backend preference used by the workflow."""
    if sys.platform == "darwin":
        cap = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
    elif sys.platform == "win32":
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(index)

    if not cap.isOpened():
        cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        return None

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(FRAME_SIZE[0]))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(FRAME_SIZE[1]))
    return cap


def label_for_index(index):
    labels = []
    if index == LEFT_CAMERA_INDEX:
        labels.append("configured L")
    if index == RIGHT_CAMERA_INDEX:
        labels.append("configured R")
    return ", ".join(labels) if labels else "unassigned"


def draw_label(frame, index):
    text = f"camera index {index} | {label_for_index(index)} | q quit"
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 46), (0, 0, 0), -1)
    cv2.putText(
        frame,
        text,
        (18, 31),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    return frame


def main():
    args = parse_args()
    captures = []

    print(f"Configured L camera index: {LEFT_CAMERA_INDEX}")
    print(f"Configured R camera index: {RIGHT_CAMERA_INDEX}")
    print(f"Requested frame size: {FRAME_SIZE[0]}x{FRAME_SIZE[1]}")

    for index in args.indices:
        cap = open_index(index)
        if cap is None:
            print(f"Camera index {index}: not available")
            continue
        captures.append((index, cap))
        actual = (
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
        print(f"Camera index {index}: opened ({actual[0]}x{actual[1]}), {label_for_index(index)}")

    if not captures:
        print("No camera indices opened.")
        return

    try:
        while True:
            for index, cap in captures:
                ok, frame = cap.read()
                if not ok:
                    continue
                cv2.imshow(f"Camera index {index}", draw_label(frame, index))

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
    finally:
        for _, cap in captures:
            cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
