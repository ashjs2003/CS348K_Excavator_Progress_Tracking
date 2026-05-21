"""Shared settings for the LiDAR-camera calibration prototype."""

# Checkerboard geometry.
SQUARE_SIZE_M = 0.025
BOARD_SQUARES_ACROSS = 10
BOARD_SQUARES_DOWN = 7
CHECKERBOARD_INNER_CORNERS = (9, 6)

# If the LiDAR scan crosses the full board left-to-right, use "width".
# If it crosses the full board top-to-bottom, use "height".
EXPECTED_SEGMENT_AXIS = "width"

# Full physical board dimensions. 0.025 m is one square, not the full board.
BOARD_WIDTH_M = BOARD_SQUARES_ACROSS * SQUARE_SIZE_M   # 0.250 m
BOARD_HEIGHT_M = BOARD_SQUARES_DOWN * SQUARE_SIZE_M   # 0.175 m
EXPECTED_SEGMENT_LENGTH_M = (
    BOARD_WIDTH_M if EXPECTED_SEGMENT_AXIS == "width" else BOARD_HEIGHT_M
)

# You said captures will place the board within 1 m, so the capture preview filters
# farther points to make the board segment easier to see.
MIN_CAPTURE_DISTANCE_M = 0.02
MAX_CAPTURE_DISTANCE_M = 1.0

# Selection quality hints. These are not hard calibration limits.
GOOD_LINE_RMS_M = 0.01
GOOD_LENGTH_TOLERANCE_M = 0.03
