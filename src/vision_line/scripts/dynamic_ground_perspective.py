"""Detect the blue competition map and create a ground-only bird's-eye view."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# ======================== 直接修改以下运行参数 ========================
SCRIPT_DIR = Path(__file__).resolve().parent
PICTURES_DIR = SCRIPT_DIR.parent / "pictures"

INPUT_IMAGE = PICTURES_DIR / "captured_image_20260720_150144.jpg"

# 四点围成区域的实际尺寸；请替换为现场测量值。
REAL_WIDTH_M = 0.5
REAL_LENGTH_M = 0.75
PIXELS_PER_M = 200.0

FALLBACK_Y = 139
LOWER_HSV = (90, 60, 35)
UPPER_HSV = (135, 255, 255)
MIN_AREA_RATIO = 0.05
BOTTOM_BAND_HEIGHT = 3
MORPHOLOGY_KERNEL_SIZE = 7
BOUNDARY_SMOOTHING_ALPHA = 0.35
BOUNDARY_MAX_STEP = 5.0

SOURCE_POINTS = np.array(
    [
        [114.0, 146.0],
        [206.0, 146.0],
        [271.0, 184.0],
        [35.0, 187.0],
    ],
    dtype=np.float32,
)
# ====================================================================


@dataclass(frozen=True)
class GroundDetectionConfig:
    """Parameters for blue-ground segmentation and temporal tracking."""

    lower_hsv: tuple[int, int, int] = LOWER_HSV
    upper_hsv: tuple[int, int, int] = UPPER_HSV
    fallback_y: int = FALLBACK_Y
    min_area_ratio: float = MIN_AREA_RATIO
    bottom_band_height: int = BOTTOM_BAND_HEIGHT
    morphology_kernel_size: int = MORPHOLOGY_KERNEL_SIZE
    alpha: float = BOUNDARY_SMOOTHING_ALPHA
    max_step: float = BOUNDARY_MAX_STEP

    def __post_init__(self) -> None:
        if not 0.0 <= self.min_area_ratio <= 1.0:
            raise ValueError("min_area_ratio must be between 0 and 1")
        if self.bottom_band_height <= 0:
            raise ValueError("bottom_band_height must be positive")
        if self.morphology_kernel_size <= 0:
            raise ValueError("morphology_kernel_size must be positive")
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError("alpha must be between 0 and 1")
        if self.max_step <= 0.0:
            raise ValueError("max_step must be positive")


class GroundBoundaryTracker:
    """Track the upper edge of the bottom-connected blue map region."""

    def __init__(self, config: GroundDetectionConfig | None = None) -> None:
        self.config = config or GroundDetectionConfig()
        self.previous_boundary: np.ndarray | None = None

    def update_from_blue_mask(
        self,
        blue_mask: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Update tracking from a binary blue mask and return ROI and boundary."""
        if blue_mask.ndim != 2:
            raise ValueError("blue_mask must be a two-dimensional array")
        height, width = blue_mask.shape
        component = select_bottom_connected_component(
            blue_mask,
            min_area_ratio=self.config.min_area_ratio,
            bottom_band_height=self.config.bottom_band_height,
        )

        if component is None:
            if self.previous_boundary is not None:
                boundary = self.previous_boundary.copy()
            else:
                boundary = np.full(
                    width,
                    float(np.clip(self.config.fallback_y, 0, max(0, height - 1))),
                )
        else:
            detected = fit_top_boundary(component, self.config.fallback_y)
            boundary = smooth_boundary(
                detected,
                self.previous_boundary,
                alpha=self.config.alpha,
                max_step=self.config.max_step,
            )
        boundary = np.clip(boundary, 0.0, max(0.0, float(height - 1)))
        self.previous_boundary = boundary.copy()
        return boundary_to_mask(boundary, height), boundary

    def detect(self, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Segment blue pixels from a BGR frame and update the tracked boundary."""
        try:
            import cv2  # type: ignore
        except (ImportError, OSError) as exc:
            raise RuntimeError("OpenCV (cv2) is required for ground detection") from exc
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("frame must be a BGR image with shape (H, W, 3)")

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        blue_mask = cv2.inRange(
            hsv,
            np.asarray(self.config.lower_hsv, dtype=np.uint8),
            np.asarray(self.config.upper_hsv, dtype=np.uint8),
        )
        size = int(self.config.morphology_kernel_size)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, kernel)
        blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, kernel)
        return self.update_from_blue_mask(blue_mask)


def select_bottom_connected_component(
    binary_mask: np.ndarray,
    min_area_ratio: float = 0.05,
    bottom_band_height: int = 3,
) -> np.ndarray | None:
    """Return the largest foreground component touching the bottom band."""
    if binary_mask.ndim != 2:
        raise ValueError("binary_mask must be a two-dimensional array")
    if not 0.0 <= min_area_ratio <= 1.0:
        raise ValueError("min_area_ratio must be between 0 and 1")

    height, width = binary_mask.shape
    if height == 0 or width == 0:
        return None
    bottom_band_height = max(1, min(int(bottom_band_height), height))
    foreground = binary_mask.astype(bool)
    min_area = int(np.ceil(height * width * min_area_ratio))

    # OpenCV is used in the target runtime. The NumPy/Python fallback keeps
    # the numerical helpers testable on development hosts without cv2 DLLs.
    try:
        import cv2  # type: ignore
    except (ImportError, OSError):
        cv2 = None

    if cv2 is not None:
        count, labels, stats, _ = cv2.connectedComponentsWithStats(
            foreground.astype(np.uint8),
            connectivity=8,
        )
        candidates: list[tuple[int, int]] = []
        bottom_start = height - bottom_band_height
        for label in range(1, count):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area < min_area:
                continue
            if np.any(labels[bottom_start:, :] == label):
                candidates.append((area, label))
        if not candidates:
            return None
        selected_label = max(candidates)[1]
        return labels == selected_label

    visited = np.zeros_like(foreground, dtype=bool)
    best_pixels: list[tuple[int, int]] = []
    bottom_start = height - bottom_band_height
    neighbours = (
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    )

    for start_y in range(bottom_start, height):
        for start_x in range(width):
            if not foreground[start_y, start_x] or visited[start_y, start_x]:
                continue
            queue = deque([(start_y, start_x)])
            visited[start_y, start_x] = True
            pixels: list[tuple[int, int]] = []
            while queue:
                y, x = queue.popleft()
                pixels.append((y, x))
                for dy, dx in neighbours:
                    ny, nx = y + dy, x + dx
                    if not (0 <= ny < height and 0 <= nx < width):
                        continue
                    if foreground[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        queue.append((ny, nx))
            if len(pixels) >= min_area and len(pixels) > len(best_pixels):
                best_pixels = pixels

    if not best_pixels:
        return None
    selected = np.zeros_like(foreground, dtype=bool)
    ys, xs = zip(*best_pixels)
    selected[np.asarray(ys), np.asarray(xs)] = True
    return selected


def fit_top_boundary(
    component_mask: np.ndarray,
    fallback_y: float,
) -> np.ndarray:
    """Fit a robust straight upper edge to a ground component."""
    if component_mask.ndim != 2:
        raise ValueError("component_mask must be a two-dimensional array")
    height, width = component_mask.shape
    if width == 0:
        return np.empty(0, dtype=np.float64)

    component = component_mask.astype(bool)
    valid = np.any(component, axis=0)
    if height == 0 or np.count_nonzero(valid) < 2:
        return np.full(width, float(fallback_y), dtype=np.float64)

    xs = np.flatnonzero(valid).astype(np.float64)
    top = np.argmax(component[:, valid], axis=0).astype(np.float64)
    median = float(np.median(top))
    mad = float(np.median(np.abs(top - median)))
    tolerance = max(4.0, 3.0 * mad)
    inliers = np.abs(top - median) <= tolerance
    if np.count_nonzero(inliers) < 2:
        return np.full(width, float(fallback_y), dtype=np.float64)

    slope, intercept = np.polyfit(xs[inliers], top[inliers], deg=1)
    boundary = intercept + slope * np.arange(width, dtype=np.float64)
    return np.clip(boundary, 0.0, max(0.0, float(height - 1)))


def smooth_boundary(
    detected: np.ndarray,
    previous: np.ndarray | None,
    alpha: float = 0.35,
    max_step: float = 5.0,
) -> np.ndarray:
    """Exponentially smooth a boundary while limiting per-frame motion."""
    detected_array = np.asarray(detected, dtype=np.float64)
    if previous is None:
        return detected_array.copy()
    previous_array = np.asarray(previous, dtype=np.float64)
    if detected_array.shape != previous_array.shape:
        raise ValueError("detected and previous boundaries must have the same shape")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be between 0 and 1")
    if max_step <= 0.0:
        raise ValueError("max_step must be positive")

    delta = alpha * (detected_array - previous_array)
    return previous_array + np.clip(delta, -max_step, max_step)


def boundary_to_mask(boundary: np.ndarray, height: int) -> np.ndarray:
    """Convert a per-column top boundary to a uint8 ground ROI mask."""
    if height < 0:
        raise ValueError("height must not be negative")
    boundary_array = np.asarray(boundary, dtype=np.float64)
    if boundary_array.ndim != 1:
        raise ValueError("boundary must be one-dimensional")
    if height == 0:
        return np.zeros((0, boundary_array.size), dtype=np.uint8)

    top = np.clip(np.rint(boundary_array).astype(int), 0, height)
    rows = np.arange(height, dtype=int)[:, None]
    return np.where(rows >= top[None, :], 255, 0).astype(np.uint8)


def metric_destination(
    width_m: float,
    length_m: float,
    pixels_per_m: float,
) -> np.ndarray:
    """Build TL, TR, BR, BL destination points in metric pixel units."""
    if width_m <= 0.0 or length_m <= 0.0:
        raise ValueError("width_m and length_m must be positive")
    if pixels_per_m <= 0.0:
        raise ValueError("pixels_per_m must be positive")
    width_px = float(width_m * pixels_per_m)
    length_px = float(length_m * pixels_per_m)
    return np.array(
        [
            [0.0, 0.0],
            [width_px, 0.0],
            [width_px, length_px],
            [0.0, length_px],
        ],
        dtype=np.float32,
    )


def homography_from_four_points(
    source_points: np.ndarray,
    destination_points: np.ndarray,
) -> np.ndarray:
    """Solve the projective transform from four point correspondences."""
    source = np.asarray(source_points, dtype=np.float64)
    destination = np.asarray(destination_points, dtype=np.float64)
    if source.shape != (4, 2) or destination.shape != (4, 2):
        raise ValueError("source_points and destination_points must have shape (4, 2)")

    coefficients: list[list[float]] = []
    values: list[float] = []
    for (x, y), (u, v) in zip(source, destination):
        coefficients.append([x, y, 1.0, 0.0, 0.0, 0.0, -u * x, -u * y])
        values.append(u)
        coefficients.append([0.0, 0.0, 0.0, x, y, 1.0, -v * x, -v * y])
        values.append(v)
    try:
        solution = np.linalg.solve(
            np.asarray(coefficients, dtype=np.float64),
            np.asarray(values, dtype=np.float64),
        )
    except np.linalg.LinAlgError as exc:
        raise ValueError("point correspondences do not define a homography") from exc
    return np.append(solution, 1.0).reshape(3, 3)


def transform_points(
    homography: np.ndarray,
    points: np.ndarray,
) -> np.ndarray:
    """Apply a homography to an array of two-dimensional points."""
    matrix = np.asarray(homography, dtype=np.float64)
    point_array = np.asarray(points, dtype=np.float64)
    if matrix.shape != (3, 3):
        raise ValueError("homography must have shape (3, 3)")
    if point_array.ndim != 2 or point_array.shape[1] != 2:
        raise ValueError("points must have shape (N, 2)")

    homogeneous = np.column_stack(
        (point_array, np.ones(point_array.shape[0], dtype=np.float64))
    )
    transformed = (matrix @ homogeneous.T).T
    divisor = transformed[:, 2]
    if np.any(np.isclose(divisor, 0.0)):
        raise ValueError("a transformed point lies on the homography horizon")
    return transformed[:, :2] / divisor[:, None]


def ground_roi_points(mask: np.ndarray) -> np.ndarray:
    """Return TL, TR, BR, BL points enclosing a per-column ground mask."""
    mask_array = np.asarray(mask)
    if mask_array.ndim != 2:
        raise ValueError("mask must be two-dimensional")
    foreground = mask_array.astype(bool)
    valid_columns = np.flatnonzero(np.any(foreground, axis=0))
    if valid_columns.size == 0:
        raise ValueError("mask contains no ground pixels")

    left_x = int(valid_columns[0])
    right_x = int(valid_columns[-1])
    left_top = int(np.flatnonzero(foreground[:, left_x])[0])
    right_top = int(np.flatnonzero(foreground[:, right_x])[0])
    bottom = int(np.flatnonzero(np.any(foreground, axis=1))[-1])
    return np.array(
        [
            [left_x, left_top],
            [right_x, right_top],
            [right_x, bottom],
            [left_x, bottom],
        ],
        dtype=np.float64,
    )


def expanded_homography(
    homography: np.ndarray,
    roi_points: np.ndarray,
) -> tuple[np.ndarray, tuple[int, int]]:
    """Translate a homography so all transformed ROI points fit the canvas."""
    transformed = transform_points(homography, roi_points)
    minimum = np.floor(np.min(transformed, axis=0))
    maximum = np.ceil(np.max(transformed, axis=0))
    output_width = int(maximum[0] - minimum[0]) + 1
    output_height = int(maximum[1] - minimum[1]) + 1
    if output_width <= 0 or output_height <= 0:
        raise ValueError("transformed ROI has an invalid output size")

    translation = np.array(
        [
            [1.0, 0.0, -minimum[0]],
            [0.0, 1.0, -minimum[1]],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return translation @ np.asarray(homography, dtype=np.float64), (
        output_width,
        output_height,
    )


def warp_ground(
    frame: np.ndarray,
    ground_mask: np.ndarray,
    source_points: np.ndarray,
    width_m: float,
    length_m: float,
    pixels_per_m: float,
) -> np.ndarray:
    """Mask non-ground pixels and warp the complete ground ROI to bird view."""
    try:
        import cv2  # type: ignore
    except (ImportError, OSError) as exc:
        raise RuntimeError("OpenCV (cv2) is required to warp images") from exc

    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("frame must be a BGR image with shape (H, W, 3)")
    if ground_mask.shape != frame.shape[:2]:
        raise ValueError("ground_mask dimensions must match frame")

    destination = metric_destination(width_m, length_m, pixels_per_m)
    homography = homography_from_four_points(source_points, destination)
    homography, output_size = expanded_homography(
        homography,
        ground_roi_points(ground_mask),
    )
    masked_frame = np.where(ground_mask[:, :, None] != 0, frame, 0).astype(
        frame.dtype,
        copy=False,
    )
    return cv2.warpPerspective(
        masked_frame,
        homography,
        output_size,
        flags=cv2.INTER_LINEAR,
    )


def show_results(
    bird_view: np.ndarray,
    ground_mask: np.ndarray,
    cv2_module=None,
) -> None:
    """Display the bird view and ground mask until the user presses a key."""
    if cv2_module is None:
        try:
            import cv2 as cv2_module  # type: ignore
        except (ImportError, OSError) as exc:
            raise RuntimeError("OpenCV (cv2) is required to display images") from exc

    cv2_module.imshow("Bird's Eye View", bird_view)
    cv2_module.imshow("Ground Mask", ground_mask)
    cv2_module.waitKey(0)
    cv2_module.destroyAllWindows()


def main() -> int:
    """Process the image configured in the module-level parameter block."""
    try:
        import cv2  # type: ignore
    except (ImportError, OSError) as exc:
        raise RuntimeError("OpenCV (cv2) is required to run this command") from exc

    frame = cv2.imread(str(INPUT_IMAGE))
    if frame is None:
        raise ValueError(f"cannot read input image: {INPUT_IMAGE}")

    tracker = GroundBoundaryTracker(GroundDetectionConfig())
    ground_mask, boundary = tracker.detect(frame)
    bird_view = warp_ground(
        frame,
        ground_mask,
        SOURCE_POINTS,
        width_m=REAL_WIDTH_M,
        length_m=REAL_LENGTH_M,
        pixels_per_m=PIXELS_PER_M,
    )

    print(
        "ground boundary: "
        f"left={boundary[0]:.1f}, right={boundary[-1]:.1f}; "
        f"bird view={bird_view.shape[1]}x{bird_view.shape[0]}"
    )
    show_results(bird_view, ground_mask, cv2_module=cv2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
