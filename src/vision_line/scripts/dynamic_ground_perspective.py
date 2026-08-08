"""Create a ground-only bird's-eye view from a live camera stream."""

from __future__ import annotations

import numpy as np

# ======================== 直接修改以下运行参数 ========================
CAMERA_INDEX = 0
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
FRAME_WIDTH = 320
FRAME_HEIGHT = 240

# 四点围成区域的实际尺寸，请使用现场测量值。
REAL_WIDTH_M = 0.5
REAL_LENGTH_M = 0.75
PIXELS_PER_M = 200.0

# 固定地面起始行：只保留 y >= GROUND_START_Y 的图像区域。
GROUND_START_Y = 139

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


def fixed_ground_mask(height: int, width: int, start_y: int) -> np.ndarray:
    """Return a mask that keeps every pixel at or below ``start_y``."""
    if height < 0 or width < 0:
        raise ValueError("height and width must not be negative")
    top = int(np.clip(start_y, 0, height))
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[top:, :] = 255
    return mask


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


def transform_points(homography: np.ndarray, points: np.ndarray) -> np.ndarray:
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
    """Return TL, TR, BR, BL points enclosing the fixed ground mask."""
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
    if not getattr(warp_ground, "_printed", False):
        print("homography:\n", homography)
        warp_ground._printed = True
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


def prepare_camera_frame(frame, camera, cv2_module) -> np.ndarray:
    """Match the correction, flip, and resize pipeline used for calibration."""
    corrected = camera.correct_img(frame)
    corrected = cv2_module.flip(corrected, 1)
    return cv2_module.resize(
        corrected,
        (FRAME_WIDTH, FRAME_HEIGHT),
        interpolation=cv2_module.INTER_AREA,
    )


def should_exit(key_code: int) -> bool:
    """Return whether an OpenCV key code requests stream termination."""
    key = key_code & 0xFF
    return key in (27, ord("q"), ord("Q"))


def process_camera_frame(
    raw_frame,
    camera,
    cv2_module,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Prepare, mask, and warp one frame from the camera."""
    frame = prepare_camera_frame(raw_frame, camera, cv2_module)
    ground_mask = fixed_ground_mask(
        height=frame.shape[0],
        width=frame.shape[1],
        start_y=GROUND_START_Y,
    )
    bird_view = warp_ground(
        frame,
        ground_mask,
        SOURCE_POINTS,
        width_m=REAL_WIDTH_M,
        length_m=REAL_LENGTH_M,
        pixels_per_m=PIXELS_PER_M,
    )
    return frame, bird_view, ground_mask


def run_camera_stream(camera, cv2_module, frame_processor=None) -> None:
    """Process camera frames until q/Esc and always release GUI resources."""
    if frame_processor is None:
        frame_processor = lambda raw: process_camera_frame(raw, camera, cv2_module)

    try:
        while True:
            raw_frame = camera.get_picture()
            frame, bird_view, ground_mask = frame_processor(raw_frame)
            cv2_module.imshow("Camera Frame", frame)
            cv2_module.imshow("Bird's Eye View", bird_view)
            cv2_module.imshow("Ground Mask", ground_mask)
            if should_exit(cv2_module.waitKey(1)):
                break
    except KeyboardInterrupt:
        pass
    finally:
        camera.close()
        cv2_module.destroyAllWindows()


def main() -> int:
    """Open the configured camera and process frames until q/Esc."""
    try:
        import cv2  # type: ignore
    except (ImportError, OSError) as exc:
        raise RuntimeError("OpenCV (cv2) is required to run this command") from exc

    from camera_capture import CameraCapture

    camera = CameraCapture(CAMERA_INDEX, CAMERA_WIDTH, CAMERA_HEIGHT)
    print("Camera stream started. Press q or Esc to exit.")
    run_camera_stream(camera, cv2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
