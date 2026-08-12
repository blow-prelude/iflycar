from __future__ import annotations

import argparse
import logging
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

BBox = tuple[int, int, int, int]
LOGGER = logging.getLogger(__name__)
WINDOW_NAME = "traditional_light_cv"


@dataclass(frozen=True)
class Config:
    roi_x: tuple[float, float] = (0.25, 0.78)
    roi_y: tuple[float, float] = (0.28, 0.78)
    reference_size: tuple[int, int] = (640, 480)
    green_low: tuple[int, int, int] = (35, 80, 100)
    green_high: tuple[int, int, int] = (100, 255, 255)
    red_low_1: tuple[int, int, int] = (0, 80, 100)
    red_high_1: tuple[int, int, int] = (12, 255, 255)
    red_low_2: tuple[int, int, int] = (165, 80, 100)
    red_high_2: tuple[int, int, int] = (179, 255, 255)
    # Slightly lower V tolerates dimmer illumination; shape-density filtering
    # below prevents the extra bright noise from becoming a candidate.
    bright_low: tuple[int, int, int] = (0, 0, 212)
    bright_high: tuple[int, int, int] = (179, 110, 255)
    close_kernel_size: int = 3
    component_width: tuple[int, int] = (15, 90)
    component_height: tuple[int, int] = (12, 90)
    min_component_area: int = 80
    min_component_fill_density: float = 0.15
    candidate_padding: int = 10
    min_color_score: int = 200
    min_color_density: float = 0.10


DEFAULT_CONFIG = Config()

PACKAGE_DIR = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PACKAGE_DIR.parents[1]
SAMPLE_IMAGE_PATHS = (
    PACKAGE_DIR / "pictures" / "02051.jpg",
    PACKAGE_DIR / "pictures" / "02052.jpg",
    PACKAGE_DIR / "pictures" / "003_0030.jpg",
    PACKAGE_DIR / "pictures" / "004_0001.jpg",
)
OUTPUT_DIR = WORKSPACE_ROOT / "build" / "traditional_light_cv_results"


@dataclass(frozen=True)
class DirectionDiagnostics:
    principal_axis_abs: tuple[float, float]
    axis_margin: float
    eigenvalue_ratio: float
    projection_bands: tuple[int, ...]
    projection_density: tuple[float, ...] = ()


@dataclass(frozen=True)
class Detection:
    label: str
    bbox: BBox | None = None
    color: str = "unknown"
    color_score: int = 0
    green_score: int = 0
    red_score: int = 0
    color_density: float = 0.0
    component_area: int = 0
    component_fill_density: float = 0.0
    orientation: str = "unknown"
    projection_peak: int | None = None
    diagnostics: DirectionDiagnostics | None = field(default=None, compare=False)


@dataclass(frozen=True)
class ShapeResult:
    label: str
    orientation: str
    projection_peak: int | None
    diagnostics: DirectionDiagnostics | None = field(default=None, compare=False)


@dataclass(frozen=True)
class Masks:
    roi: np.ndarray
    green: np.ndarray
    red: np.ndarray
    bright: np.ndarray
    roi_rect: BBox
    scale: float


@dataclass(frozen=True)
class Candidate:
    bbox: BBox
    component_mask: np.ndarray
    color: str
    color_score: int
    green_score: int
    red_score: int
    component_area: int
    component_fill_density: float
    color_density: float


@dataclass(frozen=True)
class CandidateSearchDiagnostics:
    """Counts showing at which filter stage bright components were rejected."""

    total_components: int = 0
    rejected_width: int = 0
    rejected_height: int = 0
    rejected_area: int = 0
    rejected_component_fill: int = 0
    rejected_color_tie: int = 0
    rejected_color_score: int = 0
    rejected_color_density: int = 0
    accepted_candidates: int = 0


@dataclass(frozen=True)
class FrameDiagnostics:
    """Compact per-frame values useful while tuning detector thresholds."""

    roi_rect: BBox
    scale: float
    green_pixels: int
    red_pixels: int
    bright_pixels: int
    search: CandidateSearchDiagnostics


def _scaled_length(value: int, scale: float) -> int:
    return max(1, int(round(value * scale)))


def _scaled_area(value: int, scale: float) -> int:
    return max(1, int(round(value * scale * scale)))


def _validate_image(image: np.ndarray) -> None:
    if not isinstance(image, np.ndarray) or image.size == 0:
        raise ValueError("image must be a non-empty numpy array")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must be a three-channel BGR image")
    if image.dtype != np.uint8:
        raise ValueError("image dtype must be uint8")


def _roi_rect(image: np.ndarray, config: Config) -> BBox:
    height, width = image.shape[:2]
    x1 = int(config.roi_x[0] * width)
    x2 = int(config.roi_x[1] * width)
    y1 = int(config.roi_y[0] * height)
    y2 = int(config.roi_y[1] * height)
    return x1, y1, x2, y2


def build_masks(
    image: np.ndarray, config: Config = DEFAULT_CONFIG
) -> Masks:
    _validate_image(image)
    height, width = image.shape[:2]
    reference_width, reference_height = config.reference_size
    scale = min(width / reference_width, height / reference_height)
    roi_rect = _roi_rect(image, config)
    x1, y1, x2, y2 = roi_rect

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    green_low = np.asarray(config.green_low, dtype=np.uint8)
    green_high = np.asarray(config.green_high, dtype=np.uint8)
    red_low_1 = np.asarray(config.red_low_1, dtype=np.uint8)
    red_high_1 = np.asarray(config.red_high_1, dtype=np.uint8)
    red_low_2 = np.asarray(config.red_low_2, dtype=np.uint8)
    red_high_2 = np.asarray(config.red_high_2, dtype=np.uint8)
    bright_low = np.asarray(config.bright_low, dtype=np.uint8)
    bright_high = np.asarray(config.bright_high, dtype=np.uint8)
    green = cv2.inRange(hsv, green_low, green_high)
    red_1 = cv2.inRange(hsv, red_low_1, red_high_1)
    red_2 = cv2.inRange(hsv, red_low_2, red_high_2)
    red = cv2.bitwise_or(red_1, red_2)
    bright = cv2.inRange(hsv, bright_low, bright_high)

    roi = np.zeros((height, width), dtype=np.uint8)
    roi[y1:y2, x1:x2] = 255
    green = cv2.bitwise_and(green, roi)
    red = cv2.bitwise_and(red, roi)
    bright = cv2.bitwise_and(bright, roi)

    kernel_size = max(1, int(config.close_kernel_size))
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
    )
    bright = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, kernel)

    return Masks(roi, green, red, bright, roi_rect, scale)


def find_candidate(
    masks: Masks, config: Config = DEFAULT_CONFIG
) -> Candidate | None:
    candidate, _ = _find_candidate_with_diagnostics(masks, config)
    return candidate


def _find_candidate_with_diagnostics(
    masks: Masks, config: Config = DEFAULT_CONFIG
) -> tuple[Candidate | None, CandidateSearchDiagnostics]:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        masks.bright, connectivity=8
    )
    image_height, image_width = masks.bright.shape
    min_width = _scaled_length(config.component_width[0], masks.scale)
    max_width = _scaled_length(config.component_width[1], masks.scale)
    min_height = _scaled_length(config.component_height[0], masks.scale)
    max_height = _scaled_length(config.component_height[1], masks.scale)
    min_area = _scaled_area(config.min_component_area, masks.scale)
    min_color_score = _scaled_area(config.min_color_score, masks.scale)
    padding = _scaled_length(config.candidate_padding, masks.scale)
    candidates = []
    rejected_width = 0
    rejected_height = 0
    rejected_area = 0
    rejected_component_fill = 0
    rejected_color_tie = 0
    rejected_color_score = 0
    rejected_color_density = 0

    for component_id in range(1, count):
        x, y, width, height, area = stats[component_id]
        if not (min_width <= width <= max_width):
            rejected_width += 1
            continue
        if not (min_height <= height <= max_height):
            rejected_height += 1
            continue
        if area < min_area:
            rejected_area += 1
            continue
        component_fill_density = float(area) / float(width * height)
        if component_fill_density < config.min_component_fill_density:
            rejected_component_fill += 1
            continue

        ex1 = max(0, x - padding)
        ey1 = max(0, y - padding)
        ex2 = min(image_width, x + width + padding)
        ey2 = min(image_height, y + height + padding)
        green_score = cv2.countNonZero(masks.green[ey1:ey2, ex1:ex2])
        red_score = cv2.countNonZero(masks.red[ey1:ey2, ex1:ex2])
        if green_score == red_score:
            rejected_color_tie += 1
            continue
        color = "green" if green_score > red_score else "red"
        color_score = max(green_score, red_score)
        expanded_area = (ex2 - ex1) * (ey2 - ey1)
        color_density = color_score / expanded_area
        if color_score < min_color_score:
            rejected_color_score += 1
            continue
        if color_density < config.min_color_density:
            rejected_color_density += 1
            continue

        component_mask = np.where(
            labels[y : y + height, x : x + width] == component_id,
            255,
            0,
        ).astype(np.uint8)
        candidates.append(
            Candidate(
                bbox=(int(x), int(y), int(width), int(height)),
                component_mask=component_mask,
                color=color,
                color_score=int(color_score),
                green_score=int(green_score),
                red_score=int(red_score),
                component_area=int(area),
                component_fill_density=component_fill_density,
                color_density=float(color_density),
            )
        )

    diagnostics = CandidateSearchDiagnostics(
        total_components=count - 1,
        rejected_width=rejected_width,
        rejected_height=rejected_height,
        rejected_area=rejected_area,
        rejected_component_fill=rejected_component_fill,
        rejected_color_tie=rejected_color_tie,
        rejected_color_score=rejected_color_score,
        rejected_color_density=rejected_color_density,
        accepted_candidates=len(candidates),
    )
    if not candidates:
        return None, diagnostics
    return (
        max(candidates, key=lambda item: (item.color_score, item.component_area)),
        diagnostics,
    )


def classify_green_shape(component_mask: np.ndarray) -> ShapeResult:
    if not isinstance(component_mask, np.ndarray):
        return ShapeResult("unknown", "unknown", None)
    if component_mask.ndim != 2 or component_mask.size == 0:
        return ShapeResult("unknown", "unknown", None)

    ys, xs = np.nonzero(component_mask)
    if xs.size < 2:
        return ShapeResult("unknown", "unknown", None)

    points_xy = np.column_stack((xs, ys)).astype(np.float64)
    covariance = np.cov(points_xy, rowvar=False)
    if covariance.shape != (2, 2) or not np.isfinite(covariance).all():
        return ShapeResult("unknown", "unknown", None)

    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    principal_axis = eigenvectors[:, int(np.argmax(eigenvalues))]
    vx, vy = principal_axis
    abs_vx = abs(float(vx))
    abs_vy = abs(float(vy))
    band_parts = np.array_split(component_mask, 8, axis=1)
    projection_bands = tuple(int(cv2.countNonZero(part)) for part in band_parts)
    projection_density = tuple(
        count / part.size if part.size else 0.0
        for count, part in zip(projection_bands, band_parts)
    )
    minor_eigenvalue = float(eigenvalues[0])
    major_eigenvalue = float(eigenvalues[-1])
    eigenvalue_ratio = (
        major_eigenvalue / minor_eigenvalue
        if minor_eigenvalue > np.finfo(np.float64).eps
        else float("inf")
    )
    diagnostics = DirectionDiagnostics(
        principal_axis_abs=(abs_vx, abs_vy),
        axis_margin=abs_vx - abs_vy,
        eigenvalue_ratio=eigenvalue_ratio,
        projection_bands=projection_bands,
        projection_density=projection_density,
    )

    if abs_vy >= abs_vx:
        return ShapeResult("straight", "vertical", None, diagnostics)

    density = np.asarray(projection_density)
    peak_indices = np.flatnonzero(density == density.max())
    if peak_indices.size != 1:
        return ShapeResult("unknown", "horizontal", None, diagnostics)
    peak = int(peak_indices[0])
    label = "left" if peak <= 3 else "right"
    return ShapeResult(label, "horizontal", peak, diagnostics)


def detect_traffic_light(
    image: np.ndarray, config: Config = DEFAULT_CONFIG
) -> Detection:
    detection, _ = analyze_traffic_light(image, config)
    return detection


def analyze_traffic_light(
    image: np.ndarray, config: Config = DEFAULT_CONFIG
) -> tuple[Detection, FrameDiagnostics]:
    """Detect one frame and return threshold-tuning diagnostics with it."""
    masks = build_masks(image, config)
    candidate, search = _find_candidate_with_diagnostics(masks, config)
    frame_diagnostics = FrameDiagnostics(
        roi_rect=masks.roi_rect,
        scale=masks.scale,
        green_pixels=int(cv2.countNonZero(masks.green)),
        red_pixels=int(cv2.countNonZero(masks.red)),
        bright_pixels=int(cv2.countNonZero(masks.bright)),
        search=search,
    )
    if candidate is None:
        return Detection(label="unknown"), frame_diagnostics
    if candidate.color == "red":
        return (
            Detection(
                label="stop",
                bbox=candidate.bbox,
                color="red",
                color_score=candidate.color_score,
                green_score=candidate.green_score,
                red_score=candidate.red_score,
                color_density=candidate.color_density,
                component_area=candidate.component_area,
                component_fill_density=candidate.component_fill_density,
            ),
            frame_diagnostics,
        )
    shape = classify_green_shape(candidate.component_mask)
    return (
        Detection(
            label=shape.label,
            bbox=candidate.bbox,
            color="green",
            color_score=candidate.color_score,
            green_score=candidate.green_score,
            red_score=candidate.red_score,
            color_density=candidate.color_density,
            component_area=candidate.component_area,
            component_fill_density=candidate.component_fill_density,
            orientation=shape.orientation,
            projection_peak=shape.projection_peak,
            diagnostics=shape.diagnostics,
        ),
        frame_diagnostics,
    )


def draw_detection(
    image: np.ndarray,
    detection: Detection,
) -> np.ndarray:
    _validate_image(image)
    canvas = image.copy()

    if detection.bbox is not None:
        x, y, width, height = detection.bbox
        box_color = (0, 0, 255) if detection.color == "red" else (0, 255, 0)
        cv2.rectangle(canvas, (x, y), (x + width, y + height), box_color, 2)
        text_origin = (x, max(18, y - 8))
    else:
        text_origin = (10, 24)

    cv2.putText(
        canvas,
        detection.label,
        text_origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return canvas


def draw_stream_status(
    image: np.ndarray, fps: float, detection_ms: float
) -> np.ndarray:
    """Draw performance data away from the candidate label and bounding box."""
    text = f"FPS {fps:.1f} | detect {detection_ms:.1f} ms"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    thickness = 1
    (text_width, text_height), _ = cv2.getTextSize(
        text, font, font_scale, thickness
    )
    origin = (max(8, image.shape[1] - text_width - 8), text_height + 8)
    cv2.putText(
        image,
        text,
        origin,
        font,
        font_scale,
        (0, 255, 255),
        thickness,
        cv2.LINE_AA,
    )
    return image


def format_detection_log(
    frame_id: int,
    detection: Detection,
    frame_diagnostics: FrameDiagnostics,
    fps: float,
    detection_ms: float,
) -> str:
    """Format one compact but complete threshold-tuning log record."""
    bbox_ratio = (
        "none"
        if detection.bbox is None or detection.bbox[3] <= 0
        else f"{detection.bbox[2] / detection.bbox[3]:.3f}"
    )
    direction = detection.diagnostics
    if direction is None:
        pca_abs = "none"
        axis_margin = "none"
        eigenvalue_ratio = "none"
        projection_bands = "()"
        projection_density = "()"
    else:
        pca_abs = (
            f"({direction.principal_axis_abs[0]:.3f},"
            f"{direction.principal_axis_abs[1]:.3f})"
        )
        axis_margin = f"{direction.axis_margin:+.3f}"
        eigenvalue_ratio = f"{direction.eigenvalue_ratio:.3f}"
        projection_bands = str(direction.projection_bands)
        projection_density = "(" + ",".join(
            f"{value:.3f}" for value in direction.projection_density
        ) + ")"

    search = frame_diagnostics.search
    rejected = (
        f"w:{search.rejected_width},h:{search.rejected_height},"
        f"area:{search.rejected_area},"
        f"fill:{search.rejected_component_fill},"
        f"tie:{search.rejected_color_tie},"
        f"score:{search.rejected_color_score},"
        f"density:{search.rejected_color_density}"
    )
    return (
        f"frame={frame_id} fps={fps:.1f} detect_ms={detection_ms:.2f} "
        f"label={detection.label} bbox={detection.bbox} "
        f"bbox_ratio={bbox_ratio} color={detection.color} "
        f"color_score={detection.color_score} "
        f"green_score={detection.green_score} red_score={detection.red_score} "
        f"color_density={detection.color_density:.3f} "
        f"component_area={detection.component_area} "
        f"component_fill={detection.component_fill_density:.3f} "
        f"axis={detection.orientation} peak={detection.projection_peak} "
        f"pca_abs={pca_abs} axis_margin={axis_margin} "
        f"eig_ratio={eigenvalue_ratio} bands={projection_bands} "
        f"band_density={projection_density} roi={frame_diagnostics.roi_rect} "
        f"scale={frame_diagnostics.scale:.3f} "
        f"mask_pixels=(green:{frame_diagnostics.green_pixels},"
        f"red:{frame_diagnostics.red_pixels},"
        f"bright:{frame_diagnostics.bright_pixels}) "
        f"components=(total:{search.total_components},"
        f"accepted:{search.accepted_candidates},rejected={{{rejected}}})"
    )


class FpsMeter:
    """Exponentially smoothed FPS meter for the camera loop."""

    def __init__(self, smoothing: float = 0.2) -> None:
        self._smoothing = smoothing
        self._previous_time: float | None = None
        self._fps = 0.0

    def update(self, now: float) -> float:
        if self._previous_time is not None:
            elapsed = now - self._previous_time
            if elapsed > 0.0:
                current_fps = 1.0 / elapsed
                if self._fps == 0.0:
                    self._fps = current_fps
                else:
                    self._fps += self._smoothing * (current_fps - self._fps)
        self._previous_time = now
        return self._fps


class RealtimeLogTracker:
    """Throttle regular logs and cap noisy classification-change logging."""

    def __init__(
        self, interval_seconds: float = 1.0, min_change_interval: float = 0.25
    ) -> None:
        if interval_seconds <= 0.0:
            raise ValueError("log interval must be greater than zero")
        self._interval_seconds = interval_seconds
        self._min_change_interval = min_change_interval
        self._last_state: tuple[str, str, str] | None = None
        self._last_log_time: float | None = None
        self._suppressed_changes = 0

    def update(
        self, detection: Detection, now: float
    ) -> tuple[str, int] | None:
        state = (detection.label, detection.color, detection.orientation)
        changed = state != self._last_state
        first = self._last_state is None
        self._last_state = state

        since_last_log = (
            float("inf")
            if self._last_log_time is None
            else now - self._last_log_time
        )
        if first:
            reason = "initial"
        elif changed and since_last_log >= self._min_change_interval:
            reason = "state_changed"
        elif since_last_log >= self._interval_seconds:
            reason = "periodic"
        else:
            if changed:
                self._suppressed_changes += 1
            return None

        suppressed_changes = self._suppressed_changes
        self._suppressed_changes = 0
        self._last_log_time = now
        return reason, suppressed_changes


def run_camera(
    camera_index: int = 0,
    width: int = 640,
    height: int = 480,
    config: Config = DEFAULT_CONFIG,
    log_interval: float = 1.0,
) -> int:
    """Read and undistort camera frames, then show live detector output."""
    # Keep camera_capture's executable demo side effects out of library imports.
    from camera_capture import CameraCapture

    tracker = RealtimeLogTracker(interval_seconds=log_interval)
    fps_meter = FpsMeter()
    frame_id = 0
    LOGGER.info(
        "camera_start index=%d requested_size=%dx%d undistort=required "
        "log_interval=%.2fs",
        camera_index,
        width,
        height,
        log_interval,
    )
    LOGGER.info("detector_config %s", config)

    camera = CameraCapture(camera_index, width, height)
    try:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        while True:
            frame = camera.get_picture()
            frame = cv2.flip(frame, 1)
            frame = camera.correct_img(frame)
            frame = cv2.resize(frame, (1280,960), interpolation=cv2.INTER_LINEAR)

            started = time.perf_counter()
            detection, diagnostics = analyze_traffic_light(frame, config)
            detection_ms = (time.perf_counter() - started) * 1000.0
            now = time.perf_counter()
            fps = fps_meter.update(now)
            frame_id += 1

            canvas = draw_detection(frame, detection)
            draw_stream_status(canvas, fps, detection_ms)
            cv2.imshow(WINDOW_NAME, canvas)

            log_record = format_detection_log(
                frame_id, detection, diagnostics, fps, detection_ms
            )
            log_decision = tracker.update(detection, now)
            if log_decision is not None:
                reason, suppressed_changes = log_decision
                LOGGER.info(
                    "%s suppressed_changes=%d %s",
                    reason,
                    suppressed_changes,
                    log_record,
                )
            elif LOGGER.isEnabledFor(logging.DEBUG):
                LOGGER.debug("per_frame %s", log_record)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                LOGGER.info("camera_stop reason=keyboard frame=%d", frame_id)
                break
    except KeyboardInterrupt:
        LOGGER.info("camera_stop reason=interrupt frame=%d", frame_id)
    finally:
        camera.close()
        cv2.destroyAllWindows()

    return 0


def _format_detection(path: Path, detection: Detection) -> str:
    return (
        f"{path.name}: label={detection.label} bbox={detection.bbox} "
        f"color={detection.color} score={detection.color_score} "
        f"area={detection.component_area} axis={detection.orientation} "
        f"peak={detection.projection_peak}"
    )


def process_images(image_paths: Sequence[Path], output_dir: Path) -> int:
    failures = 0
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        print(
            f"cannot create output directory {output_dir}: {error}",
            file=sys.stderr,
        )
        return 1

    for image_path in image_paths:
        image_path = Path(image_path)
        if not image_path.is_file():
            print(f"cannot read image: {image_path}", file=sys.stderr)
            failures += 1
            continue
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"cannot read image: {image_path}", file=sys.stderr)
            failures += 1
            continue

        detection = detect_traffic_light(image)
        print(_format_detection(image_path, detection))

        suffix = image_path.suffix or ".png"
        output_path = output_dir / f"{image_path.stem}_traditional{suffix}"
        if not cv2.imwrite(str(output_path), draw_detection(image, detection)):
            print(f"cannot write image: {output_path}", file=sys.stderr)
            failures += 1

    return 1 if failures else 0


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Traditional OpenCV traffic-light detector"
    )
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument(
        "--log-interval",
        type=float,
        default=1.0,
        help="seconds between periodic detailed logs (default: 1.0)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="also emit detailed per-frame logs",
    )
    parser.add_argument(
        "--batch-demo",
        action="store_true",
        help="process the four configured sample images instead of the camera",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_argument_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    if args.batch_demo:
        return process_images(SAMPLE_IMAGE_PATHS, OUTPUT_DIR)
    try:
        return run_camera(
            camera_index=args.camera_index,
            width=args.width,
            height=args.height,
            log_interval=args.log_interval,
        )
    except (RuntimeError, ValueError, cv2.error) as error:
        LOGGER.error("camera detector failed: %s", error)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
