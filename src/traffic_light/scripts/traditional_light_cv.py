from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple

import cv2
import numpy as np

BBox = Tuple[int, int, int, int]


@dataclass(frozen=True)
class Config:
    roi_x: Tuple[float, float] = (0.25, 0.78)
    roi_y: Tuple[float, float] = (0.28, 0.78)
    reference_size: Tuple[int, int] = (640, 480)
    green_low: Tuple[int, int, int] = (35, 80, 100)
    green_high: Tuple[int, int, int] = (100, 255, 255)
    red_low_1: Tuple[int, int, int] = (0, 80, 100)
    red_high_1: Tuple[int, int, int] = (12, 255, 255)
    red_low_2: Tuple[int, int, int] = (165, 80, 100)
    red_high_2: Tuple[int, int, int] = (179, 255, 255)
    bright_low: Tuple[int, int, int] = (0, 0, 220)
    bright_high: Tuple[int, int, int] = (179, 110, 255)
    close_kernel_size: int = 3
    component_width: Tuple[int, int] = (15, 90)
    component_height: Tuple[int, int] = (12, 90)
    min_component_area: int = 80
    candidate_padding: int = 10
    min_color_score: int = 200
    min_color_density: float = 0.10


DEFAULT_CONFIG = Config()


@dataclass(frozen=True)
class Detection:
    label: str
    bbox: Optional[BBox] = None
    color: str = "unknown"
    color_score: int = 0
    component_area: int = 0
    orientation: str = "unknown"
    projection_peak: Optional[int] = None


@dataclass(frozen=True)
class ShapeResult:
    label: str
    orientation: str
    projection_peak: Optional[int]


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
    color_density: float


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
    green = cv2.inRange(hsv, config.green_low, config.green_high)
    red_1 = cv2.inRange(hsv, config.red_low_1, config.red_high_1)
    red_2 = cv2.inRange(hsv, config.red_low_2, config.red_high_2)
    red = cv2.bitwise_or(red_1, red_2)
    bright = cv2.inRange(hsv, config.bright_low, config.bright_high)

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
) -> Optional[Candidate]:
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

    for component_id in range(1, count):
        x, y, width, height, area = stats[component_id]
        if not (min_width <= width <= max_width):
            continue
        if not (min_height <= height <= max_height):
            continue
        if area < min_area:
            continue

        ex1 = max(0, x - padding)
        ey1 = max(0, y - padding)
        ex2 = min(image_width, x + width + padding)
        ey2 = min(image_height, y + height + padding)
        green_score = cv2.countNonZero(masks.green[ey1:ey2, ex1:ex2])
        red_score = cv2.countNonZero(masks.red[ey1:ey2, ex1:ex2])
        if green_score == red_score:
            continue
        color = "green" if green_score > red_score else "red"
        color_score = max(green_score, red_score)
        expanded_area = (ex2 - ex1) * (ey2 - ey1)
        color_density = color_score / expanded_area
        if color_score < min_color_score:
            continue
        if color_density < config.min_color_density:
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
                color_density=float(color_density),
            )
        )

    if not candidates:
        return None
    return max(candidates, key=lambda item: (item.color_score, item.component_area))


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
    if abs(vy) >= abs(vx):
        return ShapeResult("straight", "vertical", None)

    bands = np.array(
        [cv2.countNonZero(part) for part in np.array_split(component_mask, 8, axis=1)]
    )
    peak_indices = np.flatnonzero(bands == bands.max())
    if peak_indices.size != 1:
        return ShapeResult("unknown", "horizontal", None)
    peak = int(peak_indices[0])
    label = "left" if peak <= 3 else "right"
    return ShapeResult(label, "horizontal", peak)


def detect_traffic_light(
    image: np.ndarray, config: Config = DEFAULT_CONFIG
) -> Detection:
    masks = build_masks(image, config)
    candidate = find_candidate(masks, config)
    if candidate is None:
        return Detection(label="unknown")
    if candidate.color == "red":
        return Detection(
            label="stop",
            bbox=candidate.bbox,
            color="red",
            color_score=candidate.color_score,
            component_area=candidate.component_area,
        )
    shape = classify_green_shape(candidate.component_mask)
    return Detection(
        label=shape.label,
        bbox=candidate.bbox,
        color="green",
        color_score=candidate.color_score,
        component_area=candidate.component_area,
        orientation=shape.orientation,
        projection_peak=shape.projection_peak,
    )
