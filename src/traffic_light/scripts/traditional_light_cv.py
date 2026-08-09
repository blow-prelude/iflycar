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
class Masks:
    roi: np.ndarray
    green: np.ndarray
    red: np.ndarray
    bright: np.ndarray
    roi_rect: BBox
    scale: float


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
