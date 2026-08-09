# Traditional Traffic Light OpenCV Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a standalone Python/OpenCV example that recognizes the four supplied traffic-light images as `right`, `straight`, `left`, and `stop` without YOLO, RKNN, or training data.

**Architecture:** A single importable script validates a BGR frame, converts the normalized ROI to per-frame pixel bounds, builds HSV masks, locates a low-saturation bright component surrounded by red or green pixels, and classifies green glyphs using PCA plus an eight-band column projection. The detector remains independent of file I/O for later camera-stream reuse; a no-argument demo entry point processes code-defined sample paths and writes annotations containing only the candidate box and final label.

**Tech Stack:** Python 3.9.23, OpenCV 4.10.0, NumPy 2.0.2, Python standard-library `dataclasses`, `pathlib`, `typing`, `unittest`, and `unittest.mock`.

## Global Constraints

- Use `D:\Anaconda\envs\opencv39\python.exe` for every test and demo command.
- Do not add Python packages or modify the existing YOLO/RKNN implementation.
- Keep the implementation independent of ROS, camera capture, C++ code, and model files.
- Treat input as a non-empty, three-channel BGR `uint8` image; reject other shapes or dtypes with `ValueError`.
- Return `unknown` when no candidate or no reliable direction exists; never guess a traffic instruction.
- Store the ROI only as normalized coordinates `x=[0.25, 0.78)` and `y=[0.28, 0.78)`; convert them from each frame's actual width and height inside detection.
- Use OpenCV HSV thresholds: green `H=35..100, S>=80, V>=100`; red `H=0..12` or `165..179, S>=80, V>=100`; bright core `S<=110, V>=220`.
- At 640×480, filter bright components to width `15..90`, height `12..90`, and area at least `80`; scale lengths by `scale=min(width/640, height/480)` and areas by `scale²`.
- Require neighborhood `color_score >= 200×scale²` and `color_score/expanded_area >= 0.10`.
- Do not parse command-line parameters; `main()` reads `SAMPLE_IMAGE_PATHS` and `OUTPUT_DIR` constants defined with project-relative paths.
- Never overwrite source images; annotated outputs use the source stem plus `_traditional` under `build/traditional_light_cv_results`.
- Draw only the bright-core candidate box and final label on output images. Do not draw or print the ROI. Print color score, component area, orientation, and projection peak only to the terminal.
- The reference specification is `docs/superpowers/specs/2026-08-09-traditional-traffic-light-opencv-design.md`.

---

## File Structure

- Create `src/traffic_light/scripts/traditional_light_cv.py`: configuration/result types, frame validation, normalized-ROI masks, candidate selection, shape classification, minimal annotation, and a fixed-path demo.
- Create `src/traffic_light/test/test_traditional_light_cv.py`: standard-library unit tests, four-image regression tests, rendering-call tests, and fixed-path batch tests.
- Do not modify `CMakeLists.txt`, `package.xml`, existing RKNN scripts, model files, or source images.

### Task 1: Validation, Configuration, and HSV Masks

**Files:**
- Create: `src/traffic_light/scripts/traditional_light_cv.py`
- Create: `src/traffic_light/test/test_traditional_light_cv.py`

**Interfaces:**
- Consumes: a BGR `np.ndarray` and an optional `Config`.
- Produces: `Config`, `Detection`, `Masks`, and `build_masks(image: np.ndarray, config: Config = DEFAULT_CONFIG) -> Masks` for later tasks.

- [ ] **Step 1: Write failing validation and mask tests**

Create `src/traffic_light/test/test_traditional_light_cv.py` with the module-path setup, shared image paths, and these first tests:

```python
import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

PACKAGE_DIR = Path(__file__).resolve().parents[1]
SCRIPT_DIR = PACKAGE_DIR / "scripts"
PICTURES_DIR = PACKAGE_DIR / "pictures"
sys.path.insert(0, str(SCRIPT_DIR))

from traditional_light_cv import Config, build_masks


class MaskAndValidationTests(unittest.TestCase):
    def test_build_masks_applies_hsv_rules_inside_roi_only(self):
        image = np.zeros((480, 640, 3), dtype=np.uint8)

        green_bgr = cv2.cvtColor(
            np.uint8([[[60, 200, 230]]]), cv2.COLOR_HSV2BGR
        )[0, 0]
        red_bgr = cv2.cvtColor(
            np.uint8([[[5, 200, 230]]]), cv2.COLOR_HSV2BGR
        )[0, 0]
        bright_bgr = cv2.cvtColor(
            np.uint8([[[90, 50, 240]]]), cv2.COLOR_HSV2BGR
        )[0, 0]

        image[200, 200] = green_bgr
        image[210, 210] = red_bgr
        image[220, 220] = bright_bgr
        image[20, 20] = green_bgr

        masks = build_masks(image, Config())

        self.assertEqual(255, int(masks.green[200, 200]))
        self.assertEqual(255, int(masks.red[210, 210]))
        self.assertEqual(255, int(masks.bright[220, 220]))
        self.assertEqual(0, int(masks.green[20, 20]))
        self.assertEqual((160, 134, 499, 374), masks.roi_rect)
        self.assertAlmostEqual(1.0, masks.scale)

    def test_build_masks_rejects_invalid_images(self):
        invalid_images = (
            None,
            np.array([], dtype=np.uint8),
            np.zeros((480, 640), dtype=np.uint8),
            np.zeros((480, 640, 3), dtype=np.float32),
        )

        for image in invalid_images:
            with self.subTest(shape=getattr(image, "shape", None)):
                with self.assertRaises(ValueError):
                    build_masks(image, Config())


if __name__ == "__main__":
    unittest.main()
```

The expected ROI endpoint uses Python's integer truncation: `int(0.78*640)=499` and `int(0.78*480)=374`.

- [ ] **Step 2: Run the tests and verify the module is missing**

Run:

```powershell
& 'D:\Anaconda\envs\opencv39\python.exe' 'src\traffic_light\test\test_traditional_light_cv.py' -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'traditional_light_cv'`.

- [ ] **Step 3: Implement the immutable types, validation, ROI, and masks**

Create `src/traffic_light/scripts/traditional_light_cv.py` with these definitions and behavior:

```python
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
```

Keep `sys`, `Path`, and `Sequence` for the fixed-path demo in Task 4; do not add argument parsing.

- [ ] **Step 4: Run the tests and verify they pass**

Run:

```powershell
& 'D:\Anaconda\envs\opencv39\python.exe' 'src\traffic_light\test\test_traditional_light_cv.py' -v
```

Expected: `Ran 2 tests` and `OK`.

- [ ] **Step 5: Commit the validated mask foundation**

```powershell
git add src/traffic_light/scripts/traditional_light_cv.py src/traffic_light/test/test_traditional_light_cv.py
git commit -m "feat: add traditional traffic light HSV masks"
```

### Task 2: Bright-Core Candidate Selection and Stop Detection

**Files:**
- Modify: `src/traffic_light/scripts/traditional_light_cv.py`
- Modify: `src/traffic_light/test/test_traditional_light_cv.py`

**Interfaces:**
- Consumes: `Masks` from `build_masks` and `Config`.
- Produces: `Candidate`, `find_candidate(masks: Masks, config: Config = DEFAULT_CONFIG) -> Optional[Candidate]`, and an initial `detect_traffic_light(image: np.ndarray, config: Config = DEFAULT_CONFIG) -> Detection` supporting `stop` and `unknown`.

- [ ] **Step 1: Add failing blank-image and red-stop tests**

Update the test import and append the candidate tests:

```python
from traditional_light_cv import Config, build_masks, detect_traffic_light


class CandidateDetectionTests(unittest.TestCase):
    def test_blank_image_returns_unknown(self):
        image = np.zeros((480, 640, 3), dtype=np.uint8)

        detection = detect_traffic_light(image)

        self.assertEqual("unknown", detection.label)
        self.assertIsNone(detection.bbox)
        self.assertEqual("unknown", detection.color)

    def test_red_sample_returns_stop(self):
        image = cv2.imread(str(PICTURES_DIR / "004_0001.jpg"))

        detection = detect_traffic_light(image)

        self.assertEqual("stop", detection.label)
        self.assertEqual("red", detection.color)
        self.assertIsNotNone(detection.bbox)
        self.assertGreaterEqual(detection.color_score, 200)

    def test_stop_candidate_bbox_is_inside_normalized_roi(self):
        image = cv2.imread(str(PICTURES_DIR / "004_0001.jpg"))

        detection = detect_traffic_light(image)

        x, y, width, height = detection.bbox
        roi_x1, roi_y1, roi_x2, roi_y2 = build_masks(image).roi_rect
        self.assertGreaterEqual(x, roi_x1)
        self.assertGreaterEqual(y, roi_y1)
        self.assertLessEqual(x + width, roi_x2)
        self.assertLessEqual(y + height, roi_y2)
```

- [ ] **Step 2: Run only the new tests and verify the missing API failure**

Run:

```powershell
& 'D:\Anaconda\envs\opencv39\python.exe' 'src\traffic_light\test\test_traditional_light_cv.py' CandidateDetectionTests -v
```

Expected: FAIL during import because `detect_traffic_light` is not defined.

- [ ] **Step 3: Implement scaled component filtering and color-neighborhood scoring**

Add the candidate data type and helpers below `Masks`:

```python
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
```

Implement `find_candidate` with exact coordinate and threshold rules:

```python
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
```

Implement the first detection pipeline. Green candidates deliberately remain `unknown` until Task 3:

```python
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
    return Detection(
        label="unknown",
        bbox=candidate.bbox,
        color="green",
        color_score=candidate.color_score,
        component_area=candidate.component_area,
    )
```

- [ ] **Step 4: Run all current tests**

Run:

```powershell
& 'D:\Anaconda\envs\opencv39\python.exe' 'src\traffic_light\test\test_traditional_light_cv.py' -v
```

Expected: `Ran 5 tests` and `OK`. Confirm the red sample's reported candidate is approximately `(354, 210, 41, 59)`; minor pixel-level variation is acceptable, so the test does not hard-code it.

- [ ] **Step 5: Commit candidate selection and stop detection**

```powershell
git add src/traffic_light/scripts/traditional_light_cv.py src/traffic_light/test/test_traditional_light_cv.py
git commit -m "feat: locate traditional traffic light candidates"
```

### Task 3: PCA and Projection Arrow Classification

**Files:**
- Modify: `src/traffic_light/scripts/traditional_light_cv.py`
- Modify: `src/traffic_light/test/test_traditional_light_cv.py`

**Interfaces:**
- Consumes: `Candidate.component_mask` from Task 2.
- Produces: `ShapeResult` and `classify_green_shape(component_mask: np.ndarray) -> ShapeResult`; extends `detect_traffic_light` to return all five labels.

- [ ] **Step 1: Add failing synthetic-shape and real-image regression tests**

Update the import and append:

```python
from traditional_light_cv import (
    Config,
    build_masks,
    classify_green_shape,
    detect_traffic_light,
)


def make_arrow_mask(direction):
    right = np.zeros((48, 64), dtype=np.uint8)
    cv2.rectangle(right, (8, 20), (40, 28), 255, thickness=-1)
    cv2.fillConvexPoly(
        right,
        np.array([(36, 7), (59, 24), (36, 41)], dtype=np.int32),
        255,
    )
    if direction == "right":
        return right
    if direction == "left":
        return np.fliplr(right).copy()
    if direction == "straight":
        return cv2.rotate(right, cv2.ROTATE_90_COUNTERCLOCKWISE)
    raise ValueError(direction)


class ArrowClassificationTests(unittest.TestCase):
    def test_synthetic_arrow_shapes_have_expected_directions(self):
        for expected in ("straight", "left", "right"):
            with self.subTest(expected=expected):
                result = classify_green_shape(make_arrow_mask(expected))
                self.assertEqual(expected, result.label)

    def test_four_supplied_images_have_expected_labels(self):
        expected_labels = {
            "02051.jpg": "right",
            "02052.jpg": "straight",
            "003_0030.jpg": "left",
            "004_0001.jpg": "stop",
        }

        for filename, expected in expected_labels.items():
            with self.subTest(filename=filename):
                image = cv2.imread(str(PICTURES_DIR / filename))
                self.assertIsNotNone(image)
                detection = detect_traffic_light(image)
                self.assertEqual(expected, detection.label)
                self.assertIsNotNone(detection.bbox)
                x, y, width, height = detection.bbox
                roi_x1, roi_y1, roi_x2, roi_y2 = build_masks(image).roi_rect
                self.assertGreaterEqual(x, roi_x1)
                self.assertGreaterEqual(y, roi_y1)
                self.assertLessEqual(x + width, roi_x2)
                self.assertLessEqual(y + height, roi_y2)
```

- [ ] **Step 2: Run the arrow tests and verify the missing classifier failure**

Run:

```powershell
& 'D:\Anaconda\envs\opencv39\python.exe' 'src\traffic_light\test\test_traditional_light_cv.py' ArrowClassificationTests -v
```

Expected: FAIL during import because `classify_green_shape` is not defined.

- [ ] **Step 3: Implement PCA orientation and unique eight-band peak classification**

Add the result type and classifier:

```python
@dataclass(frozen=True)
class ShapeResult:
    label: str
    orientation: str
    projection_peak: Optional[int]


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
```

Replace the green fallback at the end of `detect_traffic_light` with:

```python
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
```

- [ ] **Step 4: Run all tests and inspect diagnostic values**

Run:

```powershell
& 'D:\Anaconda\envs\opencv39\python.exe' 'src\traffic_light\test\test_traditional_light_cv.py' -v
```

Expected: `Ran 7 tests` and `OK`. During a temporary local diagnostic print, verify the real green images produce a vertical axis for `02052.jpg`, a left-half projection peak for `003_0030.jpg`, and a right-half peak for `02051.jpg`; remove the temporary print before committing.

- [ ] **Step 5: Commit arrow classification**

```powershell
git add src/traffic_light/scripts/traditional_light_cv.py src/traffic_light/test/test_traditional_light_cv.py
git commit -m "feat: classify traffic light arrow directions"
```

### Task 4: Minimal Annotation and Fixed-Path Demo

**Files:**
- Modify: `src/traffic_light/scripts/traditional_light_cv.py`
- Modify: `src/traffic_light/test/test_traditional_light_cv.py`

**Interfaces:**
- Consumes: `Detection` from `detect_traffic_light`.
- Produces: `draw_detection(image: np.ndarray, detection: Detection) -> np.ndarray`, `process_images(image_paths: Sequence[Path], output_dir: Path) -> int`, and the no-argument `main() -> int`.

- [ ] **Step 1: Add failing minimal-rendering and fixed-batch tests**

Add imports and the final test class:

```python
import io
import tempfile
from contextlib import redirect_stderr, redirect_stdout
from unittest import mock

import traditional_light_cv as detector

from traditional_light_cv import (
    Config,
    build_masks,
    classify_green_shape,
    detect_traffic_light,
    draw_detection,
    process_images,
)


class DrawingAndFixedDemoTests(unittest.TestCase):
    def test_draw_detection_only_draws_candidate_box_and_final_label(self):
        image = cv2.imread(str(PICTURES_DIR / "02051.jpg"))
        original = image.copy()
        detection = detect_traffic_light(image)

        with mock.patch.object(
            detector.cv2, "rectangle", wraps=cv2.rectangle
        ) as rectangle_mock, mock.patch.object(
            detector.cv2, "putText", wraps=cv2.putText
        ) as text_mock:
            annotated = draw_detection(image, detection)

        self.assertTrue(np.array_equal(original, image))
        self.assertFalse(np.array_equal(original, annotated))
        self.assertEqual(1, rectangle_mock.call_count)
        self.assertEqual(1, text_mock.call_count)
        self.assertEqual(detection.label, text_mock.call_args.args[1])

    def test_process_images_writes_four_annotations_and_terminal_metrics(self):
        with tempfile.TemporaryDirectory() as output_dir:
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                status = process_images(
                    detector.SAMPLE_IMAGE_PATHS, Path(output_dir)
                )

            self.assertEqual(0, status)
            terminal_output = stdout.getvalue()
            expected_labels = {
                "02051.jpg": "right",
                "02052.jpg": "straight",
                "003_0030.jpg": "left",
                "004_0001.jpg": "stop",
            }
            for name, label in expected_labels.items():
                self.assertIn(f"{name}: label={label}", terminal_output)
                output_name = f"{Path(name).stem}_traditional{Path(name).suffix}"
                output_path = Path(output_dir) / output_name
                self.assertTrue(output_path.is_file(), output_path)
                self.assertGreater(output_path.stat().st_size, 0)
            self.assertIn("score=", terminal_output)
            self.assertIn("area=", terminal_output)
            self.assertIn("axis=", terminal_output)
            self.assertIn("peak=", terminal_output)
            self.assertNotIn("roi", terminal_output.lower())

    def test_process_images_continues_after_bad_input_and_returns_nonzero(self):
        with tempfile.TemporaryDirectory() as output_dir:
            stderr = io.StringIO()
            with redirect_stderr(stderr):
                status = process_images(
                    (
                        PICTURES_DIR / "missing.jpg",
                        PICTURES_DIR / "004_0001.jpg",
                    ),
                    Path(output_dir),
                )

            self.assertEqual(1, status)
            self.assertIn("missing.jpg", stderr.getvalue())
            self.assertTrue(
                (Path(output_dir) / "004_0001_traditional.jpg").is_file()
            )

    def test_main_uses_code_defined_paths_without_parameters(self):
        with mock.patch.object(detector, "process_images", return_value=0) as batch:
            status = detector.main()

        self.assertEqual(0, status)
        batch.assert_called_once_with(
            detector.SAMPLE_IMAGE_PATHS, detector.OUTPUT_DIR
        )
```

- [ ] **Step 2: Run the new tests and verify the missing renderer failure**

Run:

```powershell
& 'D:\Anaconda\envs\opencv39\python.exe' 'src\traffic_light\test\test_traditional_light_cv.py' DrawingAndFixedDemoTests -v
```

Expected: FAIL during import because `draw_detection` is not defined.

- [ ] **Step 3: Implement annotation without mutating the input**

Add:

```python
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
```

- [ ] **Step 4: Implement project-relative constants, batch processing, and no-argument main**

Add these constants after `DEFAULT_CONFIG`. They are relative to the script location and therefore contain no machine-specific workspace path:

```python
PACKAGE_DIR = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PACKAGE_DIR.parents[1]
SAMPLE_IMAGE_PATHS = (
    PACKAGE_DIR / "pictures" / "02051.jpg",
    PACKAGE_DIR / "pictures" / "02052.jpg",
    PACKAGE_DIR / "pictures" / "003_0030.jpg",
    PACKAGE_DIR / "pictures" / "004_0001.jpg",
)
OUTPUT_DIR = WORKSPACE_ROOT / "build" / "traditional_light_cv_results"
```

Add the terminal formatter, batch function, and entry point. The formatted line deliberately omits ROI coordinates:

```python
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


def main() -> int:
    return process_images(SAMPLE_IMAGE_PATHS, OUTPUT_DIR)


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 5: Run all eleven tests**

Run:

```powershell
& 'D:\Anaconda\envs\opencv39\python.exe' 'src\traffic_light\test\test_traditional_light_cv.py' -v
```

Expected: `Ran 11 tests` and `OK`.

- [ ] **Step 6: Run the fixed-path demo and visually inspect all annotations**

Run:

```powershell
& 'D:\Anaconda\envs\opencv39\python.exe' `
  'src\traffic_light\scripts\traditional_light_cv.py'
```

Expected stdout labels, in code-defined order: `right`, `straight`, `left`, `stop`, with score/area/axis/peak metrics and no ROI coordinates. Open the four images under `build/traditional_light_cv_results` and verify each image contains exactly one bright-core candidate box and the final label only; no ROI boundary or diagnostic metrics are drawn.

- [ ] **Step 7: Commit the standalone example**

```powershell
git add src/traffic_light/scripts/traditional_light_cv.py src/traffic_light/test/test_traditional_light_cv.py
git commit -m "feat: add traditional traffic light OpenCV demo"
```

### Task 5: Final Verification and Scope Audit

**Files:**
- Verify: `src/traffic_light/scripts/traditional_light_cv.py`
- Verify: `src/traffic_light/test/test_traditional_light_cv.py`
- Verify unchanged: `src/traffic_light/scripts/judge_light.py`
- Verify unchanged: `src/traffic_light/scripts/judge_light_ros_correct.py`

**Interfaces:**
- Consumes: the completed single-frame API and fixed-path demo from Tasks 1–4.
- Produces: verification evidence only; no production-file changes are expected.

- [ ] **Step 1: Run the complete focused test suite from the workspace root**

```powershell
& 'D:\Anaconda\envs\opencv39\python.exe' 'src\traffic_light\test\test_traditional_light_cv.py' -v
```

Expected: all eleven tests pass with no warnings or tracebacks.

- [ ] **Step 2: Verify the no-argument demo, terminal diagnostics, and saved images**

```powershell
& 'D:\Anaconda\envs\opencv39\python.exe' `
  'src\traffic_light\scripts\traditional_light_cv.py'
```

Expected: exit code `0`; one line per code-defined sample image; labels exactly `right`, `straight`, `left`, and `stop`; score/area/axis/peak are printed; ROI coordinates are absent; four `_traditional.jpg` images exist under `build/traditional_light_cv_results` and visually contain only one candidate box plus the final label.

- [ ] **Step 3: Audit the final diff for accidental scope expansion**

```powershell
git status --short
git diff HEAD~4 -- src/traffic_light
```

Expected: only `src/traffic_light/scripts/traditional_light_cv.py` and `src/traffic_light/test/test_traditional_light_cv.py` differ across the implementation commits. Existing YOLO/RKNN files and source images remain unchanged. The generated `build/traditional_light_cv_results` directory must not be staged.

- [ ] **Step 4: Record the final verification state**

Run:

```powershell
git log -5 --oneline
git status --short
```

Expected: the four feature commits from Tasks 1–4 are present after the plan/spec commits, and the working tree contains no source changes. If an implementation adjustment was required after the last feature commit, rerun all tests and commit only the two planned source/test files with a precise fix message.
