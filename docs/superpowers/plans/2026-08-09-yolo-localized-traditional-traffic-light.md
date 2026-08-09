# YOLO-Localized Traditional Traffic-Light Classifier Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Keep YOLO/RKNN as the traffic-light locator in `judge_light_ros_correct.py` while making directly embedded traditional OpenCV code the sole source of the published direction.

**Architecture:** The existing RKNN pipeline continues to produce letterboxed candidate boxes and scores. The highest-score box is mapped to the original BGR frame, then an in-file HSV/bright-core/PCA/eight-band classifier returns `stop`, `straight`, `left`, `right`, or `unknown`; only valid traditional results enter the existing publish throttle.

**Tech Stack:** Python 3.9, ROS `rospy`, OpenCV, NumPy, RKNNLite, `cv_bridge`, `std_msgs/String`

## Global Constraints

- Modify only `src/traffic_light/scripts/judge_light_ros_correct.py` during implementation.
- Do not import or call `traditional_light_cv.py` or `traditional_light_cv_ros.py`.
- Keep the existing YOLO/RKNN preprocessing, three inference workers, post-processing, ROS parameters, topics, cleanup, publish throttle, and exit policy.
- YOLO/RKNN locates a candidate only; its class must never determine or replace the published direction.
- A traditional result of `unknown` skips the frame and does not increment the valid-result publish counter.
- Per user instruction, do not add or run automated tests. Verification is limited to Python syntax compilation, focused source inspection, and `git diff --check`.

---

### Task 1: Embed the YOLO-region traditional OpenCV classifier

**Files:**
- Modify: `src/traffic_light/scripts/judge_light_ros_correct.py:1-330`

**Interfaces:**
- Consumes: original `uint8` BGR frame plus an original-image `xyxy` box returned by `COCO_test_helper.get_real_box`.
- Produces: `detect_traffic_light_in_roi(image: np.ndarray, roi_xyxy, config: Config = DEFAULT_CONFIG) -> Detection`.
- Produces: `draw_detection(image: np.ndarray, detection: Detection) -> np.ndarray` for the existing display loop.

- [ ] **Step 1: Add local result/configuration types and exact traditional thresholds**

Insert `from __future__ import annotations` immediately after the shebang, add `dataclass`/`field` imports, and define local `BBox`, `Config`, `DirectionDiagnostics`, `Detection`, `ShapeResult`, `Masks`, and `Candidate` types. Use these exact configuration defaults:

```python
@dataclass(frozen=True)
class Config:
    reference_size: tuple[int, int] = (640, 480)
    green_low: tuple[int, int, int] = (35, 80, 100)
    green_high: tuple[int, int, int] = (100, 255, 255)
    red_low_1: tuple[int, int, int] = (0, 80, 100)
    red_high_1: tuple[int, int, int] = (12, 255, 255)
    red_low_2: tuple[int, int, int] = (165, 80, 100)
    red_high_2: tuple[int, int, int] = (179, 255, 255)
    bright_low: tuple[int, int, int] = (0, 0, 220)
    bright_high: tuple[int, int, int] = (179, 110, 255)
    close_kernel_size: int = 3
    component_width: tuple[int, int] = (15, 90)
    component_height: tuple[int, int] = (12, 90)
    min_component_area: int = 80
    roi_padding: int = 10
    candidate_padding: int = 10
    min_color_score: int = 200
    min_color_density: float = 0.10


DEFAULT_CONFIG = Config()
```

`Detection` must default to `label="unknown"`, `bbox=None`, `color="unknown"`, zero scores/area, `orientation="unknown"`, no projection peak, and optional diagnostics. Store candidate boxes as `(x, y, width, height)` in original-image coordinates.

- [ ] **Step 2: Add image validation, source-scale calculation, and safe YOLO-region normalization**

Add `_scaled_length`, `_scaled_area`, `_validate_image`, and `_normalize_roi`. `_normalize_roi` must flatten the supplied box, require four finite coordinates, expand it by `roi_padding * scale`, clamp it to `[0, width] × [0, height]`, and return `None` when `x2 <= x1` or `y2 <= y1`:

```python
def _normalize_roi(image, roi_xyxy, config, scale):
    values = np.asarray(roi_xyxy, dtype=np.float64).reshape(-1)
    if values.size < 4 or not np.isfinite(values[:4]).all():
        return None
    pad = _scaled_length(config.roi_padding, scale)
    height, width = image.shape[:2]
    x1 = max(0, int(np.floor(values[0])) - pad)
    y1 = max(0, int(np.floor(values[1])) - pad)
    x2 = min(width, int(np.ceil(values[2])) + pad)
    y2 = min(height, int(np.ceil(values[3])) + pad)
    return None if x2 <= x1 or y2 <= y1 else (x1, y1, x2, y2)
```

Scale must always come from the full source frame:

```python
reference_width, reference_height = config.reference_size
scale = min(image.shape[1] / reference_width, image.shape[0] / reference_height)
```

- [ ] **Step 3: Add HSV masks and bright-core candidate selection inside the normalized YOLO region**

Implement `build_masks(image, roi_rect, scale, config)` and `find_candidate(masks, config)`. Generate green/red/bright masks from the full original BGR image, mask only the bright image by `roi_rect`, and close the bright mask with the configured 3×3 ellipse. Keep the full red and green masks available so candidate-neighborhood scoring can include the glow around a tight locator box.

Filter connected components using full-frame-scaled width, height, area, color-score and density thresholds. Expand each component by `candidate_padding * scale`; determine red versus green by the greater nonzero count; reject equal counts, low score and low density; choose the candidate maximizing `(color_score, component_area)`.

```python
return max(candidates, key=lambda item: (item.color_score, item.component_area))
```

- [ ] **Step 4: Add the exact PCA and normalized eight-band direction classifier**

Implement `classify_green_shape(component_mask)`. Use the largest covariance eigenvector to distinguish vertical from horizontal. Vertical returns `straight`. Horizontal uses `np.array_split(component_mask, 8, axis=1)`, calculates nonzero density per band, requires a unique maximum, then maps peak `0..3` to `left` and `4..7` to `right`:

```python
band_parts = np.array_split(component_mask, 8, axis=1)
projection_bands = tuple(int(cv2.countNonZero(part)) for part in band_parts)
projection_density = tuple(
    count / part.size if part.size else 0.0
    for count, part in zip(projection_bands, band_parts)
)
density = np.asarray(projection_density)
peak_indices = np.flatnonzero(density == density.max())
if peak_indices.size != 1:
    return ShapeResult("unknown", "horizontal", None, diagnostics)
peak = int(peak_indices[0])
label = "left" if peak <= 3 else "right"
```

Preserve the traditional diagnostics fields: absolute principal axis, axis margin, eigenvalue ratio, raw band counts, and band densities.

- [ ] **Step 5: Add the pure entry point and display annotation**

Implement `detect_traffic_light_in_roi`. Validate the image, derive full-frame scale, normalize the ROI, return `Detection()` for an invalid region or missing candidate, map a red candidate to `stop`, and classify a green candidate through PCA:

```python
def detect_traffic_light_in_roi(image, roi_xyxy, config=DEFAULT_CONFIG):
    _validate_image(image)
    reference_width, reference_height = config.reference_size
    scale = min(image.shape[1] / reference_width, image.shape[0] / reference_height)
    roi_rect = _normalize_roi(image, roi_xyxy, config, scale)
    if roi_rect is None:
        return Detection()
    masks = build_masks(image, roi_rect, scale, config)
    candidate = find_candidate(masks, config)
    if candidate is None:
        return Detection()
    if candidate.color == "red":
        return Detection(
            label="stop", bbox=candidate.bbox, color="red",
            color_score=candidate.color_score,
            component_area=candidate.component_area,
        )
    shape = classify_green_shape(candidate.component_mask)
    return Detection(
        label=shape.label, bbox=candidate.bbox, color="green",
        color_score=candidate.color_score,
        component_area=candidate.component_area,
        orientation=shape.orientation,
        projection_peak=shape.projection_peak,
        diagnostics=shape.diagnostics,
    )
```

Add `draw_detection` that copies the frame, draws only the traditional bright-core box and final label, and never draws a YOLO class name.

- [ ] **Step 6: Perform the user-approved non-test verification for the embedded code**

Run a Python 3.9 syntax-only compile without importing ROS/RKNN and without writing bytecode:

```powershell
& 'D:\Anaconda\envs\opencv39\python.exe' -c "from pathlib import Path; p=Path(r'src\traffic_light\scripts\judge_light_ros_correct.py'); compile(p.read_text(encoding='utf-8'), str(p), 'exec')"
```

Expected: exit code `0`, no output.

Run:

```powershell
git diff --check -- src/traffic_light/scripts/judge_light_ros_correct.py
```

Expected: exit code `0`.

- [ ] **Step 7: Commit the embedded classifier**

```powershell
git add src/traffic_light/scripts/judge_light_ros_correct.py
git commit -m "feat: embed traditional traffic light classifier"
```

### Task 2: Make traditional vision the sole published classification

**Files:**
- Modify: `src/traffic_light/scripts/judge_light_ros_correct.py:450-540`

**Interfaces:**
- Consumes: `detect_traffic_light_in_roi`, `draw_detection`, the existing highest-score YOLO box, original BGR frame, coordinate helper, ROS publisher, and valid-result counter.
- Produces: `/vision_line_direction` messages whose data is exclusively `Detection.label` for valid traditional detections.

- [ ] **Step 1: Remove obsolete YOLO-class CV verification and YOLO-class drawing helpers**

Delete `check_arrow_direction` and the old `draw(image, boxes, scores, classes)` function. Keep `CLASSES` because YOLO post-processing still uses class indices internally, but do not use `CLASSES[best_class]` to build `direction_name`.

- [ ] **Step 2: Replace the best-class decision with best-box traditional classification**

Retain highest-score box selection but stop calculating or logging `best_class`. Initialize `detection = Detection()` each result cycle. When `best_idx` exists and `best_score >= OBJ_THRESH`, map only that box back to original coordinates and classify it:

```python
real_box = co_helper.get_real_box(
    np.asarray(boxes[best_idx], dtype=np.float32).reshape(1, -1)
)[0]
detection = detect_traffic_light_in_roi(img_src, real_box)
direction_name = detection.label
```

Log YOLO worker, best score and inference time separately from traditional label/bbox/color/score/area/orientation/peak diagnostics.

- [ ] **Step 3: Gate the existing publish throttle solely on the traditional label**

Publish only when `direction_name != "unknown"`. Preserve the existing `pub_skip_n = 10` behavior, incrementing `pub_counter` only for valid traditional detections. Preserve the current one-second wait and loop exit after publishing a non-`stop` label:

```python
if direction_name != "unknown":
    if pub_counter % pub_skip_n != 0:
        pub_counter += 1
    else:
        pub_counter += 1
        direction_pub.publish(String(direction_name))
        rospy.loginfo(f"Detected direction: {direction_name}")
        if direction_name != "stop":
            time.sleep(1.0)
            break
else:
    rospy.loginfo_throttle(1.0, "Traditional classifier: unknown, skip frame")
```

There must be no YOLO fallback path. An `unknown` result neither publishes nor increments `pub_counter`.

- [ ] **Step 4: Draw only the traditional result and retain FPS**

Replace the old YOLO rendering call with:

```python
canvas = draw_detection(img_src, detection)
```

Keep `img_processor.draw_fps(canvas)`, `cv2.imshow`, `cv2.waitKey`, the queue-empty display path and all cleanup behavior.

- [ ] **Step 5: Perform final non-test source and syntax verification**

Run the syntax-only compile command from Task 1 Step 6 and `git diff --check` again. Then run these focused source inspections:

```powershell
rg -n "detect_traffic_light_in_roi|direction_name = detection.label|Traditional classifier" src/traffic_light/scripts/judge_light_ros_correct.py
rg -n "from traditional_light_cv|import traditional_light_cv|direction_name = CLASSES|check_arrow_direction" src/traffic_light/scripts/judge_light_ros_correct.py
```

Expected: the first command finds the new integration; the second command produces no matches. Do not run unit, integration, ROS, camera, or RKNN tests.

- [ ] **Step 6: Commit the main-loop integration**

```powershell
git add src/traffic_light/scripts/judge_light_ros_correct.py
git commit -m "feat: classify YOLO traffic light boxes with OpenCV"
```

- [ ] **Step 7: Record target-device follow-up without executing it locally**

In the handoff, state that runtime ROS/RKNN behavior was not tested per user request. Provide the unchanged target command:

```bash
rosrun traffic_light judge_light_ros_correct.py
```

The operator should observe that `unknown` frames produce no `/vision_line_direction` message, `stop` keeps the node running, and `left`/`right`/`straight` exit after their throttled publish.
