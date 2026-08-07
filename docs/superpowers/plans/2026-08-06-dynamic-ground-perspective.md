# Dynamic Ground Perspective Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a standalone Python command that detects the blue map connected to the image bottom, masks field equipment, and applies the calibrated perspective transform without clipping the visible map.

**Architecture:** Keep the existing camera and navigation nodes unchanged. A new script owns blue-region detection, robust top-boundary fitting, temporal boundary smoothing, metric destination construction, expanded-canvas warping, and the command-line entry point; pure numerical helpers are separated so they can be tested without opening a camera or GUI.

**Tech Stack:** Python 3, NumPy, OpenCV, pytest

## Global Constraints

- Source calibration points are ordered TL, TR, BR, BL: `(114,146)`, `(206,146)`, `(271,184)`, `(35,187)`.
- The ground component must touch the bottom image band; unrelated blue field equipment must not be selected.
- On detection failure, reuse the preceding boundary; without history, fall back to `y=139`.
- Apply the homography to the whole masked ground and translate it to a canvas containing the transformed ground ROI.
- Do not modify existing ROS nodes or camera capture code.

---

### Task 1: Ground-boundary tracking

**Files:**
- Create: `src/vision_line/scripts/dynamic_ground_perspective.py`
- Test: `src/vision_line/tests/test_dynamic_ground_perspective.py`

**Interfaces:**
- Produces: `fit_top_boundary(component_mask, fallback_y) -> np.ndarray`
- Produces: `smooth_boundary(detected, previous, alpha, max_step) -> np.ndarray`
- Produces: `boundary_to_mask(boundary, height) -> np.ndarray`
- Produces: `GroundBoundaryTracker.detect(frame) -> tuple[np.ndarray, np.ndarray]`

- [ ] **Step 1: Write failing tests for a sloped bottom-connected region, fallback behavior, and bounded temporal smoothing.**

```python
def test_fit_top_boundary_recovers_sloped_edge():
    component = synthetic_component(width=80, height=60, intercept=18, slope=0.1)
    boundary = module.fit_top_boundary(component, fallback_y=25)
    np.testing.assert_allclose(boundary[[0, 79]], [18, 26], atol=1)

def test_smooth_boundary_limits_frame_to_frame_motion():
    previous = np.full(8, 20.0)
    detected = np.full(8, 40.0)
    actual = module.smooth_boundary(detected, previous, alpha=1.0, max_step=4.0)
    np.testing.assert_array_equal(actual, np.full(8, 24.0))
```

- [ ] **Step 2: Run `python -m pytest src/vision_line/tests/test_dynamic_ground_perspective.py -v` and verify failure because the new module does not exist.**
- [ ] **Step 3: Implement the numerical boundary helpers and OpenCV-backed bottom-connected blue component selection.**
- [ ] **Step 4: Re-run the focused test file and verify all boundary tests pass.**

### Task 2: Metric perspective transform and expanded canvas

**Files:**
- Modify: `src/vision_line/scripts/dynamic_ground_perspective.py`
- Modify: `src/vision_line/tests/test_dynamic_ground_perspective.py`

**Interfaces:**
- Consumes: the mask returned by `GroundBoundaryTracker.detect`.
- Produces: `metric_destination(width_m, length_m, pixels_per_m) -> np.ndarray`
- Produces: `expanded_homography(H, roi_points) -> tuple[np.ndarray, tuple[int, int]]`
- Produces: `warp_ground(frame, mask, src_points, width_m, length_m, pixels_per_m) -> np.ndarray`

- [ ] **Step 1: Write failing tests verifying destination dimensions, translated non-negative coordinates, and complete transformed ROI bounds.**
- [ ] **Step 2: Run the focused tests and verify the new tests fail for missing functions.**
- [ ] **Step 3: Implement metric destination construction, homogeneous point transformation, output-bound calculation, translation, and `cv2.warpPerspective`.**
- [ ] **Step 4: Run the focused tests and verify they pass.**

### Task 3: Standalone command and sample-image verification

**Files:**
- Modify: `src/vision_line/scripts/dynamic_ground_perspective.py`
- Modify: `src/vision_line/tests/test_dynamic_ground_perspective.py`

**Interfaces:**
- Consumes: `--input`, `--output`, `--real-width`, `--real-length`, `--scale`, and optional `--mask-output` arguments.
- Produces: a saved bird's-eye image and, when requested, a saved ground mask.

- [ ] **Step 1: Write failing argument-validation tests for positive real dimensions and pixels-per-meter.**
- [ ] **Step 2: Run the focused tests and verify the validation tests fail for missing behavior.**
- [ ] **Step 3: Implement the CLI, readable error messages, image loading checks, and output-directory creation.**
- [ ] **Step 4: Run the complete test file, compile the script with `python -m py_compile`, and process `src/vision_line/pictures/captured_image_20260720_164057.jpg` in the OpenCV runtime.**

