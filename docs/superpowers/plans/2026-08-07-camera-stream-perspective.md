# Camera Stream Perspective Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace single-image processing with a continuous camera loop while preserving the exact preprocessing coordinate system used to capture the calibration image.

**Architecture:** Reuse `CameraCapture` for acquisition and undistortion. A pure `prepare_camera_frame` helper applies the second horizontal flip and 320×240 resize, and a stream loop keeps one `GroundBoundaryTracker` alive across frames before displaying the source frame, bird view, and mask.

**Tech Stack:** Python 3, NumPy, OpenCV, existing `CameraCapture`, pytest

## Global Constraints

- Keep `REAL_WIDTH_M = 0.5`, `REAL_LENGTH_M = 0.75`, and the existing source points unchanged.
- Match the captured calibration pipeline: `get_picture()`, `correct_img()`, second horizontal flip, then resize to 320×240.
- Do not save frames, masks, or bird-view images.
- Exit on `q` or `Esc`, always release the camera, and close all OpenCV windows.

---

### Task 1: Camera stream processing

**Files:**
- Modify: `src/vision_line/scripts/dynamic_ground_perspective.py`
- Modify: `src/vision_line/tests/test_dynamic_ground_perspective.py`

**Interfaces:**
- Produces: `prepare_camera_frame(frame, camera, cv2_module) -> np.ndarray`
- Produces: `should_exit(key_code: int) -> bool`
- Produces: `run_camera_stream(camera, cv2_module) -> None`

- [ ] **Step 1: Write failing tests for correction/flip/resize order, `q`/Esc exit behavior, nonblocking display, and camera cleanup.**
- [ ] **Step 2: Run `python -m pytest src/vision_line/tests/test_dynamic_ground_perspective.py -v -p no:cacheprovider` and confirm failure because the stream helpers do not exist.**
- [ ] **Step 3: Add camera constants, the preprocessing helper, exit-key helper, stream loop, and a `main()` that owns `CameraCapture` with guaranteed cleanup.**
- [ ] **Step 4: Re-run the focused tests, `py_compile`, and static error checks; do not open a real camera during automated verification.**

