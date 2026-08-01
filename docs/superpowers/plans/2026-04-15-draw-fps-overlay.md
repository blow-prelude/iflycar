# Draw FPS Overlay on Output Frame Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an optional FPS overlay to `draw_line()` and compute FPS in `main()`, removing the existing `frame_count` progress logic.

**Architecture:** Compute an instant FPS in the video loop using `time.time()` deltas, store it for `draw_line()` to read, and overlay text via `cv2.putText` at the top-left of the drawn frame.

**Tech Stack:** Python 3, OpenCV (`cv2`), NumPy

---

## File map / responsibilities

- Modify: `src/vision_line/scripts/image_process.py`
  - `ImageProcess.__init__(...)`: add `self.fps = None` (float or None).
  - `ImageProcess.draw_line(canvas, draw_fps=True)`: add new param `draw_fps: bool = True`; when `draw_fps` is true and `self.fps` is not None, draw `FPS: xx.x` at top-left.
  - `main()`: remove `frame_count` increment + periodic logging; compute FPS each processed frame and store into `imgprocess.fps`.

> Note: This repo currently has no unit test harness visible in this file. This change is UI-ish (overlay text), so verification will be manual by running `main()`.

---

### Task 1: Update `draw_line()` signature and overlay FPS

**Files:**
- Modify: `src/vision_line/scripts/image_process.py:32-70` (add `self.fps`)
- Modify: `src/vision_line/scripts/image_process.py:825-870` (`draw_line`)

- [ ] **Step 1: Add an FPS field to `ImageProcess`**

In `ImageProcess.__init__`, add:

```python
self.fps = None  # float FPS computed in main(), or None
```

- [ ] **Step 2: Change `draw_line()` signature**
  - Update:
    - from: `def draw_line(self, canvas):`
    - to: `def draw_line(self, canvas, draw_fps=True):`

- [ ] **Step 3: Add FPS overlay implementation (minimal)**

Add near the end of `draw_line()` (after drawing lines, before `return canvas`):

```python
if draw_fps and self.fps is not None:
    cv2.putText(
        canvas,
        f"FPS: {self.fps:.1f}",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
```

- [ ] **Step 4: Quick manual sanity check**
  - Ensure existing callers that do `draw_line(canvas)` still work.

- [ ] **Step 5: Commit**

```bash
git add src/vision_line/scripts/image_process.py
git commit -m "$(cat <<'EOF'
feat: optionally draw FPS overlay

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: Compute FPS in `main()` and remove `frame_count` logic

**Files:**
- Modify: `src/vision_line/scripts/image_process.py:1086-1250`

- [ ] **Step 1: Remove `frame_count` tracking and progress logs**
  - Delete:
    - `frame_count = 0` (currently around `image_process.py:1106`)
    - `frame_count += 1` (currently around `image_process.py:1126`)
    - the `if frame_count % 30 == 0: ...` progress logging block (currently around `image_process.py:1127-1130`)
  - Update/remove all logs that reference `frame_count` to avoid `NameError`, including:
    - preprocess failure warning (currently around `image_process.py:1162`)
    - quit log (currently around `image_process.py:1243`)
    - pause/resume log (currently around `image_process.py:1247-1249`)
    - KeyboardInterrupt log (currently around `image_process.py:1252`)

- [ ] **Step 2: Add FPS tracking variables**

Add before the loop:

```python
prev_t = None
```

(Keep it instant FPS; no smoothing unless requested.)

- [ ] **Step 3: Compute FPS each time a new frame is read**

Right after successful `cap.read()`:

```python
now = time.time()
if prev_t is not None:
    dt = now - prev_t
    if dt > 1e-6:
        imgprocess.fps = 1.0 / dt
prev_t = now
```

Pause/resume edge case: when toggling `paused = not paused`, set `prev_t = None` so the first FPS after resuming doesn’t dip to near-zero.

- [ ] **Step 4: Call `draw_line()` with `draw_fps=True`**

Change:

```python
canvas = imgprocess.draw_line(canvas)
```

to:

```python
canvas = imgprocess.draw_line(canvas, draw_fps=True)
```

- [ ] **Step 5: Manual verification run**

Run the script and confirm FPS text appears in the top-left and updates in real time:

```bash
python src/vision_line/scripts/image_process.py
```

Note: `main()` currently hardcodes `video_path = D:\\programs\\ucar_ws\\src\\vision_line\\videos\\test2.avi`. Ensure that file exists, or adjust the path before running.

Expected: Two windows (`binary`, `processed_img`) and top-left yellow `FPS: xx.x` overlay on `processed_img`.

- [ ] **Step 6: Commit**

```bash
git add src/vision_line/scripts/image_process.py
git commit -m "$(cat <<'EOF'
refactor: compute FPS in main and drop frame_count logs

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
EOF
)"
```
