# STRAIGHT_TRACKING Fixed Perspective Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `find_way` apply the calibrated fixed ground mask and perspective transform after resize only while it is in `STRAIGHT_TRACKING`.

**Architecture:** Add one fixed-runtime transform API to the existing `ground_perspective` library. The API validates the calibrated 320×240 BGR input, masks rows above 139, and warps with a literal 3×3 matrix to 484×299; `find_way` selects this result only for straight tracking and otherwise preserves its current frame path.

**Tech Stack:** C++11, OpenCV, ROS catkin/CMake, GoogleTest, WSL2 Ubuntu 20.04.

## Global Constraints

- The delivered edits live in `D:\programs\ucar_ws`; compile and test an identical mirror in `/home/wtr/program/iflycar` under the `Ubuntu-20.04` WSL2 distribution.
- Preserve the ground-perspective demo and validation changes already recorded by commit `5de5ec3`; do not modify `src/vision_line/src/dynamic_ground_perspective.cpp`.
- Do not modify or discard the existing WSL-only uncommitted change in `src/vision_line/scripts/picture_cli.py`.
- Fixed input is exactly a non-empty 320×240 `CV_8UC3` BGR image.
- Rows with `y < 139` are cleared before warping.
- Fixed output is exactly 484×299.
- Do not calculate the homography from calibration parameters at runtime.
- Do not remove or change `CameraCapture::perspectiveFrame` in this task.

---

## File Map

- `src/vision_line/include/ground_perspective.h`: declares the fixed runtime warp interface.
- `src/vision_line/src/ground_perspective.cpp`: owns the fixed calibration constants, validation, mask, and warp implementation.
- `src/vision_line/test/test_ground_perspective.cpp`: verifies fixed geometry, masking, and invalid-input behavior while retaining the tests from commit `5de5ec3`.
- `src/vision_line/src/find_way.cpp`: selects the fixed warp only for `STRAIGHT_TRACKING` and leaves downstream processing unchanged.
- `src/vision_line/CMakeLists.txt`: links `find_way` to the existing `ground_perspective` target while retaining the compile-option changes from commit `5de5ec3`.

### Task 1: Fixed runtime ground perspective API

**Files:**

- Modify: `src/vision_line/include/ground_perspective.h:29`
- Modify: `src/vision_line/src/ground_perspective.cpp:10-152`
- Modify: `src/vision_line/test/test_ground_perspective.cpp` after the existing tests

**Interfaces:**

- Consumes: `const cv::Mat &frame`, exactly 320×240 and `CV_8UC3`.
- Produces: `cv::Mat vision_line::warpFixedGroundPerspective(const cv::Mat &frame)`, exactly 484×299 and `CV_8UC3`.

- [ ] **Step 1: Add focused failing tests**

Append these tests without changing or deleting the validation tests from commit `5de5ec3`:

```cpp
TEST(FixedGroundPerspective, UsesCalibratedOutputGeometry)
{
    const cv::Mat frame(240, 320, CV_8UC3, cv::Scalar(10, 20, 30));

    const cv::Mat warped = vision_line::warpFixedGroundPerspective(frame);

    EXPECT_EQ(cv::Size(484, 299), warped.size());
    EXPECT_EQ(CV_8UC3, warped.type());
}

TEST(FixedGroundPerspective, RemovesPixelsAboveGroundStartBeforeWarp)
{
    cv::Mat frame = cv::Mat::zeros(240, 320, CV_8UC3);
    frame(cv::Rect(0, 0, 320, 139)).setTo(cv::Scalar(10, 20, 30));

    const cv::Mat warped = vision_line::warpFixedGroundPerspective(frame);

    EXPECT_EQ(0, cv::countNonZero(warped.reshape(1)));
}

TEST(FixedGroundPerspective, RejectsEmptyFrame)
{
    EXPECT_THROW(vision_line::warpFixedGroundPerspective(cv::Mat()),
                 std::invalid_argument);
}

TEST(FixedGroundPerspective, RejectsNonBgr8BitFrame)
{
    const cv::Mat gray(240, 320, CV_8UC1);
    EXPECT_THROW(vision_line::warpFixedGroundPerspective(gray),
                 std::invalid_argument);
}

TEST(FixedGroundPerspective, RejectsUncalibratedFrameSize)
{
    const cv::Mat wrong_size(239, 320, CV_8UC3);
    EXPECT_THROW(vision_line::warpFixedGroundPerspective(wrong_size),
                 std::invalid_argument);
}
```

- [ ] **Step 2: Mirror the test-only edit to WSL2 and verify RED**

First confirm the WSL target test file has no independent edits, then mirror the Windows test file without touching `picture_cli.py`:

```powershell
wsl.exe -d Ubuntu-20.04 -- bash -lc "cd /home/wtr/program/iflycar && git status --short src/vision_line/test/test_ground_perspective.cpp"
wsl.exe -d Ubuntu-20.04 -- bash -lc "cp /mnt/d/programs/ucar_ws/src/vision_line/test/test_ground_perspective.cpp /home/wtr/program/iflycar/src/vision_line/test/test_ground_perspective.cpp"
wsl.exe -d Ubuntu-20.04 -- bash -lc "source /opt/ros/noetic/setup.bash && cd /home/wtr/program/iflycar && catkin_make run_tests_vision_line_gtest_test_ground_perspective"
```

Expected: compilation fails because `vision_line::warpFixedGroundPerspective` has not been declared. This is the required RED result, not an environment or syntax error.

- [ ] **Step 3: Declare the fixed API**

Add this declaration after `warpGround` in `ground_perspective.h`:

```cpp
cv::Mat warpFixedGroundPerspective(const cv::Mat &frame);
```

- [ ] **Step 4: Implement the minimal fixed transform**

Add the fixed constants to the anonymous namespace before `namespace vision_line` in `ground_perspective.cpp`:

```cpp
namespace
{
const cv::Size kFixedInputSize(320, 240);
const cv::Size kFixedOutputSize(484, 299);
const int kFixedGroundStartY = 139;
const cv::Matx33d kFixedPerspectiveMatrix(
    -0.214974396135, -1.920230497309, 266.769032700576,
     0.006976740047, -2.918053000379, 403.289239923271,
     0.000068399412, -0.008376745778, 1.0);
} // namespace
```

Add the implementation after `warpGround`:

```cpp
cv::Mat warpFixedGroundPerspective(const cv::Mat &frame)
{
    if (frame.empty() || frame.type() != CV_8UC3)
    {
        throw std::invalid_argument(
            "fixed ground perspective requires a non-empty CV_8UC3 frame");
    }
    if (frame.size() != kFixedInputSize)
    {
        throw std::invalid_argument(
            "fixed ground perspective requires a 320x240 frame");
    }

    const cv::Mat ground_mask =
        makeFixedGroundMask(frame.size(), kFixedGroundStartY);
    cv::Mat masked_frame;
    cv::bitwise_and(frame, frame, masked_frame, ground_mask);

    cv::Mat warped;
    cv::warpPerspective(masked_frame, warped,
                        cv::Mat(kFixedPerspectiveMatrix),
                        kFixedOutputSize, cv::INTER_LINEAR);
    return warped;
}
```

- [ ] **Step 5: Mirror the implementation and verify GREEN**

```powershell
wsl.exe -d Ubuntu-20.04 -- bash -lc "cp /mnt/d/programs/ucar_ws/src/vision_line/include/ground_perspective.h /home/wtr/program/iflycar/src/vision_line/include/ground_perspective.h"
wsl.exe -d Ubuntu-20.04 -- bash -lc "cp /mnt/d/programs/ucar_ws/src/vision_line/src/ground_perspective.cpp /home/wtr/program/iflycar/src/vision_line/src/ground_perspective.cpp"
wsl.exe -d Ubuntu-20.04 -- bash -lc "source /opt/ros/noetic/setup.bash && cd /home/wtr/program/iflycar && catkin_make run_tests_vision_line_gtest_test_ground_perspective && catkin_test_results --verbose"
```

Expected: all `test_ground_perspective` cases pass with zero failures, including the five new fixed-transform cases.

- [ ] **Step 6: Review and commit Task 1**

```powershell
git diff --check -- src/vision_line/include/ground_perspective.h src/vision_line/src/ground_perspective.cpp src/vision_line/test/test_ground_perspective.cpp
git diff -- src/vision_line/include/ground_perspective.h src/vision_line/src/ground_perspective.cpp src/vision_line/test/test_ground_perspective.cpp
git add src/vision_line/include/ground_perspective.h src/vision_line/src/ground_perspective.cpp src/vision_line/test/test_ground_perspective.cpp
git commit -m "feat: add fixed ground perspective transform"
```

Confirm that the tests from commit `5de5ec3` are preserved and only Task 1 files enter the task commit.

### Task 2: Wire the transform into STRAIGHT_TRACKING

**Files:**

- Modify: `src/vision_line/src/find_way.cpp:1-195`
- Modify: `src/vision_line/CMakeLists.txt:252-258`

**Interfaces:**

- Consumes: the 320×240 frame produced by `ImageProcess::resize_frame` and the existing `ProcessState` value.
- Produces: a 484×299 fixed ground view for straight tracking; the resize result itself for left/right turning.

- [ ] **Step 1: Replace the old straight-state transform call**

Add the include near the existing project headers:

```cpp
#include "ground_perspective.h"
```

Replace the current `pers_frame` selection block with:

```cpp
cv::Mat process_frame = frame;
if (state == ProcessState::STRAIGHT_TRACKING)
{
    process_frame = vision_line::warpFixedGroundPerspective(frame);
}

img_process.set_frame(process_frame);
cv::Mat binary_img = img_process.preprocess(process_frame);
```

Remove the old `CameraCapture::perspectiveFrame` call, its original-frame fallback, and the redundant empty check. Leave all logic after `preprocess` unchanged apart from renaming `pers_frame` to `process_frame` in the perspective preview.

- [ ] **Step 2: Link `find_way` after the library target exists**

Immediately after the existing `ground_perspective` target definition and its include directories, add:

```cmake
target_link_libraries(find_way ground_perspective)
```

Keep the `target_compile_options(... -std=c++11)` lines from commit `5de5ec3` exactly as they are.

- [ ] **Step 3: Mirror only the touched runtime files to WSL2**

Confirm the WSL copies are clean first, then mirror them:

```powershell
wsl.exe -d Ubuntu-20.04 -- bash -lc "cd /home/wtr/program/iflycar && git status --short src/vision_line/src/find_way.cpp src/vision_line/CMakeLists.txt"
wsl.exe -d Ubuntu-20.04 -- bash -lc "cp /mnt/d/programs/ucar_ws/src/vision_line/src/find_way.cpp /home/wtr/program/iflycar/src/vision_line/src/find_way.cpp"
wsl.exe -d Ubuntu-20.04 -- bash -lc "cp /mnt/d/programs/ucar_ws/src/vision_line/CMakeLists.txt /home/wtr/program/iflycar/src/vision_line/CMakeLists.txt"
```

- [ ] **Step 4: Build `find_way` and run the full package tests**

```powershell
wsl.exe -d Ubuntu-20.04 -- bash -lc "source /opt/ros/noetic/setup.bash && cd /home/wtr/program/iflycar && catkin_make --pkg vision_line --make-args find_way"
wsl.exe -d Ubuntu-20.04 -- bash -lc "source /opt/ros/noetic/setup.bash && cd /home/wtr/program/iflycar && catkin_make run_tests_vision_line && catkin_test_results --verbose"
```

Expected: `find_way` links successfully; all `vision_line` tests pass with zero failures.

- [ ] **Step 5: Verify scope and data-flow requirements**

```powershell
rg -n -C 8 "resize_frame|warpFixedGroundPerspective|preprocess|perspectiveFrame" src/vision_line/src/find_way.cpp
git diff --check -- src/vision_line/include/ground_perspective.h src/vision_line/src/ground_perspective.cpp src/vision_line/src/find_way.cpp src/vision_line/CMakeLists.txt src/vision_line/test/test_ground_perspective.cpp
git status --short
```

Confirm in order:

1. `resize_frame(frame)` runs before the state branch.
2. Only `STRAIGHT_TRACKING` calls `warpFixedGroundPerspective`.
3. `set_frame` and `preprocess` receive the selected `process_frame`.
4. Left/right turning still use the resized original frame.
5. No task edit was made to `CameraCapture::perspectiveFrame`.
6. The WSL-only `picture_cli.py` edit and Windows `Testing/` directory remain untouched.

- [ ] **Step 6: Commit Task 2**

Stage only the runtime wiring files, inspect the staged diff, and commit:

```powershell
git add src/vision_line/src/find_way.cpp src/vision_line/CMakeLists.txt
git diff --cached --check
git diff --cached
git commit -m "feat: use fixed ground perspective for straight tracking"
```
