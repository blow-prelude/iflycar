# Fixed Ground Perspective C++ Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a tested C++ camera executable that applies the fixed `y = 139` ground mask and expanded bird's-eye perspective transform used by the Python prototype.

**Architecture:** Put hardware-independent OpenCV operations in a small `ground_perspective` library and keep camera ownership plus GUI control in one executable. Reuse the existing `CameraCapture` class without changing its current perspective API, so existing line-following targets are unaffected.

**Tech Stack:** C++11, OpenCV, catkin/CMake, GoogleTest through `catkin_add_gtest`.

## Global Constraints

- Reuse `CameraCapture(0, 640, 480)`.
- Process every frame in this exact order: capture, undistort, horizontal flip, resize to `320 x 240`, mask, warp.
- Keep only rows where `y >= 139`; do not detect blue dynamically.
- Use source points `(114,146)`, `(206,146)`, `(271,184)`, `(35,187)` in TL/TR/BR/BL order.
- Use `0.5 m` width, `0.75 m` length, and `200 px/m`.
- Display the prepared frame, bird's-eye image, and mask with `cv::imshow`.
- Do not save images and do not add command-line or ROS parameters.
- Exit on `q`, `Q`, or `Esc`.
- Do not modify the existing Python implementation or existing perspective behavior.

---

### Task 1: Tested ground-perspective algorithm library

**Files:**
- Create: `src/vision_line/include/ground_perspective.h`
- Create: `src/vision_line/src/ground_perspective.cpp`
- Create: `src/vision_line/test/test_ground_perspective.cpp`
- Modify: `src/vision_line/CMakeLists.txt`

**Interfaces:**
- Consumes: OpenCV `cv::Mat`, `cv::Size`, and four TL/TR/BR/BL `cv::Point2f` values.
- Produces: `vision_line::WarpGeometry`, `makeFixedGroundMask`, `makeMetricDestination`, `makeExpandedHomography`, `warpGround`, `flipAndResize`, and `shouldExit`.

- [ ] **Step 1: Add the failing algorithm tests**

Create `src/vision_line/test/test_ground_perspective.cpp` with these behaviors:

```cpp
#include "ground_perspective.h"

#include <gtest/gtest.h>
#include <opencv2/core.hpp>

#include <vector>

namespace
{
const std::vector<cv::Point2f> kSourcePoints{
    {114.0F, 146.0F}, {206.0F, 146.0F},
    {271.0F, 184.0F}, {35.0F, 187.0F}};
}

TEST(FixedGroundMask, KeepsRowsStartingAt139)
{
    const cv::Mat mask = vision_line::makeFixedGroundMask({320, 240}, 139);
    EXPECT_EQ(0, cv::countNonZero(mask.row(138)));
    EXPECT_EQ(320, cv::countNonZero(mask.row(139)));
    EXPECT_EQ(320 * (240 - 139), cv::countNonZero(mask));
}

TEST(FixedGroundMask, ClampsStartRowToImage)
{
    EXPECT_EQ(12, cv::countNonZero(
                      vision_line::makeFixedGroundMask({4, 3}, -1)));
    EXPECT_EQ(0, cv::countNonZero(
                     vision_line::makeFixedGroundMask({4, 3}, 9)));
}

TEST(MetricDestination, ConvertsMetresToPixels)
{
    const auto points = vision_line::makeMetricDestination(0.5, 0.75, 200.0);
    ASSERT_EQ(4U, points.size());
    EXPECT_EQ(cv::Point2f(0.0F, 0.0F), points[0]);
    EXPECT_EQ(cv::Point2f(100.0F, 0.0F), points[1]);
    EXPECT_EQ(cv::Point2f(100.0F, 150.0F), points[2]);
    EXPECT_EQ(cv::Point2f(0.0F, 150.0F), points[3]);
}

TEST(ExpandedHomography, PreservesCalibrationGeometryAndContainsGroundRoi)
{
    const cv::Mat mask = vision_line::makeFixedGroundMask({320, 240}, 139);
    const auto destination =
        vision_line::makeMetricDestination(0.5, 0.75, 200.0);
    const auto geometry = vision_line::makeExpandedHomography(
        kSourcePoints, destination, mask);

    std::vector<cv::Point2f> mapped_source;
    cv::perspectiveTransform(kSourcePoints, mapped_source,
                             geometry.homography);
    EXPECT_NEAR(100.0, mapped_source[1].x - mapped_source[0].x, 1.0e-3);
    EXPECT_NEAR(0.0, mapped_source[1].y - mapped_source[0].y, 1.0e-3);
    EXPECT_NEAR(0.0, mapped_source[3].x - mapped_source[0].x, 1.0e-3);
    EXPECT_NEAR(150.0, mapped_source[3].y - mapped_source[0].y, 1.0e-3);

    const std::vector<cv::Point2f> roi{
        {0.0F, 139.0F}, {319.0F, 139.0F},
        {319.0F, 239.0F}, {0.0F, 239.0F}};
    std::vector<cv::Point2f> mapped_roi;
    cv::perspectiveTransform(roi, mapped_roi, geometry.homography);
    ASSERT_GT(geometry.output_size.width, 0);
    ASSERT_GT(geometry.output_size.height, 0);
    for (const auto &point : mapped_roi)
    {
        EXPECT_GE(point.x, -1.0e-3F);
        EXPECT_GE(point.y, -1.0e-3F);
        EXPECT_LE(point.x, geometry.output_size.width - 1.0F + 1.0e-3F);
        EXPECT_LE(point.y, geometry.output_size.height - 1.0F + 1.0e-3F);
    }
}

TEST(WarpGround, KeepsTheCompleteFixedGroundRoi)
{
    const cv::Mat frame(4, 4, CV_8UC3, cv::Scalar(10, 20, 30));
    const cv::Mat mask = vision_line::makeFixedGroundMask({4, 4}, 2);
    const std::vector<cv::Point2f> source{
        {0.0F, 0.0F}, {3.0F, 0.0F}, {3.0F, 3.0F}, {0.0F, 3.0F}};
    const cv::Mat warped =
        vision_line::warpGround(frame, mask, source, 3.0, 3.0, 1.0);
    EXPECT_EQ(cv::Size(4, 2), warped.size());
    EXPECT_EQ(frame.type(), warped.type());
    EXPECT_EQ(cv::Vec3b(10, 20, 30), warped.at<cv::Vec3b>(0, 0));
}

TEST(FramePreparation, FlipsCorrectedFrameBeforeResize)
{
    cv::Mat corrected(1, 2, CV_8UC1);
    corrected.at<unsigned char>(0, 0) = 1;
    corrected.at<unsigned char>(0, 1) = 2;
    const cv::Mat prepared = vision_line::flipAndResize(corrected, {2, 1});
    EXPECT_EQ(2, prepared.at<unsigned char>(0, 0));
    EXPECT_EQ(1, prepared.at<unsigned char>(0, 1));
}

TEST(KeyHandling, AcceptsQUppercaseQAndEscape)
{
    EXPECT_TRUE(vision_line::shouldExit('q'));
    EXPECT_TRUE(vision_line::shouldExit('Q'));
    EXPECT_TRUE(vision_line::shouldExit(27));
    EXPECT_FALSE(vision_line::shouldExit('x'));
}
```

Add the test target to `src/vision_line/CMakeLists.txt` before creating the
production files:

```cmake
if(CATKIN_ENABLE_TESTING)
  catkin_add_gtest(test_ground_perspective
    test/test_ground_perspective.cpp)
  if(TARGET test_ground_perspective)
    target_link_libraries(test_ground_perspective
      ground_perspective
      ${OpenCV_LIBS})
    target_include_directories(test_ground_perspective PUBLIC
      include
      ${catkin_INCLUDE_DIRS}
      ${OpenCV_INCLUDE_DIRS})
  endif()
endif()
```

- [ ] **Step 2: Run the test build and verify RED**

Run:

```bash
catkin_make run_tests_vision_line_gtest_test_ground_perspective
```

Expected: compilation or configuration fails because `ground_perspective.h`
and the `ground_perspective` target do not exist yet. The failure must be about
the missing production interface, not a typo in the test.

- [ ] **Step 3: Declare the minimal public API**

Create `src/vision_line/include/ground_perspective.h`:

```cpp
#pragma once

#include <opencv2/core.hpp>

#include <vector>

namespace vision_line
{
struct WarpGeometry
{
    cv::Mat homography;
    cv::Size output_size;
};

cv::Mat makeFixedGroundMask(const cv::Size &frame_size, int start_y);
std::vector<cv::Point2f> makeMetricDestination(
    double width_m, double length_m, double pixels_per_m);
WarpGeometry makeExpandedHomography(
    const std::vector<cv::Point2f> &source_points,
    const std::vector<cv::Point2f> &destination_points,
    const cv::Mat &ground_mask);
cv::Mat warpGround(
    const cv::Mat &frame,
    const cv::Mat &ground_mask,
    const std::vector<cv::Point2f> &source_points,
    double width_m,
    double length_m,
    double pixels_per_m);
cv::Mat flipAndResize(const cv::Mat &corrected_frame,
                      const cv::Size &output_size);
bool shouldExit(int key_code);
} // namespace vision_line
```

- [ ] **Step 4: Implement the smallest code that satisfies the tests**

Create `src/vision_line/src/ground_perspective.cpp`. The implementation must:

- reject non-positive image/output sizes and empty images with
  `std::invalid_argument`;
- clamp mask `start_y` with `std::max(0, std::min(start_y, height))`;
- create the mask with `CV_8UC1` and set the retained rectangle to `255`;
- reject non-positive metric dimensions and require exactly four source and
  destination points;
- use `cv::getPerspectiveTransform` for the base homography;
- use `cv::findNonZero` plus `cv::boundingRect` to obtain the retained ground
  ROI, rejecting an empty mask;
- transform the four inclusive ROI corners with `cv::perspectiveTransform`;
- use floor/ceil bounds and a translation matrix to make all output
  coordinates non-negative;
- use `cv::bitwise_and(frame, frame, masked_frame, ground_mask)` followed by
  `cv::warpPerspective` with `cv::INTER_LINEAR`;
- implement `flipAndResize` as horizontal `cv::flip` followed by
  `cv::resize(..., cv::INTER_AREA)`;
- mask `key_code` with `0xFF` before checking `27`, `'q'`, and `'Q'`.

Use this implementation:

```cpp
#include "ground_perspective.h"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace vision_line
{
cv::Mat makeFixedGroundMask(const cv::Size &frame_size, int start_y)
{
    if (frame_size.width <= 0 || frame_size.height <= 0)
    {
        throw std::invalid_argument("frame size must be positive");
    }
    const int top = std::max(0, std::min(start_y, frame_size.height));
    cv::Mat mask = cv::Mat::zeros(frame_size, CV_8UC1);
    if (top < frame_size.height)
    {
        mask(cv::Rect(0, top, frame_size.width,
                      frame_size.height - top)).setTo(255);
    }
    return mask;
}

std::vector<cv::Point2f> makeMetricDestination(
    double width_m, double length_m, double pixels_per_m)
{
    if (width_m <= 0.0 || length_m <= 0.0 || pixels_per_m <= 0.0)
    {
        throw std::invalid_argument("metric dimensions must be positive");
    }
    const float width_px = static_cast<float>(width_m * pixels_per_m);
    const float length_px = static_cast<float>(length_m * pixels_per_m);
    return {{0.0F, 0.0F}, {width_px, 0.0F},
            {width_px, length_px}, {0.0F, length_px}};
}

WarpGeometry makeExpandedHomography(
    const std::vector<cv::Point2f> &source_points,
    const std::vector<cv::Point2f> &destination_points,
    const cv::Mat &ground_mask)
{
    if (source_points.size() != 4U || destination_points.size() != 4U)
    {
        throw std::invalid_argument("exactly four point pairs are required");
    }
    if (ground_mask.empty() || ground_mask.type() != CV_8UC1)
    {
        throw std::invalid_argument("ground mask must be a non-empty CV_8UC1 image");
    }

    std::vector<cv::Point> foreground;
    cv::findNonZero(ground_mask, foreground);
    if (foreground.empty())
    {
        throw std::invalid_argument("ground mask contains no retained pixels");
    }
    const cv::Rect bounds = cv::boundingRect(foreground);
    const float left = static_cast<float>(bounds.x);
    const float top = static_cast<float>(bounds.y);
    const float right = static_cast<float>(bounds.x + bounds.width - 1);
    const float bottom = static_cast<float>(bounds.y + bounds.height - 1);
    const std::vector<cv::Point2f> roi{
        {left, top}, {right, top}, {right, bottom}, {left, bottom}};

    const cv::Mat base =
        cv::getPerspectiveTransform(source_points, destination_points);
    std::vector<cv::Point2f> transformed;
    cv::perspectiveTransform(roi, transformed, base);

    double min_x = std::numeric_limits<double>::infinity();
    double min_y = std::numeric_limits<double>::infinity();
    double max_x = -std::numeric_limits<double>::infinity();
    double max_y = -std::numeric_limits<double>::infinity();
    for (const auto &point : transformed)
    {
        min_x = std::min(min_x, static_cast<double>(point.x));
        min_y = std::min(min_y, static_cast<double>(point.y));
        max_x = std::max(max_x, static_cast<double>(point.x));
        max_y = std::max(max_y, static_cast<double>(point.y));
    }
    min_x = std::floor(min_x);
    min_y = std::floor(min_y);
    max_x = std::ceil(max_x);
    max_y = std::ceil(max_y);

    const cv::Size output_size(
        static_cast<int>(max_x - min_x) + 1,
        static_cast<int>(max_y - min_y) + 1);
    if (output_size.width <= 0 || output_size.height <= 0)
    {
        throw std::invalid_argument("transformed ground ROI has invalid size");
    }

    cv::Mat translation = cv::Mat::eye(3, 3, CV_64F);
    translation.at<double>(0, 2) = -min_x;
    translation.at<double>(1, 2) = -min_y;
    return {translation * base, output_size};
}

cv::Mat warpGround(
    const cv::Mat &frame,
    const cv::Mat &ground_mask,
    const std::vector<cv::Point2f> &source_points,
    double width_m,
    double length_m,
    double pixels_per_m)
{
    if (frame.empty() || frame.channels() != 3)
    {
        throw std::invalid_argument("frame must be a non-empty BGR image");
    }
    if (ground_mask.size() != frame.size() || ground_mask.type() != CV_8UC1)
    {
        throw std::invalid_argument("ground mask must match the frame");
    }
    const auto destination =
        makeMetricDestination(width_m, length_m, pixels_per_m);
    const auto geometry =
        makeExpandedHomography(source_points, destination, ground_mask);
    cv::Mat masked_frame;
    cv::bitwise_and(frame, frame, masked_frame, ground_mask);
    cv::Mat warped;
    cv::warpPerspective(masked_frame, warped, geometry.homography,
                        geometry.output_size, cv::INTER_LINEAR);
    return warped;
}

cv::Mat flipAndResize(const cv::Mat &corrected_frame,
                      const cv::Size &output_size)
{
    if (corrected_frame.empty() || output_size.width <= 0 ||
        output_size.height <= 0)
    {
        throw std::invalid_argument("frame and output size must be valid");
    }
    cv::Mat flipped;
    cv::flip(corrected_frame, flipped, 1);
    cv::Mat resized;
    cv::resize(flipped, resized, output_size, 0.0, 0.0, cv::INTER_AREA);
    return resized;
}

bool shouldExit(int key_code)
{
    const int key = key_code & 0xFF;
    return key == 27 || key == 'q' || key == 'Q';
}
} // namespace vision_line
```

Add the algorithm library to `src/vision_line/CMakeLists.txt` before the test
block:

```cmake
add_library(ground_perspective src/ground_perspective.cpp)
target_link_libraries(ground_perspective ${OpenCV_LIBS})
target_include_directories(ground_perspective PUBLIC
  include
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS})
```

- [ ] **Step 5: Run the algorithm tests and verify GREEN**

Run:

```bash
catkin_make run_tests_vision_line_gtest_test_ground_perspective
catkin_test_results --verbose
```

Expected: all `test_ground_perspective` cases pass with zero failures.

- [ ] **Step 6: Commit the tested algorithm module**

```bash
git add src/vision_line/include/ground_perspective.h \
        src/vision_line/src/ground_perspective.cpp \
        src/vision_line/test/test_ground_perspective.cpp \
        src/vision_line/CMakeLists.txt
git commit -m "feat: add fixed ground perspective library"
```

---

### Task 2: Live C++ camera executable

**Files:**
- Create: `src/vision_line/src/dynamic_ground_perspective.cpp`
- Modify: `src/vision_line/CMakeLists.txt`

**Interfaces:**
- Consumes: existing `CameraCapture::{captureFrame, correctFrame, closeCamera}` and all Task 1 `vision_line` functions.
- Produces: CMake target and executable `dynamic_ground_perspective_cpp`.

- [ ] **Step 1: Add the executable target before its source exists**

Add to `src/vision_line/CMakeLists.txt`:

```cmake
add_executable(dynamic_ground_perspective_cpp
  src/dynamic_ground_perspective.cpp)
target_link_libraries(dynamic_ground_perspective_cpp
  camera_capture
  ground_perspective
  ${catkin_LIBRARIES}
  ${OpenCV_LIBS})
target_include_directories(dynamic_ground_perspective_cpp PUBLIC
  include
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS})
```

- [ ] **Step 2: Run the target build and verify RED**

Run:

```bash
catkin_make --pkg vision_line --make-args dynamic_ground_perspective_cpp
```

Expected: configuration fails because
`src/vision_line/src/dynamic_ground_perspective.cpp` does not exist.

- [ ] **Step 3: Implement the live camera entry point**

Create `src/vision_line/src/dynamic_ground_perspective.cpp` with constants in
an unnamed namespace:

```cpp
#include "camera_capture.h"
#include "ground_perspective.h"

#include <opencv2/highgui.hpp>

#include <exception>
#include <iostream>
#include <vector>

namespace
{
constexpr int kCameraIndex = 0;
constexpr int kCameraWidth = 640;
constexpr int kCameraHeight = 480;
constexpr int kFrameWidth = 320;
constexpr int kFrameHeight = 240;
constexpr int kGroundStartY = 139;
constexpr double kRealWidthM = 0.5;
constexpr double kRealLengthM = 0.75;
constexpr double kPixelsPerM = 200.0;

const std::vector<cv::Point2f> kSourcePoints{
    {114.0F, 146.0F}, {206.0F, 146.0F},
    {271.0F, 184.0F}, {35.0F, 187.0F}};
} // namespace
```

The `main()` loop must use this exact ordering:

```cpp
cv::Mat raw_frame = camera.captureFrame();
if (raw_frame.empty())
{
    std::cerr << "Error: captured frame is empty." << std::endl;
    break;
}
const cv::Mat corrected_frame = camera.correctFrame(raw_frame);
const cv::Mat frame = vision_line::flipAndResize(
    corrected_frame, {kFrameWidth, kFrameHeight});
const cv::Mat ground_mask = vision_line::makeFixedGroundMask(
    frame.size(), kGroundStartY);
const cv::Mat bird_view = vision_line::warpGround(
    frame, ground_mask, kSourcePoints,
    kRealWidthM, kRealLengthM, kPixelsPerM);
```

Display `Camera Frame`, `Bird's Eye View`, and `Ground Mask`, and break when
`vision_line::shouldExit(cv::waitKey(1))` returns true. Print a startup hint,
call `camera.closeCamera()` and `cv::destroyAllWindows()` after the loop, catch
`std::exception`, destroy windows in the catch path, print the exception, and
return `1`.

Complete `main()` as follows:

```cpp
int main()
{
    try
    {
        CameraCapture camera(kCameraIndex, kCameraWidth, kCameraHeight);
        std::cout << "Camera stream started. Press q or Esc to exit."
                  << std::endl;
        while (true)
        {
            const cv::Mat raw_frame = camera.captureFrame();
            if (raw_frame.empty())
            {
                std::cerr << "Error: captured frame is empty." << std::endl;
                break;
            }
            const cv::Mat corrected_frame = camera.correctFrame(raw_frame);
            const cv::Mat frame = vision_line::flipAndResize(
                corrected_frame, {kFrameWidth, kFrameHeight});
            const cv::Mat ground_mask = vision_line::makeFixedGroundMask(
                frame.size(), kGroundStartY);
            const cv::Mat bird_view = vision_line::warpGround(
                frame, ground_mask, kSourcePoints,
                kRealWidthM, kRealLengthM, kPixelsPerM);

            cv::imshow("Camera Frame", frame);
            cv::imshow("Bird's Eye View", bird_view);
            cv::imshow("Ground Mask", ground_mask);
            if (vision_line::shouldExit(cv::waitKey(1)))
            {
                break;
            }
        }
        camera.closeCamera();
        cv::destroyAllWindows();
        return 0;
    }
    catch (const std::exception &error)
    {
        cv::destroyAllWindows();
        std::cerr << "Error: " << error.what() << std::endl;
        return 1;
    }
}
```

- [ ] **Step 4: Build the executable and rerun tests**

Run:

```bash
catkin_make --pkg vision_line --make-args dynamic_ground_perspective_cpp
catkin_make run_tests_vision_line_gtest_test_ground_perspective
catkin_test_results --verbose
```

Expected: the executable compiles and every algorithm test passes.

- [ ] **Step 5: Verify the source contains no excluded behavior**

Run:

```bash
rg -n "inRange|HSV|imwrite|VideoWriter|CommandLineParser" \
  src/vision_line/src/dynamic_ground_perspective.cpp \
  src/vision_line/src/ground_perspective.cpp
```

Expected: no matches.

- [ ] **Step 6: Commit the executable**

```bash
git add src/vision_line/src/dynamic_ground_perspective.cpp \
        src/vision_line/CMakeLists.txt
git commit -m "feat: add C++ ground perspective camera demo"
```

---

### Task 3: Final verification and target-device handoff

**Files:**
- Verify only; no additional files are required.

**Interfaces:**
- Consumes: the CMake target and tests from Tasks 1 and 2.
- Produces: fresh build/test evidence and the exact target-device run command.

- [ ] **Step 1: Run the full package build and tests**

```bash
catkin_make --pkg vision_line
catkin_make run_tests_vision_line
catkin_test_results --verbose
git diff --check HEAD~2
```

Expected: build exit code `0`, no failed tests, and no whitespace errors.

- [ ] **Step 2: Run on the target device with a connected camera**

```bash
source devel/setup.bash
rosrun vision_line dynamic_ground_perspective_cpp
```

Confirm all three windows update continuously, the top `139` rows are black in
the mask, the bird's-eye view retains the visible ground region, and `q`, `Q`,
or `Esc` exits without saving files.

- [ ] **Step 3: If OpenCV cannot be found, request only the missing paths**

Record the exact `find_package(OpenCV)` or linker error, then request the
target device's OpenCV include directory and library directory from the user.
Do not hard-code speculative paths.
