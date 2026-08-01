# Traffic-Light YOLOv8 Multi-NPU Demo Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 src/traffic_light 中实现一个固定使用 bestfp.rknn 的持续相机演示程序，通过 3 个分别绑定 NPU core 0、1、2 的线程并行执行 YOLOv8 RKNN 推理并显示结果。

**Architecture:** 直接移植参考工程的 ThreadPool 和 rknnPool 结构，增加一个薄的 YoloV8Model 适配器，复用已有的官方 YOLOv8 初始化、letterbox、RKNN 推理和 DFL/NMS 后处理。主线程用现有 CameraCapture 采集 BGR 帧，worker 将帧转换为 RGB 后调用现有 C 接口，按 future FIFO 顺序返回绘制后的帧。

**Tech Stack:** C++11、OpenCV、RKNN Runtime API、RGA、pthread、catkin/gtest。

## Global Constraints

- 运行模式是持续相机流，不是单帧模式。
- 固定创建 3 个推理线程，分别绑定 RKNN_NPU_CORE_0、RKNN_NPU_CORE_1、RKNN_NPU_CORE_2。
- 模型路径必须写在代码中：src/traffic_light/models/bestfp.rknn；不读取命令行模型参数。
- 摄像头固定为 index 0、640x480；按 q 退出。
- 复用已有 YOLOv8 官方前处理、推理、DFL 和 NMS；不移植 YOLOv5 检测算法。
- 自定义模型类别数从 score tensor channel 读取；显示标签固定为 stop、straight、right、left。
- yolov8_zero_copy.cc 不参与本 executable，避免与普通实现重复定义符号。
- 只暂存本计划列出的文件；src/traffic_light 中已有用户未跟踪文件不得被覆盖或加入提交。

---

## File map

- Create: src/traffic_light/include/thread_pool.hpp — 参考工程 ThreadPool.hpp 的最小移植。
- Create: src/traffic_light/include/rknn_pool.hpp — 参考工程 rknnPool.hpp 的最小移植，新增显式 worker_id 初始化参数。
- Create: src/traffic_light/include/yolov8_model.hpp — YoloV8Model 的接口和 worker 到 NPU core 的映射。
- Create: src/traffic_light/src/yolov8_model.cc — BGR/RGB 转换、现有 YOLOv8 C API 调用和 OpenCV 绘制。
- Create: src/traffic_light/src/traffic_light_demo.cc — 固定模型路径的相机采集、线程池提交/取回和窗口显示。
- Create: src/traffic_light/test/test_rknn_pool.cpp — 不依赖 RKNN 硬件的线程池 FIFO 与 core id 测试。
- Create: src/traffic_light/test/test_yolov8_postprocess.cpp — 自定义 score tensor 类别数测试。
- Modify: src/traffic_light/include/yolov8.h — 增加带共享 context/core mask 的初始化接口和无硬件依赖的 inline core 映射。
- Modify: src/traffic_light/src/yolov8.cc — 保留现有官方流程，支持 rknn_init/rknn_dup_context、core mask 和失败清理。
- Modify: src/traffic_light/include/postprocess.h — 暴露类别数读取辅助函数。
- Modify: src/traffic_light/src/postprocess.cc — 让 output branch 按 score tensor channel 动态遍历类别。
- Modify: src/traffic_light/CMakeLists.txt — 加入 OpenCV、RKNN、RGA、pthread、demo target 和 gtest targets。

## Interfaces

The implementation tasks use these exact interfaces:

    // include/yolov8.h
    int init_yolov8_model_on_core(const char *model_path,
                                  rknn_app_context_t *app_ctx,
                                  rknn_context *shared_ctx,
                                  rknn_core_mask core_mask);
    rknn_core_mask yolov8_core_mask_for_worker(int worker_id);

    // include/yolov8_model.hpp
    class YoloV8Model {
    public:
        explicit YoloV8Model(const std::string &model_path);
        int init(rknn_context *shared_ctx, bool share_weight, int worker_id);
        rknn_context *get_pctx();
        cv::Mat infer(const cv::Mat &bgr_frame);
    };

    // include/postprocess.h
    int get_yolov8_class_count(const rknn_tensor_attr *score_attr);

The pool calls YoloV8Model::init(models[0]->get_pctx(), i != 0, i), so worker ids are explicit and deterministic.

### Task 1: Add the failing pool contract test

Files:
- Create: src/traffic_light/test/test_rknn_pool.cpp

Interfaces:
- Consumes: the future rknnPool<Model, Input, Output> contract below.
- Produces: a red test that requires rknn_pool.hpp and proves FIFO output plus worker ids 0, 1, 2.

- [ ] Step 1: Write the failing test

Create a fake model that never touches RKNN:

    #include <gtest/gtest.h>
    #include <string>
    #include <vector>
    #include "rknn_api.h"
    #include "rknn_pool.hpp"

    class FakeModel {
    public:
        static std::vector<int> initialized_workers;

        explicit FakeModel(const std::string &) {}

        int init(rknn_context *, bool, int worker_id) {
            initialized_workers.push_back(worker_id);
            return 0;
        }

        rknn_context *get_pctx() { return nullptr; }
        int infer(int input) { return input * 10; }
    };

    std::vector<int> FakeModel::initialized_workers;

    TEST(RknnPool, InitializesThreeWorkersAndPreservesFifoResults) {
        FakeModel::initialized_workers.clear();
        rknnPool<FakeModel, int, int> pool("ignored", 3);

        ASSERT_EQ(pool.init(), 0);
        ASSERT_EQ(FakeModel::initialized_workers,
                  (std::vector<int>{0, 1, 2}));

        ASSERT_EQ(pool.put(1), 0);
        ASSERT_EQ(pool.put(2), 0);
        ASSERT_EQ(pool.put(3), 0);

        int output = 0;
        ASSERT_EQ(pool.get(output), 0);
        EXPECT_EQ(output, 10);
        ASSERT_EQ(pool.get(output), 0);
        EXPECT_EQ(output, 20);
        ASSERT_EQ(pool.get(output), 0);
        EXPECT_EQ(output, 30);
        EXPECT_EQ(pool.get(output), 1);
    }

- [ ] Step 2: Run the test to verify it fails

    g++ -std=c++11 -pthread -I3rdparty/librknn/include -Isrc/traffic_light/include src/traffic_light/test/test_rknn_pool.cpp -lgtest -lgtest_main -o /tmp/test_rknn_pool

Expected: compilation fails because rknn_pool.hpp does not exist.

### Task 2: Port the reference thread pool and make the pool test green

Files:
- Create: src/traffic_light/include/thread_pool.hpp
- Create: src/traffic_light/include/rknn_pool.hpp

Interfaces:
- Consumes: the fake model contract in Task 1.
- Produces: rknnPool<Model, Input, Output> with init, put, get, and destructor semantics matching the reference implementation.

- [ ] Step 1: Port ThreadPool.hpp without changing scheduling semantics

Copy /home/wtr/program/rk_toolkits/rknn-cpp-Multithreading/include/ThreadPool.hpp into thread_pool.hpp, preserving its dpool::ThreadPool::submit API and Apache license header. Replace its C++14-only std::make_unique call with an explicit std::unique_ptr(new ...) construction so the package remains C++11-compatible. Do not add bounded queues, frame dropping, or a new scheduler.

- [ ] Step 2: Port rknnPool.hpp and add the explicit worker id

Use the reference implementation as the base. Keep the same members (threadNum, model path, round-robin id, mutexes, FIFO future queue, model vector), but use this initialization call:

    ret = models[i]->init(models[0]->get_pctx(), i != 0, i);

Keep put as round-robin model selection and get as FIFO future::get. The pool destructor must consume all remaining futures before the thread pool and model objects are destroyed.

- [ ] Step 3: Run the test to verify it passes

    g++ -std=c++11 -pthread -I3rdparty/librknn/include -Isrc/traffic_light/include src/traffic_light/test/test_rknn_pool.cpp -lgtest -lgtest_main -o /tmp/test_rknn_pool
    /tmp/test_rknn_pool

Expected: the test compiles and reports one passing test, including worker ids [0, 1, 2] and outputs 10, 20, 30 in submission order.

- [ ] Step 4: Commit the pool port

    git add src/traffic_light/include/thread_pool.hpp src/traffic_light/include/rknn_pool.hpp src/traffic_light/test/test_rknn_pool.cpp
    git commit -m "feat: port rknn inference thread pool"

### Task 3: Add the failing dynamic-class-count test

Files:
- Create: src/traffic_light/test/test_yolov8_postprocess.cpp

Interfaces:
- Consumes: get_yolov8_class_count(const rknn_tensor_attr *).
- Produces: a red test requiring the YOLOv8 score channel helper.

- [ ] Step 1: Write the failing test

    #include <gtest/gtest.h>
    #include "yolov8.h"
    #include "postprocess.h"

    TEST(YoloV8Postprocess, ReadsCustomClassCountFromScoreChannels) {
        rknn_tensor_attr score_attr{};
        score_attr.n_dims = 4;
        score_attr.dims[0] = 1;
        score_attr.dims[1] = 4;
        score_attr.dims[2] = 80;
        score_attr.dims[3] = 80;

        EXPECT_EQ(get_yolov8_class_count(&score_attr), 4);
    }

- [ ] Step 2: Run the test to verify it fails

    g++ -std=c++11 -I3rdparty/librknn/include -Isrc/traffic_light/include src/traffic_light/test/test_yolov8_postprocess.cpp src/traffic_light/src/postprocess.cc -lgtest -lgtest_main -lm -o /tmp/test_yolov8_postprocess

Expected: compilation fails because get_yolov8_class_count is not declared or defined.

### Task 4: Implement dynamic YOLOv8 class count and make its test green

Files:
- Modify: src/traffic_light/include/postprocess.h
- Modify: src/traffic_light/src/postprocess.cc
- Modify: src/traffic_light/test/test_yolov8_postprocess.cpp only if compiler include ordering requires it.

Interfaces:
- Consumes: rknn_tensor_attr score output attributes.
- Produces: get_yolov8_class_count, plus internal process_i8, process_u8, process_fp32, and RV1106 helpers that receive class_num and never loop past the score tensor channel count.

- [ ] Step 1: Add the minimal public helper

Declare and define:

    int get_yolov8_class_count(const rknn_tensor_attr *score_attr) {
        if (score_attr == nullptr || score_attr->n_dims < 2) {
            return 0;
        }
    #ifdef RKNPU1
        return score_attr->dims[2];
    #else
        return score_attr->dims[1];
    #endif
    }

- [ ] Step 2: Thread class_num through the existing process helpers

Add int class_num to the process helper parameters and replace only the class iteration:

    for (int c = 0; c < OBJ_CLASS_NUM; ++c)

with:

    for (int c = 0; c < class_num; ++c)

In post_process, calculate the score output index and pass:

    const int class_num =
        get_yolov8_class_count(&app_ctx->output_attrs[score_idx]);

If class_num <= 0, print an error and return -1. Do not change DFL, dequantization, threshold, sorting, NMS, or box coordinate conversion.

- [ ] Step 3: Run the test to verify it passes

    g++ -std=c++11 -I3rdparty/librknn/include -Isrc/traffic_light/include src/traffic_light/test/test_yolov8_postprocess.cpp src/traffic_light/src/postprocess.cc -lgtest -lgtest_main -lm -o /tmp/test_yolov8_postprocess
    /tmp/test_yolov8_postprocess

Expected: one passing test and no fixed 80-class iteration.

- [ ] Step 4: Commit the postprocess adaptation

    git add src/traffic_light/include/postprocess.h src/traffic_light/src/postprocess.cc src/traffic_light/test/test_yolov8_postprocess.cpp
    git commit -m "fix: use yolov8 output class count"

### Task 5: Add core mapping and YOLOv8 model adapter

Files:
- Create: src/traffic_light/include/yolov8_model.hpp
- Create: src/traffic_light/src/yolov8_model.cc
- Modify: src/traffic_light/include/yolov8.h
- Modify: src/traffic_light/src/yolov8.cc
- Modify: src/traffic_light/test/test_rknn_pool.cpp

Interfaces:
- Consumes: rknnPool model contract, rknn_app_context_t, init_yolov8_model, inference_yolov8_model, and object_detect_result_list.
- Produces: YoloV8Model::init, YoloV8Model::get_pctx, YoloV8Model::infer, init_yolov8_model_on_core, and yolov8_core_mask_for_worker.

- [ ] Step 1: Add the core mapping test before implementation

Extend test_rknn_pool.cpp with `#include "yolov8.h"` before the test body, then add:

    TEST(YoloV8Core, MapsWorkerIdsToDedicatedCores) {
        EXPECT_EQ(yolov8_core_mask_for_worker(0), RKNN_NPU_CORE_0);
        EXPECT_EQ(yolov8_core_mask_for_worker(1), RKNN_NPU_CORE_1);
        EXPECT_EQ(yolov8_core_mask_for_worker(2), RKNN_NPU_CORE_2);
        EXPECT_EQ(yolov8_core_mask_for_worker(3), RKNN_NPU_CORE_UNDEFINED);
    }

Run the test before adding the mapping. Expected: compile failure because the declaration is absent.

- [ ] Step 2: Add explicit core mapping

Declare `init_yolov8_model_on_core` in yolov8.h and define `yolov8_core_mask_for_worker` as an inline switch in the same header for ids 0, 1, and 2; return RKNN_NPU_CORE_UNDEFINED for every other id. Keeping this pure mapping inline lets test_rknn_pool link without the RKNN runtime. The model adapter must call this function instead of relying on implicit global round-robin state.

- [ ] Step 3: Extend model initialization without changing the existing wrapper contract

Keep init_yolov8_model(const char *, rknn_app_context_t *) as a compatibility wrapper that calls the new initializer with shared_ctx = nullptr and RKNN_NPU_CORE_AUTO. Implement init_yolov8_model_on_core as follows:

1. Zero-initialize the app context and local pointers.
2. If shared_ctx == nullptr, read the model file and call rknn_init; otherwise call rknn_dup_context(shared_ctx, &ctx).
3. Call rknn_set_core_mask(ctx, core_mask) immediately after context creation.
4. Reuse the existing tensor-count, input-attribute, output-attribute, quantization, and input-dimension queries.
5. On every failure, free model data and allocated attributes and destroy the newly created context.

Do not enable zero-copy, change tensor formats, or duplicate model data for child contexts.

- [ ] Step 4: Implement YoloV8Model

Use this class shape:

    class YoloV8Model {
    public:
        explicit YoloV8Model(const std::string &model_path);
        int init(rknn_context *shared_ctx, bool share_weight, int worker_id);
        rknn_context *get_pctx();
        cv::Mat infer(const cv::Mat &bgr_frame);
        ~YoloV8Model();

    private:
        std::string model_path_;
        int worker_id_ = -1;
        rknn_app_context_t app_ctx_{};
    };

infer must clone the input BGR frame, convert it with cv::cvtColor(frame, rgb, cv::COLOR_BGR2RGB), fill an image_buffer_t pointing at rgb.data with IMAGE_FORMAT_RGB888 and no ownership, call inference_yolov8_model, and draw each result with cv::rectangle/cv::putText. Use the labels stop, straight, right, left; for an id outside [0, 3], display class_<id>. On failure print the worker id and return the unmodified clone. The destructor calls release_yolov8_model exactly once and never calls the legacy label-file initializer.

- [ ] Step 5: Run the tests and compile the adapter

    g++ -std=c++11 -pthread -I3rdparty/librknn/include -Isrc/traffic_light/include src/traffic_light/test/test_rknn_pool.cpp src/traffic_light/src/yolov8.cc src/traffic_light/src/yolov8_model.cc src/traffic_light/src/postprocess.cc src/traffic_light/src/image_utils.c src/traffic_light/src/file_utils.c $(pkg-config --cflags --libs opencv4) 3rdparty/librknn/librknnrt.so 3rdparty/rga/libs/librga.so -ldl -lm -lgtest -lgtest_main -o /tmp/test_yolov8_model
    /tmp/test_yolov8_model

Expected: pool and core mapping tests pass. If the host lacks target-architecture RKNN or RGA link support, record that exact linker limitation and continue with CMake syntax verification.

- [ ] Step 6: Commit the model adapter

    git add src/traffic_light/include/yolov8.h src/traffic_light/src/yolov8.cc src/traffic_light/include/yolov8_model.hpp src/traffic_light/src/yolov8_model.cc src/traffic_light/test/test_rknn_pool.cpp
    git commit -m "feat: adapt yolov8 inference to npu workers"

### Task 6: Add the fixed-path camera demo

Files:
- Create: src/traffic_light/src/traffic_light_demo.cc

Interfaces:
- Consumes: CameraCapture, rknnPool<YoloV8Model, cv::Mat, cv::Mat>, and the model adapter from Task 5.
- Produces: a no-argument main() executable that continuously displays ordered results.

- [ ] Step 1: Write the demo loop with fixed constants

Use no argc/argv model parsing and these constants:

    static const char *kModelPath = "src/traffic_light/models/bestfp.rknn";
    static const int kThreadCount = 3;
    static const int kCameraIndex = 0;
    static const int kCameraWidth = 640;
    static const int kCameraHeight = 480;

Submit frames and retrieve results using this reference-order skeleton:

    rknnPool<YoloV8Model, cv::Mat, cv::Mat> pool(kModelPath, kThreadCount);
    if (pool.init() != 0) {
        return -1;
    }
    CameraCapture camera(kCameraIndex, kCameraWidth, kCameraHeight);
    cv::namedWindow("Camera FPS", cv::WINDOW_AUTOSIZE);
    int frames = 0;
    while (true) {
        cv::Mat frame = camera.captureFrame();
        if (frame.empty() || pool.put(frame) != 0) {
            break;
        }
        cv::Mat result;
        if (frames >= kThreadCount && pool.get(result) != 0) {
            break;
        }
        if (frames >= kThreadCount) {
            cv::imshow("Camera FPS", result);
        }
        if (cv::waitKey(1) == 'q') {
            break;
        }
        ++frames;
    }

After capture stops, repeatedly call pool.get(result) and display each remaining result until get returns nonzero or q is pressed. Catch std::exception around camera setup/capture and return a nonzero status after printing the error.

- [ ] Step 2: Compile the demo translation unit

Compile it with the adapter sources using the include and library flags from Task 5. Expected: no missing symbol or command-line model path errors.

- [ ] Step 3: Commit the demo source

    git add src/traffic_light/src/traffic_light_demo.cc
    git commit -m "feat: add multi-npu traffic light demo"

### Task 7: Wire CMake and tests

Files:
- Modify: src/traffic_light/CMakeLists.txt

Interfaces:
- Consumes: all production sources and tests from Tasks 1–6.
- Produces: traffic_light_demo, test_rknn_pool, and test_yolov8_postprocess catkin targets.

- [ ] Step 1: Add dependency discovery

Add these CMake variables and include directories:

    find_package(OpenCV REQUIRED)
    set(RKNN_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../../3rdparty/librknn)
    set(RGA_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../../3rdparty/rga)
    set(RKNN_LIBRARY ${RKNN_ROOT}/librknnrt.so)
    set(RGA_LIBRARY ${RGA_ROOT}/libs/librga.so)
    include_directories(include ${catkin_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS} ${RKNN_ROOT}/include ${RGA_ROOT}/include)

- [ ] Step 2: Add the production executable

Add exactly these sources:

    add_executable(traffic_light_demo
      src/traffic_light_demo.cc
      src/yolov8_model.cc
      src/yolov8.cc
      src/postprocess.cc
      src/camera_capture.cpp
      src/image_utils.c
      src/file_utils.c
    )
    target_link_libraries(traffic_light_demo ${catkin_LIBRARIES} ${OpenCV_LIBS} ${RKNN_LIBRARY} ${RGA_LIBRARY} pthread dl m)

Do not add yolov8_zero_copy.cc or the audio/drawing files to this target.

- [ ] Step 3: Add gtest targets

Under if(CATKIN_ENABLE_TESTING), add:

    catkin_add_gtest(test_rknn_pool test/test_rknn_pool.cpp)
    if(TARGET test_rknn_pool)
      target_link_libraries(test_rknn_pool pthread)
    endif()

    catkin_add_gtest(test_yolov8_postprocess test/test_yolov8_postprocess.cpp src/postprocess.cc)
    if(TARGET test_yolov8_postprocess)
      target_link_libraries(test_yolov8_postprocess m)
    endif()

- [ ] Step 4: Configure the package

    catkin_make --pkg traffic_light -DCATKIN_ENABLE_TESTING=ON

Expected: CMake discovers the local RKNN/RGA/OpenCV paths and creates the demo plus both test targets. If catkin is unavailable, record the exact missing environment dependency instead of claiming a build.

- [ ] Step 5: Run the unit tests

    catkin_make run_tests_traffic_light_gtest_test_rknn_pool run_tests_traffic_light_gtest_test_yolov8_postprocess

Expected: both test binaries pass; no source outside the plan is staged.

- [ ] Step 6: Commit build integration

    git add src/traffic_light/CMakeLists.txt
    git commit -m "build: add traffic light yolov8 demo targets"

### Task 8: Target-board verification and final review

Files:
- Modify: only files required by a failing verification step; report the reason before changing user-provided helper files.

Interfaces:
- Consumes: traffic_light_demo and the verified unit tests.
- Produces: evidence for core binding, camera display, ordered results, shutdown, and clean staged scope.

- [ ] Step 1: Verify repository scope

    git status --short
    git diff --check HEAD~1..HEAD
    git log -5 --oneline

Confirm that only planned files are committed and the pre-existing untracked helper files remain untracked.

- [ ] Step 2: Run host-side verification

Run the full package build and both gtests from Task 7. Record the exact output; do not claim a target-board run from host-only evidence.

- [ ] Step 3: Run the demo on the RK3588 target

From the workspace root run:

    ./devel/lib/traffic_light/traffic_light_demo

Verify the startup log contains one successful core mask for each of core 0, 1, and 2, the camera window displays detections from bestfp.rknn, and pressing q drains pending frames before exit.

- [ ] Step 4: Final diff review

    git diff --stat one_line...HEAD
    git status --short

Review every changed line for accidental YOLOv5 logic, command-line model parsing, fixed 80-class iteration, zero-copy source inclusion, or staging of the user's untracked files.
