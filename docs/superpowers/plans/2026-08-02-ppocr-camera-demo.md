# PPOCR Camera Thread-Pool Demo Implementation Plan

**Goal:** 在 `src/find_signal` 中实现持续相机采集、PPOCR 线程池推理、FIFO 结果获取和 OpenCV 可视化的独立 C++ demo。

**Architecture:** 直接使用 `src/find_signal/include` 下现有的 `thread_pool.hpp` 和 `rknn_pool.hpp`。本地 `rknnPool` 做最小双模型路径适配，每个 `PPOCRModel` worker 独立初始化 det/rec RKNN context 并绑定对应 NPU core。worker 将 BGR 相机帧转换为 RGB，调用现有 `inference_ppocr_system_model`，在 BGR 副本上绘制四边形和分数，再按 future FIFO 顺序交给主线程显示。

**Tech stack:** C++11、OpenCV、RKNN Runtime API、RGA、pthread、catkin/gtest。

**Execution status:** 实现已完成并通过主机侧无硬件单元测试；完整 demo 的链接和摄像头/NPU 端到端运行仍需在 aarch64 目标板验证。当前主机为 x86_64，而工作树自带的 `librknnrt.so`、`librga.so` 是 ARM aarch64 库，因此主机只能完成源码编译和无硬件测试。

**Design reference:** `docs/superpowers/specs/2026-08-02-ppocr-camera-demo-design.md`

## Global constraints

- 所有实现都在 `/home/wtr/program/iflycar/.worktrees/ppocr` 工作树完成。
- 直接使用：
  - `src/find_signal/include/thread_pool.hpp`
  - `src/find_signal/include/rknn_pool.hpp`
- 不移动或修改 `src/traffic_light/include` 下的线程池文件。
- 保留现有 PPOCR 检测、识别、DB 后处理和 CTC 解码算法，只增加线程适配、core mask、资源清理和必要的边界保护。
- 每个 worker 独占一组 det/rec context；第一版不使用 `rknn_dup_context` 共享二模型权重。
- 默认两个 worker，分别绑定 NPU core 0 和 core 1；允许命令行将 worker 数调为 1～3。
- 默认模型为 `ppocrv4_det.rknn` 和 `ppocrv4_rec.rknn`。
- 当前已有未跟踪文件属于用户工作内容，实施时复用但不覆盖：
  - `src/find_signal/include/camera_capture.h`
  - `src/find_signal/src/camera_capture.cpp`
  - `src/find_signal/include/thread_pool.hpp`
  - `src/find_signal/include/rknn_pool.hpp`
- 本计划不授权自动提交 git commit；每个任务完成后只报告状态和验证结果。

## File map

### Existing/prepared

- `src/find_signal/include/camera_capture.h` — 相机接口。
- `src/find_signal/src/camera_capture.cpp` — V4L2/OpenCV 相机实现。
- `src/find_signal/include/thread_pool.hpp` — 通用任务线程池。
- `src/find_signal/include/rknn_pool.hpp` — 已适配 det/rec 双路径的 future FIFO 和 worker 轮询池。

### Create

- `src/find_signal/include/ppocr_model.hpp` — PPOCR worker 适配器与可视化辅助接口。
- `src/find_signal/src/ppocr_model.cc` — det/rec context 初始化、BGR/RGB 转换和推理。
- `src/find_signal/src/ppocr_visualization.cc` — 无 NPU 依赖的 OpenCV 结果绘制实现。
- `src/find_signal/src/find_signal_demo.cc` — 摄像头采集、线程池提交/获取、参数解析和窗口显示。
- `src/find_signal/test/test_find_signal_rknn_pool.cpp` — 双路径、worker id 和 FIFO 测试。
- `src/find_signal/test/test_ppocr_model.cpp` — core 映射和无硬件可视化测试。

### Modify

- `src/find_signal/include/rknn_pool.hpp` — 从单模型路径改为 det/rec 双路径构造。
- `src/find_signal/include/ppocr_system.h` — const 参数、core-aware 初始化和 core 映射声明。
- `src/find_signal/src/ppocr_system.cc` — core mask 初始化、失败清理和裁剪边界保护。
- `src/find_signal/CMakeLists.txt` — OpenCV/RKNN/RGA、demo 和测试 target。
- `src/find_signal/src/postprocess.cc` — 仅在测试或边界检查证明必要时做最小修正。

### Explicitly unchanged

- `src/traffic_light/**`
- `src/find_signal/scripts/**`
- `src/find_signal/models/**`
- PPOCR DB/CTC 算法主体和字典内容。

## Interfaces fixed by this plan

### Local pool contract

```cpp
template <typename rknnModel, typename inputType, typename outputType>
class rknnPool
{
public:
    rknnPool(const std::string& det_model_path,
             const std::string& rec_model_path,
             int thread_num);

    int init();
    int put(inputType input_data);
    int get(outputType& output_data);
};
```

初始化每个模型时调用：

```cpp
models[i]->init(models[0]->get_pctx(), i != 0, i);
```

PPOCR 第一版不共享 context，因此 `PPOCRModel::get_pctx()` 返回 `nullptr`，`init` 保留参数以兼容现有 pool contract，但独立初始化自己的 det/rec context。

### PPOCR core-aware API

```cpp
int init_ppocr_model_on_core(const char* model_path,
                             rknn_app_context_t* app_ctx,
                             rknn_core_mask core_mask);

inline rknn_core_mask ppocr_core_mask_for_worker(int worker_id);
```

映射固定为：

```text
0 -> RKNN_NPU_CORE_0
1 -> RKNN_NPU_CORE_1
2 -> RKNN_NPU_CORE_2
other -> RKNN_NPU_CORE_UNDEFINED
```

原有 `init_ppocr_model` 保留，并委托给 `init_ppocr_model_on_core(..., RKNN_NPU_CORE_AUTO)`，避免破坏已有调用方。

### Model adapter API

```cpp
class PPOCRModel
{
public:
    PPOCRModel(const std::string& det_model_path,
               const std::string& rec_model_path);

    int init(rknn_context* unused_shared_ctx,
             bool unused_share_weight,
             int worker_id);
    rknn_context* get_pctx();
    cv::Mat infer(const cv::Mat& bgr_frame);
    ~PPOCRModel();

private:
    std::string det_model_path_;
    std::string rec_model_path_;
    int worker_id_;
    bool det_initialized_;
    bool rec_initialized_;
    ppocr_system_app_context app_ctx_;
};

void draw_ppocr_results(cv::Mat& bgr_image,
                        const ppocr_text_recog_array_result_t& results);
```

`draw_ppocr_results` 独立暴露，便于在无 NPU 环境中用 synthetic 结果测试四边形绘制。

## Task 1: Lock the local pool contract with a test

**Files:**

- Create: `src/find_signal/test/test_find_signal_rknn_pool.cpp`
- Modify later: `src/find_signal/include/rknn_pool.hpp`

**Purpose:** 先证明当前单路径 pool 不满足 PPOCR 双路径模型构造，再实现最小适配。

- [ ] 创建 `src/find_signal/test` 目录和测试文件。
- [ ] 定义 `FakePPOCRModel`，构造函数接收 det/rec 两个路径，`init` 记录 worker id，`infer` 返回可验证结果。
- [ ] 测试两个 worker 都收到相同 det/rec 路径。
- [ ] 测试 worker id 初始化顺序为 `0, 1`。
- [ ] 提交多个输入并验证 `get` 严格按提交顺序返回，即使较早任务完成更慢。
- [ ] 验证空 future 队列时 `get` 返回非零。

测试核心结构：

```cpp
class FakePPOCRModel
{
public:
    FakePPOCRModel(const std::string& det_path,
                   const std::string& rec_path);
    int init(void*, bool, int worker_id);
    void* get_pctx();
    int infer(int input);
};

TEST(FindSignalRknnPool, PassesBothModelPathsAndPreservesFifo);
```

先运行：

```bash
g++ -std=c++11 -pthread \
  -Isrc/find_signal/include \
  src/find_signal/test/test_find_signal_rknn_pool.cpp \
  -lgtest -lgtest_main \
  -o /tmp/test_find_signal_rknn_pool
```

预期：编译失败，因为当前 `rknnPool` 只有单模型路径构造。

## Task 2: Adapt the local pool to det/rec paths

**Files:**

- Modify: `src/find_signal/include/rknn_pool.hpp`
- Test: `src/find_signal/test/test_find_signal_rknn_pool.cpp`

- [ ] 将 `modelPath` 拆为 `detModelPath` 和 `recModelPath`。
- [ ] 将构造函数改为接收两个 `const std::string&` 路径和线程数。
- [ ] 使用两个路径构造每个 `rknnModel`。
- [ ] 保留 round-robin worker 选择、future FIFO、`put`/`get` 返回值和析构排空语义。
- [ ] `thread_num <= 0` 时让 `init` 返回错误，防止取模除零。
- [ ] 某个 worker 初始化失败时清理已经构造的模型和线程池状态。
- [ ] 不修改 `thread_pool.hpp` 的调度算法。

再次运行 Task 1 的命令。

预期：测试通过，两个模型路径、worker id 和 FIFO 顺序均正确。

## Task 3: Add core mapping and core-aware PPOCR initialization

**Files:**

- Modify: `src/find_signal/include/ppocr_system.h`
- Modify: `src/find_signal/src/ppocr_system.cc`
- Create/extend: `src/find_signal/test/test_ppocr_model.cpp`

### Step 1: Add the red core mapping test

- [ ] 在 `test_ppocr_model.cpp` 中断言 worker 0/1/2 映射到对应 core。
- [ ] 断言 worker -1 和 3 返回 `RKNN_NPU_CORE_UNDEFINED`。

编译命令：

```bash
g++ -std=c++11 \
  -I3rdparty/librknn/include \
  -I3rdparty/rknn_utils \
  -Isrc/find_signal/include \
  src/find_signal/test/test_ppocr_model.cpp \
  -lgtest -lgtest_main \
  -o /tmp/test_ppocr_model_core
```

预期：首次编译失败，因为 `ppocr_core_mask_for_worker` 尚不存在。

### Step 2: Implement the mapping and initializer

- [ ] 在 `ppocr_system.h` 中加入 inline core 映射函数。
- [ ] 声明 `init_ppocr_model_on_core`。
- [ ] 让原 `init_ppocr_model` 委托给 core-aware 版本并使用 `AUTO`。
- [ ] `init_ppocr_model_on_core` 开始时校验参数并清零 `app_ctx`。
- [ ] `rknn_init` 后立即调用 `rknn_set_core_mask`。
- [ ] core mask 设置失败时销毁刚创建的 context。
- [ ] tensor 查询或属性分配失败时统一释放 context 和属性数组。
- [ ] 使用 `std::vector<rknn_tensor_attr>` 代替运行时长度栈数组，保持标准 C++11。

再次运行 core mapping 测试，预期通过。

## Task 4: Harden PPOCR streaming failure paths

**Files:**

- Modify: `src/find_signal/include/ppocr_system.h`
- Modify: `src/find_signal/src/ppocr_system.cc`
- Modify only if required: `src/find_signal/src/postprocess.cc`

**Purpose:** 单图 demo 偶发失败可以直接退出，摄像头流会长期运行，因此临时内存、RKNN output 和边界框必须在每条路径安全释放。

- [ ] 将 `ppocr_det_postprocess_params::db_score_mode` 和 `db_box_type` 改为 `const char*`。
- [ ] `inference_ppocr_det_model` 使用单一 cleanup 路径释放 resize buffer。
- [ ] 只有 `rknn_outputs_get` 成功后才调用 `rknn_outputs_release`，且所有后续分支都能到达释放点。
- [ ] `inference_ppocr_rec_model` 在 inputs set、run、outputs get 和 postprocess 失败时都释放 input buffer/output。
- [ ] `inference_ppocr_system_model` 在识别失败时先释放 `text_img.virt_addr` 再返回。
- [ ] `GetRotateCropImage` 将四边形坐标限制到图像有效范围。
- [ ] 拒绝宽或高小于 1 的裁剪框，避免构造非法 `cv::Rect` 或透视矩阵。
- [ ] 初始化 `ppocr_det_result`、`ppocr_rec_result` 和最终结果结构，避免失败时读取未初始化 count/score。
- [ ] 保持阈值、DB 算法、文字排序和 CTC 字典逻辑不变。

验证方式：

- 编译 `ppocr_system.cc` 和 `postprocess.cc`。
- 审查每个 `malloc`、`rknn_outputs_get` 都有对应释放路径。
- 后续通过 catkin target 和目标板连续运行验证无持续内存增长。

## Task 5: Implement the PPOCR model adapter

**Files:**

- Create: `src/find_signal/include/ppocr_model.hpp`
- Create: `src/find_signal/src/ppocr_model.cc`
- Extend: `src/find_signal/test/test_ppocr_model.cpp`

### Step 1: Add a visualization test

- [ ] 构造一张全黑 `CV_8UC3` 图像。
- [ ] 构造一个 synthetic `ppocr_text_recog_array_result_t`，包含有效四边形、ASCII 文本和分数。
- [ ] 调用 `draw_ppocr_results`。
- [ ] 断言框边界上的像素发生变化，图像尺寸和类型不变。
- [ ] 添加 `count == 0` 测试，断言图像不变且不崩溃。

预期：首次编译失败，因为 adapter 和绘制函数尚不存在。

### Step 2: Implement adapter lifecycle

- [ ] 构造函数保存 det/rec 路径并清零 `app_ctx_`。
- [ ] `init` 校验 worker id，取得 core mask。
- [ ] 分别调用 `init_ppocr_model_on_core` 初始化 det 和 rec。
- [ ] rec 初始化失败时立即释放已经初始化的 det context。
- [ ] 成功后设置 `initialized_ = true` 并打印 worker/core 日志。
- [ ] `get_pctx()` 返回 `nullptr`，明确第一版不共享二模型 context。
- [ ] 析构函数仅释放已经成功初始化的 context，并可安全处理部分初始化状态。

### Step 3: Implement inference and drawing

- [ ] 空输入直接返回空图像，不调用 RKNN。
- [ ] 克隆 BGR 输入作为输出画布，避免跨线程修改相机帧。
- [ ] 使用 `cv::cvtColor(..., cv::COLOR_BGR2RGB)` 创建 RGB 输入。
- [ ] 用 RGB `cv::Mat` 的数据组装非 owning `image_buffer_t`。
- [ ] 使用参考 `main.cc` 的参数调用 `inference_ppocr_system_model`：
  - threshold `0.3`
  - box threshold `0.6`
  - dilation `false`
  - score mode `slow`
  - box type `poly`
  - unclip ratio `1.5`
- [ ] 推理失败时打印 worker id 和错误码，返回未绘制的 BGR 副本。
- [ ] 成功时调用 `draw_ppocr_results`。
- [ ] 四边形坐标在绘制前限制到画布范围。
- [ ] OpenCV 默认字体绘制 ASCII/分数；完整 UTF-8 识别文本始终输出到终端。

无 NPU 绘制测试通过独立的 `ppocr_visualization.cc` target 完成，避免为模型生命周期代码引入 RKNN 运行库依赖。

## Task 6: Implement the camera demo loop

**Files:**

- Create: `src/find_signal/src/find_signal_demo.cc`
- Use: `src/find_signal/include/camera_capture.h`
- Use: `src/find_signal/include/rknn_pool.hpp`
- Use: `src/find_signal/include/ppocr_model.hpp`

- [ ] 定义默认 det/rec 路径、camera 0、640x480 和两个 worker。
- [ ] 解析可选参数：`[det_model] [rec_model] [camera_index] [thread_count]`。
- [ ] 限制 `thread_count` 为 1～3，非法输入打印 usage 并退出。
- [ ] 创建 `rknnPool<PPOCRModel, cv::Mat, cv::Mat>` 并调用 `init`。
- [ ] 模型池初始化成功后再打开摄像头，避免相机失败路径泄漏模型资源。
- [ ] 使用 RAII 对象管理 camera，避免手工 `new/delete`。
- [ ] 主循环采集一帧、提交一帧，并维护 `pending_count`。
- [ ] 当 pending 达到线程数后，每提交一帧取回并显示一帧。
- [ ] `cv::waitKey(1)` 收到 `q` 后停止提交。
- [ ] 退出采集循环后按 `pending_count` 排空已提交 future。
- [ ] 排空阶段允许再次按 `q` 跳过显示，但不能在 worker 未完成时销毁其 context。
- [ ] 相机异常、空帧或 future 异常都打印明确错误并安全退出。
- [ ] 最后调用 `cv::destroyAllWindows()`。

主流程验收语义：

```text
put(frame) increments pending
pending >= thread_count -> get(result), decrements pending
stop submitting -> while pending > 0: get(result)
destroy pool only after pending == 0
```

## Task 7: Integrate the catkin build

**Files:**

- Modify: `src/find_signal/CMakeLists.txt`

- [ ] 启用 C++11。
- [ ] 增加 `find_package(OpenCV REQUIRED)`。
- [ ] 定义 RKNN、RGA、STB 和 RKNN utils 路径，与 `traffic_light` 的 vendored 路径保持一致。
- [ ] `include_directories` 加入本包 `include`、OpenCV、RKNN、RGA、STB 和 RKNN utils。
- [ ] 增加 `find_signal_demo` executable，包含：
  - `find_signal_demo.cc`
  - `ppocr_model.cc`
  - `ppocr_visualization.cc`
  - `ppocr_system.cc`
  - `postprocess.cc`
  - `clipper.cc`
  - `camera_capture.cpp`
  - `image_utils.c`
  - `file_utils.c`
- [ ] 链接 `${catkin_LIBRARIES}`、`${OpenCV_LIBS}`、RKNN Runtime、RGA、pthread、dl、m。
- [ ] 在 `CATKIN_ENABLE_TESTING` 下增加全局唯一 target 名：
  - `test_find_signal_rknn_pool`
  - `test_find_signal_ppocr_model`
- [ ] pool 测试只链接 pthread/gtest；model 测试只链接 OpenCV 和可视化实现，不要求主机存在 RKNN Runtime。
- [ ] 不修改 `traffic_light/CMakeLists.txt`。

在工作树根目录运行：

```bash
catkin_make --pkg find_signal -DCATKIN_ENABLE_TESTING=ON
```

源码编译阶段已通过；在当前 x86_64 主机链接 `find_signal_demo` 时会因 vendored ARM aarch64 `librknnrt.so` 格式不匹配而失败，需在目标板或使用 aarch64 交叉工具链完成最终链接。

运行测试：

```bash
catkin_make run_tests_find_signal
catkin_test_results
```

预期：所有无硬件测试通过；测试过程不要求存在摄像头或 NPU。

## Task 8: Host-side final verification

- [ ] 运行 `git status --short`，确认只包含计划列出的 `find_signal` 文件和两份文档。
- [ ] 运行 `git diff --check`，确认没有空白错误。
- [ ] 检查 demo 的动态链接依赖中包含 OpenCV、RKNN Runtime 和 RGA。
- [ ] 检查默认模型路径在目标工作树中存在。
- [ ] 检查所有新增 C++ 文件使用 C++11 可编译语法。
- [ ] 记录主机无法验证的项目，不把缺少摄像头/NPU误报为代码失败。

建议命令：

```bash
git status --short
git diff --check
test -f src/find_signal/models/ppocrv4_det.rknn
test -f src/find_signal/models/ppocrv4_rec.rknn
```

## Task 9: Target-board runtime verification

在具备 RKNN Runtime、RGA、摄像头和显示环境的目标板执行：

```bash
source devel/setup.bash
rosrun find_signal find_signal_demo
```

也可显式指定参数：

```bash
rosrun find_signal find_signal_demo \
  src/find_signal/models/ppocrv4_det.rknn \
  src/find_signal/models/ppocrv4_rec.rknn \
  0 2
```

验收清单：

- [ ] worker 0/1 的 det 和 rec context 均成功初始化。
- [ ] 日志显示 worker 分别绑定 core 0 和 core 1。
- [ ] 摄像头窗口持续更新，无检测结果时也不会卡死或崩溃。
- [ ] 检测到文字时四边形坐标正确，终端输出文字和置信度。
- [ ] 与参考单图 C++ demo 使用同一图片时，框和文字结果基本一致。
- [ ] 连续运行至少 10 分钟，进程内存不持续增长。
- [ ] 按 `q` 后停止提交、排空 future、释放 context 和摄像头并正常退出。
- [ ] 分别用 1、2、3 worker 运行，非法 worker 数能在初始化前拒绝。

## Completion criteria

只有同时满足以下条件才算实现完成：

- `find_signal_demo` 源码已在 catkin 目标中完成编译；最终链接需在提供 aarch64 RKNN/RGA 库的环境完成。
- 无硬件 pool/core/绘制测试通过。
- demo 使用当前 `find_signal/include` 下的线程池头文件。
- `traffic_light` 文件没有被修改。
- BGR→RGB、PPOCR system inference、结果绘制和 FIFO 排空路径均已实现。
- 目标板可用时完成摄像头/NPU端到端验证；不可用时明确列出待板端验证项。

## Completed implementation record

- 已适配当前路径下的 `rknn_pool.hpp`，支持 det/rec 双模型路径、FIFO 结果和非法线程数检查。
- 已增加 worker 到 NPU core 的映射，并为 PPOCR det/rec context 增加 core-aware 初始化和失败清理。
- 已完成 BGR→RGB、PPOCR system inference、结果绘制、摄像头采集、线程池提交/排空和 `q` 退出流程。
- 已通过 `test_find_signal_rknn_pool`（2 tests）和 `test_find_signal_ppocr_model`（2 tests）。
- 已通过新增 C++ 源文件的 C++11 syntax-only 检查；摄像头、NPU 推理、aarch64 链接和长时间运行待目标板验证。
