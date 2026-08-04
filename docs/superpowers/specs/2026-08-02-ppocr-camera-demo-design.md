# PPOCR Camera Thread-Pool Demo Design

## Goal

在 `src/find_signal` 中增加一个类似
`src/traffic_light/src/traffic_light_demo.cc` 的独立 C++ 演示程序，持续采集摄像头图像，将图像提交到 PPOCR 推理线程池，按提交顺序获取检测与识别结果，并使用 OpenCV 可视化文字框、识别文本和置信度。

PPOCR 的预处理、检测后处理、文字裁剪、识别后处理和 RKNN 调用继续复用当前 `find_signal` 中的实现；函数调用方式参考：

`/home/wtr/program/rk_toolkits/rknn_model_zoo/examples/PPOCR/PPOCR-System/cpp/main.cc`

## Requirements

- 运行模式是持续相机流，不是单张图片模式。
- 主线程负责摄像头采集和结果显示，推理线程负责 PPOCR 检测与识别。
- 输入图像进入 FIFO 线程池，结果按提交顺序显示。
- 每个 worker 持有独立的检测模型 context 和识别模型 context，避免多个线程并发使用同一个 RKNN context。
- 默认使用 camera 0、640x480 分辨率，并支持按 `q` 退出。
- 默认模型为：
  - `src/find_signal/models/ppocrv4_det.rknn`
  - `src/find_signal/models/ppocrv4_rec.rknn`
- 预处理和后处理调用现有接口：
  - `inference_ppocr_det_model`
  - `inference_ppocr_rec_model`
  - `inference_ppocr_system_model`
  - `convert_image`
  - `dbnet_postprocess`
  - `rec_postprocess`
- 可视化内容包括四边形文字框、识别文字和文字置信度；无法由 OpenCV 默认字体显示的 UTF-8 文本至少打印到终端，同时保留框和分数显示。
- 复用 `traffic_light` 已有的线程池语义和摄像头实现，不引入 ROS topic、service 或 node API。

## Non-goals

- 不重新实现 PPOCR 的 DB 检测、透视裁剪或 CTC 解码算法。
- 不把 Python 版的队列、模型容器或预处理代码移植到 C++。
- 不实现 zero-copy 摄像头输入。
- 不在本次任务中设计通用的模型配置文件或完整参数系统。
- 不修改现有 Python demo 的行为。
- 不为了线程池共享而引入新的 ROS 消息包或独立运行时服务。

## Current state and target worktree

目标工作树为：

`/home/wtr/program/iflycar/.worktrees/ppocr`

当前 `find_signal` 已有：

- `include/ppocr_system.h`
- `src/ppocr_system.cc`
- `src/postprocess.cc`
- `src/clipper.cc`
- PPOCR RKNN/ONNX 模型和字典文件

当前 `find_signal/CMakeLists.txt` 尚未配置 OpenCV、RKNN Runtime、RGA、PPOCR C++ 源文件和 demo target。

目标工作树中已经存在未跟踪的：

- `src/find_signal/include/camera_capture.h`
- `src/find_signal/src/camera_capture.cpp`

这两个文件与 `traffic_light` 的 `CameraCapture` 实现一致。实施时先审阅并复用，不覆盖或删除用户已有内容。

## Chosen approach

整体采用以下流水线：

```text
CameraCapture::captureFrame()
        |
        v
      BGR cv::Mat
        |
        | 复制到 worker 私有结果图像，并转换为 RGB
        v
  rknnPool<PPOCRModel, cv::Mat, cv::Mat>::put()
        |
        v
  PPOCRModel::infer()
        |
        +--> image_buffer_t(IMAGE_FORMAT_RGB888)
        |
        +--> inference_ppocr_system_model()
               |
               +--> 检测预处理 / RKNN det / DB 后处理
               |
               +--> 文字区域透视裁剪
               |
               +--> 识别预处理 / RKNN rec / CTC 后处理
        |
        v
  绘制四边形、文本、置信度
        |
        v
  主线程 cv::imshow()
```

`PPOCRModel::infer` 返回绘制后的 BGR `cv::Mat`，这样保持与现有 `traffic_light_demo.cc` 相同的 `rknnPool<Model, cv::Mat, cv::Mat>` 使用方式，主线程不需要理解 RKNN 输出结构。

## Local thread-pool headers

### Decision

本次 `find_signal` demo 直接使用当前路径下已经准备好的头文件：

- `src/find_signal/include/thread_pool.hpp`
- `src/find_signal/include/rknn_pool.hpp`

`find_signal` demo 仅依赖当前包内的这两份头文件；`traffic_light/include` 下现有的线程池文件保持不变。

保留现有头文件接口和 include 名称：

```cpp
#include "thread_pool.hpp"
#include "rknn_pool.hpp"
```

### Pool semantics

保留现有线程池的核心语义：

- `ThreadPool::submit` 返回 `std::future`。
- `rknnPool::put` 轮询选择 worker。
- future 保存在 FIFO 队列中。
- `rknnPool::get` 对队首 future 调用 `get()`，因此结果顺序与提交顺序一致。
- 析构时先消费未完成 future，再释放模型和线程池资源。

不在本次任务中加入丢帧策略、无界队列改造或新的调度器。通过“提交一帧、取回一帧”的流水线窗口控制待处理 future 数量，保持与 `traffic_light_demo.cc` 一致。

## PPOCR model adapter

新增模型适配类，建议文件为：

- `src/find_signal/include/ppocr_model.hpp`
- `src/find_signal/src/ppocr_model.cc`

建议接口：

```cpp
class PPOCRModel
{
public:
    PPOCRModel(const std::string& det_model_path,
               const std::string& rec_model_path);

    int init(rknn_context* shared_ctx, bool share_weight, int worker_id);
    rknn_context* get_pctx();
    cv::Mat infer(const cv::Mat& bgr_frame);

    ~PPOCRModel();
};
```

`rknnPool` 仍使用 `init(shared_ctx, share_weight, worker_id)` 形式，以便最大程度复用 `traffic_light` 的 pool contract。由于 PPOCR 是检测模型和识别模型的二模型组合，适配类内部同时管理：

```cpp
ppocr_system_app_context app_ctx_;
```

具体职责：

1. 保存检测模型路径、识别模型路径和 worker id。
2. 初始化自己的 det/rec RKNN context。
3. 为 det/rec context 设置 worker 对应的 NPU core mask。
4. 将摄像头 BGR 图像转换为 RGB 图像。
5. 组装不拥有数据的 `image_buffer_t`。
6. 使用固定的 DB 后处理参数调用 `inference_ppocr_system_model`。
7. 在 BGR 图像副本上绘制四边形、文字和分数。
8. 析构时释放 det/rec context。

### Context initialization

实现采用每个 worker 独立初始化 det/rec context 的方案：

1. 每个 worker 分别加载 det 和 rec 模型。
2. det/rec context 均调用 `rknn_set_core_mask` 绑定 worker 对应的 core。
3. 每个 worker 保存自己的输入输出 tensor 属性。
4. 第一版不使用 `rknn_dup_context`，避免二模型 context 复制关系不明确。

建议默认使用 2 个 worker，以匹配当前 Python PPOCR 流程中 det/rec 成对创建的两个 worker，并通过命令行或常量允许调整。若目标板实测内存和 NPU 吞吐允许，可使用 3 个 worker 对应 core 0、1、2。

### Core mapping

模型适配类提供 worker 到 core mask 的映射。默认三 worker 映射为：

```text
worker 0 -> RKNN_NPU_CORE_0
worker 1 -> RKNN_NPU_CORE_1
worker 2 -> RKNN_NPU_CORE_2
```

默认两 worker 时使用前两个 core。超出设备支持范围的 worker 数量应在初始化阶段报错，而不是静默使用 `AUTO`。

### Input color and lifetime

OpenCV 摄像头默认返回 BGR，而现有 PPOCR 接口和参考 `main.cc` 使用 RGB888。因此 worker 内必须执行：

```cpp
cv::cvtColor(bgr_frame, rgb_frame, cv::COLOR_BGR2RGB);
```

并设置：

- `format = IMAGE_FORMAT_RGB888`
- `width = rgb_frame.cols`
- `height = rgb_frame.rows`
- `virt_addr = rgb_frame.data`
- `size = rgb_frame.cols * rgb_frame.rows * 3`

`rgb_frame` 必须存活到 `inference_ppocr_system_model` 返回。模型接口内部会为检测输入和识别裁剪图分配自己的临时内存；worker 不释放 `rgb_frame.data`，由 `cv::Mat` 自动管理。

## Preprocess, inference and postprocess

demo 不直接操作 RKNN tensor，而是调用现有系统级接口：

```cpp
ppocr_det_postprocess_params params;
params.threshold = 0.3f;
params.box_threshold = 0.6f;
params.use_dilate = false;
params.db_score_mode = "slow";
params.db_box_type = "poly";
params.db_unclip_ratio = 1.5f;

inference_ppocr_system_model(
    &app_ctx_, &src_image, &params, &results);
```

内部调用链保持当前代码结构：

1. `inference_ppocr_det_model` 调用 `convert_image` 将输入缩放到检测模型尺寸。
2. RKNN 执行检测推理。
3. `dbnet_postprocess` 将概率图转换为四边形，并按原图比例映射坐标。
4. `GetRotateCropImage` 对每个文字框执行透视裁剪和必要旋转。
5. `inference_ppocr_rec_model` 将裁剪图缩放、归一化并补齐到识别模型输入尺寸。
6. `rec_postprocess` 执行 CTC 解码，得到文字和平均置信度。
7. `inference_ppocr_system_model` 返回文字框和识别结果数组。

在适配 demo 前，对现有 PPOCR 源码做最小安全修正：

- 将后处理参数中的字符串字段改为 `const char*` 或使用可写的本地字符串，避免 C++ 字符串常量赋给 `char*`。
- 推理失败路径释放已经分配的输入图像和 RKNN output。
- 对透视裁剪的 `cv::Rect` 做边界裁剪，避免摄像头流中的边界框造成越界异常。
- 不改变现有检测阈值、模型输入尺寸和后处理算法含义。

## Visualization

worker 在自己的 BGR 输出副本上绘制：

- 四边形四条边，颜色使用绿色。
- 识别文本和置信度，例如 `text@0.823`。
- 没有有效文字时仍绘制检测框，并在终端输出对应错误或低置信度信息。

主线程只调用 `cv::imshow` 和 `cv::waitKey(1)`，不直接访问 worker 的 RKNN context 或共享结果数组。

建议显示窗口名为 `PPOCR Camera`，按 `q` 退出。摄像头读取失败、模型初始化失败或推理 future 抛出异常时打印明确错误并进入安全退出流程。

## Demo command and defaults

新增 executable 名称建议为 `find_signal_demo`。

默认值：

```cpp
static const char* kDetModelPath =
    "src/find_signal/models/ppocrv4_det.rknn";
static const char* kRecModelPath =
    "src/find_signal/models/ppocrv4_rec.rknn";
static const int kCameraIndex = 0;
static const int kCameraWidth = 640;
static const int kCameraHeight = 480;
static const int kThreadCount = 2;
```

建议允许以下可选参数覆盖默认值：

```text
find_signal_demo [det_model] [rec_model] [camera_index] [thread_count]
```

不传参数时必须能够使用上述默认模型和摄像头配置启动。模型路径解析失败时在初始化阶段直接退出并打印路径。

## Build integration

修改 `src/find_signal/CMakeLists.txt`：

1. 增加 `find_package(OpenCV REQUIRED)`。
2. 设置与 `traffic_light` 相同的第三方路径：
   - `3rdparty/librknn`
   - `3rdparty/rga`
   - `3rdparty/stb_image`
   - `3rdparty/rknn_utils`
3. 在 include 路径中加入：
   - `src/find_signal/include`
   - RKNN、RGA、STB 和 rknn utils 目录
4. 增加 `find_signal_demo` target，源文件包括：
   - `src/find_signal/src/find_signal_demo.cc`
   - `src/find_signal/src/ppocr_visualization.cc`
   - `src/find_signal/src/ppocr_model.cc`
   - `src/find_signal/src/ppocr_system.cc`
   - `src/find_signal/src/postprocess.cc`
   - `src/find_signal/src/clipper.cc`
   - `src/find_signal/src/camera_capture.cpp`
   - `3rdparty/rknn_utils/image_utils.c`
   - `3rdparty/rknn_utils/file_utils.c`
5. 链接：
   - `${catkin_LIBRARIES}`
   - `${OpenCV_LIBS}`
   - `librknnrt.so`
   - `librga.so`
   - `pthread`
   - `dl`
   - `m`
不将 `image_drawing` 加入 demo，因为可视化直接使用 OpenCV；不加入任何 ROS node 源文件。

## Error handling and shutdown

- 任一 det/rec 模型初始化失败，整个 pool 初始化失败，并释放已经成功创建的 context。
- 任一 core mask 设置失败，初始化失败，不回退到未指定 core 的自动模式。
- 摄像头打开失败或读取空帧时停止采集。
- 按 `q` 后停止提交新帧，继续消费已提交 future，之后再销毁 pool 和 camera。
- worker 内部推理异常时返回未绘制的输入副本或明确失败结果，不能让主线程访问未初始化结果。
- 所有 context、tensor 属性、临时 `image_buffer_t` 和 RKNN output 都必须在对应 owner 的生命周期内释放。

## Verification

### Host-side checks

- 配置并编译 `find_signal_demo`。
- 检查 OpenCV、RKNN Runtime、RGA 和 pthread 链接是否完整。
- 使用不依赖真实 RKNN 的 fake model 验证当前包内 `rknnPool` 的 worker 初始化顺序和 FIFO 结果顺序。
- 检查 det/rec 的输入颜色格式和 `image_buffer_t` 生命周期。
- 使用静态检查或小型单元测试覆盖边界框裁剪和空结果路径。

### Target-board checks

- 观察每个 worker 的 det/rec context 初始化日志和 core mask。
- 确认摄像头窗口持续显示检测框、文字和置信度。
- 确认结果顺序稳定，没有明显跳帧或 future 队列无限增长。
- 对比参考单图 demo 和现有 Python demo，确认检测框、文字和置信度基本一致。
- 按 `q` 退出，确认所有 worker、RKNN context、摄像头和 OpenCV 窗口都能正常释放。

## File map

### Create

- `src/find_signal/include/ppocr_model.hpp` — PPOCR worker 适配器接口。
- `src/find_signal/src/ppocr_model.cc` — PPOCR context、颜色转换和推理。
- `src/find_signal/src/ppocr_visualization.cc` — OpenCV 结果框、文本和分数绘制。
- `src/find_signal/src/find_signal_demo.cc` — 摄像头、线程池和显示主循环。

### Use existing/prepared

- `src/find_signal/include/thread_pool.hpp` — 当前路径下的线程池实现。
- `src/find_signal/include/rknn_pool.hpp` — 当前路径下的 RKNN future 池实现。

### Modify

- `src/find_signal/CMakeLists.txt` — demo target、第三方依赖和当前包 include 路径。
- `src/find_signal/include/ppocr_system.h` — 必要的 const/初始化接口修正。
- `src/find_signal/src/ppocr_system.cc` — core mask 支持和失败路径资源清理。
- 必要时修改 `src/find_signal/src/postprocess.cc` — 仅做 demo 所需的边界和资源安全修正。
