# Traffic-Light YOLOv8 Multi-NPU Demo Design

## Goal

在 `src/traffic_light` 中实现一个类似
`/home/wtr/program/rk_toolkits/rknn-cpp-Multithreading/src/main.cc` 的独立演示程序：持续读取相机视频，将帧按顺序提交给 3 个推理线程，每个线程的 RKNN context 固定绑定到一个 NPU 核心，完成 YOLOv8 RKNN 推理、绘框并显示结果。

## Requirements

- 运行方式是持续相机流，不是单帧模式。
- 固定创建 3 个推理线程，分别绑定 NPU core 0、1、2。
- 相机使用现有 `CameraCapture`，默认设备为 camera 0，分辨率为 640x480。
- 复用 `src/traffic_light` 中已有的官方 YOLOv8 RKNN 预处理、推理和后处理函数；不移植 YOLOv5 的检测逻辑。
- 模型路径写在代码中，不从命令行读取：
  `src/traffic_light/models/bestfp.rknn`。
- 使用 OpenCV 显示带检测框和类别/置信度的结果，按 `q` 退出。
- 采用参考工程的异步流水线语义：输入帧进入 FIFO，输出结果按提交顺序显示；程序退出前排空未完成任务。
- 现有 Python 实现中的类别顺序作为显示标签：`stop`、`straight`、`right`、`left`。

## Non-goals

- 不创建 ROS topic、service 或 node API；本次交付是与参考 `main.cc` 同类的独立演示程序。
- 不实现 zero-copy 版本；`yolov8_zero_copy.cc` 保留为现有实验代码，不参与本目标 executable，避免与普通实现产生重复符号。
- 不重新设计 YOLOv8 DFL、NMS 或 letterbox 算法。
- 不加入运行时模型选择、摄像头参数解析或通用配置系统。

## Chosen approach

直接移植参考工程的模型池/线程池结构，并为已有的 YOLOv8 C 风格接口增加薄适配层。

每个模型对象拥有一个独立的 `rknn_app_context_t`。第一个对象通过 `rknn_init` 加载模型，后续对象通过 `rknn_dup_context` 复用已加载的模型权重；每个 context 创建完成后调用 `rknn_set_core_mask`，显式设置为 core 0、1 或 2。这样保留参考工程的初始化方式，同时保留现有官方 YOLOv8 的输入输出属性查询和推理流程。

主线程只负责采集和显示。提交帧时不修改相机返回的 `cv::Mat`；工作线程将 BGR 帧转换为 RGB，调用现有 `inference_yolov8_model`，然后在自己的结果副本上绘制检测框。线程池的 future 队列保证显示顺序与参考程序一致。

## Architecture

### Model adapter

新增的 YOLOv8 模型适配类只负责以下职责：

1. 保存模型路径和 `rknn_app_context_t`。
2. 初始化 RKNN context，并根据传入的 worker id 绑定 NPU core。
3. 将 `cv::Mat` 转换为现有接口需要的 `image_buffer_t`。
4. 调用 `inference_yolov8_model` 获取 `object_detect_result_list`。
5. 将检测结果绘制到 BGR 图像副本并返回。
6. 释放 tensor 属性和 RKNN context。

模型适配类不复制官方的 YOLOv8 前处理或后处理实现；这些职责继续由 `yolov8.cc`、`image_utils.c` 和 `postprocess.cc` 承担。

### Thread pool

参考工程的 `rknnPool<model, input, output>` 结构直接移植到本包，并使用 3 个 worker。线程池初始化阶段依次创建 3 个模型实例并传入明确的 core id。`put` 将一帧提交到轮询分配的模型，`get` 从 future FIFO 中取得最早提交的结果。

主循环采用参考程序的流水线窗口：提交前 `threadNum` 个帧后，开始每轮提交一帧并取得一帧；退出采集后继续取得并显示队列中剩余结果。

### Camera and display

演示程序使用：

```cpp
static const char *kModelPath = "src/traffic_light/models/bestfp.rknn";
static const int kCameraIndex = 0;
static const int kCameraWidth = 640;
static const int kCameraHeight = 480;
```

创建 `CameraCapture(kCameraIndex, kCameraWidth, kCameraHeight)` 后持续调用 `captureFrame()`。空帧或相机读取失败时结束采集。显示窗口使用 OpenCV `imshow`，`waitKey(1)` 返回 `q` 时结束。

### YOLOv8 input path

相机输出按现有 `CameraCapture` 默认配置作为 BGR `cv::Mat` 处理。模型适配类执行 `cv::cvtColor(frame, rgb, cv::COLOR_BGR2RGB)`，创建不拥有数据的 `image_buffer_t`：

- `format = IMAGE_FORMAT_RGB888`
- `width`、`height` 使用 RGB 图像尺寸
- `virt_addr = rgb.data`
- `size = width * height * 3`

现有 `inference_yolov8_model` 继续负责模型尺寸 letterbox、RKNN input 设置、`rknn_run`、输出获取、输出释放和官方后处理。

### YOLOv8 output path

当前 `postprocess.cc` 是 RKNN Model Zoo 的 YOLOv8 后处理移植版，但其中 `OBJ_CLASS_NUM = 80` 是 COCO 默认值。自定义模型输出类别数不是 COCO 80 类，因此在不改变 DFL/NMS 算法的前提下做最小适配：

- `process_i8`、`process_u8`、`process_fp32` 和 RV1106 分支接收 `class_num` 参数。
- `class_num` 从对应 score output tensor 的 channel 维度读取，而不是使用固定 80。
- 三个检测分支都使用该动态类别数遍历 score tensor。
- `object_detect_result.cls_id` 继续保存模型输出的类别 id。
- 显示层使用 4 个现有 Python 标签，超出标签范围时显示类别 id，不能访问越界数组。
- 演示程序不调用依赖 `./model/coco_80_labels_list.txt` 的旧 `init_post_process`；类别名称由显示层的 4 类常量提供。

这样只修复自定义类别数与官方后处理之间的接口差异，不创作新的 YOLOv8 解码逻辑。

## RKNN context lifecycle

初始化按以下顺序执行：

1. worker 0 从 `kModelPath` 读取模型并调用 `rknn_init`。
2. 查询输入输出数量、输入输出 tensor 属性和模型输入尺寸。
3. 将 worker 0 的 context 传给后续 worker。
4. worker 1、2 调用 `rknn_dup_context` 创建独立 context。
5. 每个 worker 对自己的 context 调用 `rknn_set_core_mask`，分别设置 `RKNN_NPU_CORE_0`、`RKNN_NPU_CORE_1`、`RKNN_NPU_CORE_2`。
6. 每个 worker 保存自己的输入/输出属性副本，并在推理期间只使用自己的 RKNN context。
7. 释放时先释放每个 context 的属性，再调用对应的 `rknn_destroy`；线程池销毁前先消费全部 future。

任何一个 worker 初始化失败都使池初始化失败，并释放已经成功初始化的 context。单帧推理失败时记录带 worker id 的错误，并让该 future 返回未绘制的原始帧；主线程不访问未初始化的检测结果。

## Build integration

`src/traffic_light/CMakeLists.txt` 增加独立 executable 和必要依赖：

- OpenCV headers and libraries
- `3rdparty/librknn/include` and `librknnrt.so`
- `3rdparty/rga/include` and `librga.so`
- pthread
- 现有 `camera_capture.cpp`、`yolov8.cc`、`postprocess.cc`、`image_utils.c`、`file_utils.c` 等必要源文件

`yolov8_zero_copy.cc` 不加入该 target。CMake 不增加命令行模型参数，也不修改其它 ROS 包。

## Error handling and shutdown

- 模型文件读取失败、RKNN 初始化失败、core mask 设置失败、相机打开失败和推理失败都打印带 worker id 的错误信息。
- 相机读取失败或用户按 `q` 后停止提交新帧。
- 退出采集循环后排空 future 队列，保证 worker 完成并可安全析构。
- 结果绘制只在主线程或当前 worker 的私有结果图像上进行，不共享可变的 `cv::Mat` 或检测结果。

## Verification

在无法使用目标板 NPU 的主机环境中，先验证：

- CMake 能配置并编译目标（或明确记录缺少目标板 RKNN/OpenCV 依赖）。
- 线程池的 FIFO 提交/取回语义通过不依赖 RKNN 的测试模型验证。
- 自定义类别数后处理测试使用 synthetic tensor，确认 4 类 score 不会按 80 类越界，并正确返回类别 id。
- 检查三处 core mask 映射为 core 0、1、2。

在目标板上再验证：

- 三个 RKNN context 初始化日志分别显示对应 NPU core。
- 相机窗口持续显示带框结果，按 `q` 能正常退出。
- 使用 `bestfp.rknn` 的检测结果类别、框坐标和 Python 版本基本一致。
