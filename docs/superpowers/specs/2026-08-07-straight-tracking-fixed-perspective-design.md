# STRAIGHT_TRACKING 固定地面透视设计

## 目标

在 `find_way` 的 `STRAIGHT_TRACKING` 状态中，对 resize 到 320×240 的相机帧执行固定地面掩膜和透视变换，然后将变换结果交给现有预处理与寻线逻辑。其他状态继续处理 resize 后的原始帧。

## 数据流

所有非 `IDLE` 帧先沿用现有流程完成相机校正、水平翻转和 `ImageProcess::resize_frame`。resize 后的帧必须是 320×240，随后按状态选择处理帧：

- `STRAIGHT_TRACKING`：固定地面掩膜 → 固定透视变换 → `ImageProcess::preprocess` → 原有直行寻线逻辑。
- `RIGHT_TURNING`、`LEFT_TURNING`：直接使用 resize 后的帧 → `ImageProcess::preprocess` → 原有转弯寻线逻辑。

`ImageProcess::set_frame`、预处理、绘制和后续寻线统一使用状态选择后的处理帧，不改变其余算法调用顺序。

## 固定透视模块

复用 `src/vision_line/include/ground_perspective.h` 和 `src/vision_line/src/ground_perspective.cpp`，新增面向运行时的固定透视接口。`find_way.cpp` 只调用该接口，不保存标定参数，也不在每帧重新计算单应矩阵。

固定输入约束：

- 图像类型：非空 `CV_8UC3` BGR 图像。
- 图像尺寸：320×240。
- 地面起始行：`y = 139`，所有 `y < 139` 的像素在透视前清零。
- 输出尺寸：484×299。

根据 Python 参考实现中的源点、实际尺寸和像素比例离线计算并展开后的固定矩阵为：

```text
[-0.214974396135, -1.920230497309, 266.769032700576]
[ 0.006976740047, -2.918053000379, 403.289239923271]
[ 0.000068399412, -0.008376745778,   1.000000000000]
```

固定接口先构造 `y >= 139` 的掩膜并清除非地面区域，再使用 `cv::warpPerspective` 和上述固定矩阵生成 484×299 的鸟瞰图。

## 错误处理

固定接口对空图像、非 `CV_8UC3` 输入或非 320×240 尺寸抛出 `std::invalid_argument`。不回退到原始帧，因为在 `STRAIGHT_TRACKING` 中静默回退会使后续算法处理与标定坐标不一致的数据。

`find_way` 沿用最外层异常处理，输出错误并终止当前进程。固定常量经过测试后，正常相机流程不会触发这些输入错误。

## 构建集成

`find_way` 构建目标新增对 `ground_perspective` 库的链接。保留现有 `CameraCapture::perspectiveFrame`，本次不删除或重构它，以免影响仓库中可能存在的其他调用方。

## 测试与验收

使用 GTest 先增加失败测试，再实现固定接口：

1. 320×240 BGR 输入得到 484×299、类型不变的输出。
2. 只在被掩膜的上半区域放置非零像素时，输出保持全零，证明 `y < 139` 的像素不会进入透视结果。
3. 空图像、错误类型和错误尺寸均被拒绝。
4. 构建 `find_way`，证明头文件、链接关系和状态分支接线正确。
5. 运行完整 `test_ground_perspective` 测试集，确认现有动态计算接口没有回归。

编译和测试在 WSL2 的 `/home/wtr/program/iflycar` 工作空间中进行。
