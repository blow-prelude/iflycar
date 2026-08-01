# 参数调优 GUI 设计文档

**日期:** 2026-04-17
**作者:** Claude
**状态:** 草稿

## 概述

为视觉巡线处理系统创建一个基于 PyQt5 的参数调优 GUI 工具。该工具提供实时参数调整，并立即显示视觉反馈。

## 需求

- 左侧面板：图像显示区域（320x240）
- 右侧面板：参数滑动条及数值显示
- 拖动滑块时实时更新
- 使用 `CameraCapture` 类集成摄像头
- 通过名称/范围/默认值列表配置参数

## 架构

### 组件结构

```
MainWindow (QMainWindow)
├── Central Widget (QWidget)
│   └── Main Layout (QHBoxLayout)
│       ├── Left: 图像显示区域 (320x240)
│       │   └── QLabel with QPixmap
│       └── Right: 参数面板
│           └── QScrollArea
│               └── ParamSlider 组件列表
```

### 核心类

#### 1. `ParamSlider` (QWidget)

单个参数滑动条组件。

**属性:**
- `param_name`: str - 参数名称
- `min_val`: int - 最小值
- `max_val`: int - 最大值
- `default_val`: int - 默认值

**信号:**
- `valueChanged(str, int)` - 滑动条值改变时发出

**组件:**
- 名称标签 (QLabel)
- 滑动条 (QSlider)
- 数值显示 (QLabel)

#### 2. `ParamTunerWindow` (QMainWindow)

主应用窗口。

**属性:**
- `params`: dict - 当前参数值
- `source_image`: np.ndarray - 原始输入图像
- `process_callback`: Callable - 图像处理函数

**主要方法:**
- `update_source_image(img)` - 从摄像头更新输入图像
- `refresh_display()` - 使用当前参数重新处理并显示
- `on_param_changed(name, value)` - 处理滑动条变化

### 数据流

```
摄像头帧
    ↓
update_source_image()
    ↓
on_param_changed() [拖动滑块]
    ↓
process_callback(img, params)
    ↓
cv2_to_qimage()
    ↓
QLabel.setPixmap()
```

## 参数配置

使用三个列表定义参数：

```python
PARAM_NAMES = ["threshold", "dilate_iter", ...]
PARAM_RANGES = [(0, 255), (1, 10), ...]
PARAM_DEFAULTS = [180, 1, ...]
```

滑动条根据这些列表动态生成。

## 处理回调接口

用户提供回调函数：

```python
def process_callback(img: np.ndarray, params: dict) -> np.ndarray:
    """
    参数:
        img: 源图像 (BGR 格式)
        params: 当前参数值的字典
    返回:
        处理后的图像 (BGR 格式)
    """
    # 处理逻辑
    return processed_img
```

## 图像显示

### OpenCV 到 Qt 的转换

```python
def cv2_to_qimage(img: np.ndarray) -> QImage:
    height, width = img.shape[:2]
    bytes_per_line = 3 * width

    if len(img.shape) == 2:  # 灰度图
        return QImage(img.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
    else:  # BGR
        return QImage(img.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
```

### 显示属性

- 固定尺寸: 320x240
- 自动缩放: `setScaledContents(True)`
- 格式: BGR (OpenCV 默认) → RGB (Qt 显示)

## 主函数使用示例

```python
def main():
    # 打开摄像头
    cap = CameraCapture(index=0, width=320, height=240)

    # 定义参数
    PARAM_NAMES = ["threshold"]
    PARAM_RANGES = [(0, 255)]
    PARAM_DEFAULTS = [180]

    # 创建调参窗口
    tuner = ParamTunerWindow(
        param_names=PARAM_NAMES,
        param_ranges=PARAM_RANGES,
        param_defaults=PARAM_DEFAULTS,
        process_callback=process_callback
    )
    tuner.show()

    # 主循环
    while cap.is_opened():
        frame = cap.get_picture()
        tuner.update_source_image(frame)
        QApplication.processEvents()
```

## 测试示例

```python
def test_process_callback(img: np.ndarray, params: dict) -> np.ndarray:
    """测试回调：二值化处理"""
    threshold = params["threshold"]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
```

## 文件结构

```
src/vision_line/scripts/
├── tune_params.py          # 主要 GUI 实现
├── camera_capture.py       # 已存在：摄像头类
└── image_process.py        # 已存在：处理逻辑
```

## 依赖

- PyQt5
- OpenCV (cv2)
- NumPy
