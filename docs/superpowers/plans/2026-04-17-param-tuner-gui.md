# 参数调优 GUI 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**目标:** 创建一个基于 PyQt5 的参数调优工具，左侧显示图像（320x240），右侧提供参数滑动条，支持实时调参

**架构:** PyQt5 主窗口包含水平布局，左侧是 QLabel 图像显示区，右侧是 QScrollArea 包含的参数滑块列表。通过信号槽机制实现参数变化时触发处理回调并更新显示

**技术栈:** PyQt5, OpenCV (cv2), NumPy

---

## 文件结构

```
src/vision_line/scripts/
├── tune_params.py          # 创建：主要 GUI 实现
│   ├── cv2_to_qimage()     # OpenCV 到 Qt 图像转换
│   ├── ParamSlider         # 单个参数滑块组件
│   └── ParamTunerWindow    # 主窗口
├── camera_capture.py       # 已存在：使用其 CameraCapture 类
└── image_process.py        # 已存在：参考其 ImageProcess 类
```

---

### Task 1: 创建基础文件结构和导入

**Files:**
- Create: `src/vision_line/scripts/tune_params.py`

- [ ] **Step 1: 创建文件并添加导入**

```python
#!/usr/bin/env python3
"""
参数调优 GUI 工具
左侧显示图像，右侧提供参数滑动条进行实时调参
"""

import sys
import logging

import cv2
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, 
                              QHBoxLayout, QLabel, QSlider, 
                              QScrollArea, QVBoxLayout)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap

from camera_capture import CameraCapture

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
```

- [ ] **Step 2: 保存文件**

保存到 `src/vision_line/scripts/tune_params.py`

- [ ] **Step 3: 验证语法**

Run: `python -m py_compile src/vision_line/scripts/tune_params.py`
Expected: 无错误输出

---

### Task 2: 实现 cv2_to_qimage 转换函数

**Files:**
- Modify: `src/vision_line/scripts/tune_params.py`

- [ ] **Step 1: 在文件末尾添加转换函数**

```python
def cv2_to_qimage(img: np.ndarray) -> QImage:
    """将 OpenCV BGR 图像转换为 QImage
    
    Args:
        img: OpenCV 图像 (BGR 格式或灰度图)
        
    Returns:
        QImage: Qt 可显示的图像对象
    """
    if img is None:
        return None
        
    height, width = img.shape[:2]
    
    if len(img.shape) == 2:  # 灰度图
        bytes_per_line = width
        return QImage(img.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
    else:  # BGR 彩色图
        bytes_per_line = 3 * width
        return QImage(img.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
```

- [ ] **Step 2: 验证语法**

Run: `python -m py_compile src/vision_line/scripts/tune_params.py`
Expected: 无错误输出

---

### Task 3: 实现 ParamSlider 组件

**Files:**
- Modify: `src/vision_line/scripts/tune_params.py`

- [ ] **Step 1: 添加 ParamSlider 类**

在 `cv2_to_qimage` 函数之前添加：

```python
class ParamSlider(QWidget):
    """单个参数的滑动条组件"""
    
    # 信号：参数值改变时发出 (参数名, 新值)
    valueChanged = pyqtSignal(str, int)
    
    def __init__(self, param_name: str, min_val: int, max_val: int, default_val: int, parent=None):
        super().__init__(parent)
        self.param_name = param_name
        
        # 创建水平布局
        layout = QHBoxLayout()
        
        # 参数名称标签
        name_label = QLabel(param_name)
        name_label.setMinimumWidth(80)
        layout.addWidget(name_label)
        
        # 滑动条
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(min_val)
        self.slider.setMaximum(max_val)
        self.slider.setValue(default_val)
        self.slider.valueChanged.connect(self._on_value_changed)
        layout.addWidget(self.slider)
        
        # 数值显示标签
        self.value_label = QLabel(str(default_val))
        self.value_label.setMinimumWidth(40)
        layout.addWidget(self.value_label)
        
        self.setLayout(layout)
    
    def _on_value_changed(self, value: int):
        """滑动条值改变时的槽函数"""
        self.value_label.setText(str(value))
        self.valueChanged.emit(self.param_name, value)
    
    def value(self) -> int:
        """获取当前值"""
        return self.slider.value()
```

- [ ] **Step 2: 验证语法**

Run: `python -m py_compile src/vision_line/scripts/tune_params.py`
Expected: 无错误输出

---

### Task 4: 实现 ParamTunerWindow 主窗口（构造函数和UI）

**Files:**
- Modify: `src/vision_line/scripts/tune_params.py`

- [ ] **Step 1: 添加 ParamTunerWindow 类的构造函数和UI初始化**

在 `ParamSlider` 类之后添加：

```python
class ParamTunerWindow(QMainWindow):
    """参数调优主窗口"""
    
    def __init__(self, 
                 param_names: list, 
                 param_ranges: list, 
                 param_defaults: list,
                 process_callback,
                 parent=None):
        super().__init__(parent)
        
        self.param_names = param_names
        self.param_ranges = param_ranges
        self.param_defaults = param_defaults
        self.process_callback = process_callback
        
        # 当前参数值字典
        self.params = {}
        for name, default in zip(param_names, param_defaults):
            self.params[name] = default
        
        # 原始源图像
        self.source_image = None
        
        # 初始化UI
        self._init_ui()
    
    def _init_ui(self):
        """初始化用户界面"""
        self.setWindowTitle("参数调优工具")
        self.setGeometry(100, 100, 800, 400)
        
        # 创建中央部件和主布局
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        
        # 左侧：图像显示区域
        self.image_label = QLabel()
        self.image_label.setFixedSize(320, 240)
        self.image_label.setStyleSheet("border: 2px solid gray; background-color: black;")
        self.image_label.setScaledContents(True)
        self.image_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(self.image_label)
        
        # 右侧：参数面板
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setMinimumWidth(300)
        
        param_widget = QWidget()
        param_layout = QVBoxLayout(param_widget)
        
        # 创建参数滑块
        self.param_sliders = []
        for name, (min_val, max_val), default in zip(self.param_names, self.param_ranges, self.param_defaults):
            slider = ParamSlider(name, min_val, max_val, default)
            slider.valueChanged.connect(self._on_param_changed)
            param_layout.addWidget(slider)
            self.param_sliders.append(slider)
        
        # 添加弹性空间
        param_layout.addStretch()
        
        scroll_area.setWidget(param_widget)
        main_layout.addWidget(scroll_area)
```

- [ ] **Step 2: 验证语法**

Run: `python -m py_compile src/vision_line/scripts/tune_params.py`
Expected: 无错误输出

---

### Task 5: 实现 ParamTunerWindow 核心方法

**Files:**
- Modify: `src/vision_line/scripts/tune_params.py`

- [ ] **Step 1: 添加核心方法**

在 `ParamTunerWindow` 类的 `_init_ui` 方法之后添加：

```python
    def _on_param_changed(self, name: str, value: int):
        """参数改变时的槽函数"""
        self.params[name] = value
        self.refresh_display()
    
    def update_source_image(self, img: np.ndarray):
        """更新源图像
        
        Args:
            img: 新的源图像 (BGR 格式)
        """
        self.source_image = img
        self.refresh_display()
    
    def refresh_display(self):
        """使用当前参数重新处理并显示图像"""
        if self.source_image is None:
            return
        
        try:
            # 调用处理回调
            processed_img = self.process_callback(self.source_image, self.params)
            
            # 转换为 Qt 图像并显示
            qimage = cv2_to_qimage(processed_img)
            if qimage is not None:
                pixmap = QPixmap.fromImage(qimage)
                self.image_label.setPixmap(pixmap)
        except Exception as e:
            logging.error(f"图像处理出错: {e}")
```

- [ ] **Step 2: 验证语法**

Run: `python -m py_compile src/vision_line/scripts/tune_params.py`
Expected: 无错误输出

---

### Task 6: 添加测试用的处理回调和主函数

**Files:**
- Modify: `src/vision_line/scripts/tune_params.py`

- [ ] **Step 1: 添加测试回调和 main 函数**

```python
def test_process_callback(img: np.ndarray, params: dict) -> np.ndarray:
    """测试用的图像处理回调：二值化
    
    Args:
        img: 源图像 (BGR 格式)
        params: 参数字典，包含 "threshold"
        
    Returns:
        处理后的图像 (BGR 格式)
    """
    threshold = params.get("threshold", 180)
    
    # 转灰度
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 二值化
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    
    # 转回 BGR 用于显示
    return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)


def main():
    """主函数"""
    import sys
    
    # 定义参数（测试用：只包含二值化阈值）
    PARAM_NAMES = ["threshold"]
    PARAM_RANGES = [(0, 255)]
    PARAM_DEFAULTS = [180]
    
    # 创建 Qt 应用
    app = QApplication(sys.argv)
    
    # 创建调参窗口
    tuner = ParamTunerWindow(
        param_names=PARAM_NAMES,
        param_ranges=PARAM_RANGES,
        param_defaults=PARAM_DEFAULTS,
        process_callback=test_process_callback
    )
    tuner.show()
    
    # 打开摄像头
    try:
        cap = CameraCapture(index=0, width=320, height=240)
        logging.info("摄像头已打开")
        
        # 主循环
        while cap.is_opened():
            frame = cap.get_picture()
            if frame is not None:
                tuner.update_source_image(frame)
            
            # 处理 Qt 事件
            app.processEvents()
            
            # 检查窗口是否已关闭
            if not tuner.isVisible():
                break
                
    except Exception as e:
        logging.error(f"摄像头错误: {e}")
    finally:
        cap.close()
        logging.info("程序退出")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 验证完整语法**

Run: `python -m py_compile src/vision_line/scripts/tune_params.py`
Expected: 无错误输出

---

### Task 7: 测试运行

**Files:**
- Execute: `src/vision_line/scripts/tune_params.py`

- [ ] **Step 1: 运行程序**

Run: `python src/vision_line/scripts/tune_params.py`

- [ ] **Step 2: 验证行为**

Expected:
- 窗口打开，左侧有图像显示区域（320x240），右侧有一个 "threshold" 滑动条
- 摄像头图像显示在左侧
- 拖动滑动条时，图像实时更新（二值化效果变化）
- 关闭窗口时程序正常退出

---

## 使用说明

编写完成后，可以通过修改以下列表来添加更多参数：

```python
PARAM_NAMES = ["threshold", "dilate_iter", "blur_kernel"]
PARAM_RANGES = [(0, 255), (1, 10), (1, 15)]
PARAM_DEFAULTS = [180, 1, 3]
```

然后提供对应的 `process_callback` 函数处理这些参数。

## 依赖安装

如需安装 PyQt5：

```bash
pip install PyQt5
```
