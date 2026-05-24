#!/usr/bin/env python3
"""
参数调优 GUI 工具
左侧显示图像，右侧提供参数滑动条进行实时调参
"""

import logging
import sys

import cv2
import numpy as np
from camera_capture import CameraCapture
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QScrollArea,
    QSlider,
    QVBoxLayout,
    QWidget,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


class ParamSlider(QWidget):
    """单个参数的滑动条组件"""

    # 信号：参数值改变时发出 (参数名, 新值)
    valueChanged = pyqtSignal(str, int)

    def __init__(
        self, param_name: str, min_val: int, max_val: int, default_val: int, parent=None
    ):
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


class ParamTunerWindow(QMainWindow):
    """参数调优主窗口"""

    def __init__(
        self,
        param_names: list,
        param_ranges: list,
        param_defaults: list,
        process_callback,
        parent=None,
    ):
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
        self.image_label.setStyleSheet(
            "border: 2px solid gray; background-color: black;"
        )
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
        for name, (min_val, max_val), default in zip(
            self.param_names, self.param_ranges, self.param_defaults
        ):
            slider = ParamSlider(name, min_val, max_val, default)
            slider.valueChanged.connect(self._on_param_changed)
            param_layout.addWidget(slider)
            self.param_sliders.append(slider)

        # 添加弹性空间
        param_layout.addStretch()

        scroll_area.setWidget(param_widget)
        main_layout.addWidget(scroll_area)

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
        return QImage(
            img.data, width, height, bytes_per_line, QImage.Format_RGB888
        ).rgbSwapped()


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
        process_callback=test_process_callback,
    )
    tuner.show()

    # 打开摄像头
    try:
        cap = CameraCapture(index=1, width=320, height=240)
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
