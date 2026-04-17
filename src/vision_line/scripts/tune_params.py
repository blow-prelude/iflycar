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
