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
