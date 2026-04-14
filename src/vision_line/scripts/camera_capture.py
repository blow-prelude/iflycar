import logging

import cv2


class CameraConfig:
    INDEX = 0
    WIDTH = 640
    HEIGHT = 480


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


class CameraCapture:
    def __init__(self, index=None, width=None, height=None) -> None:
        """初始化摄像头

        Args:
            index: 摄像头索引，默认使用 CameraConfig.INDEX
            width: 分辨率宽度，默认使用 CameraConfig.WIDTH
            height: 分辨率高度，默认使用 CameraConfig.HEIGHT
        """
        idx = index if index is not None else CameraConfig.INDEX
        w = width if width is not None else CameraConfig.WIDTH
        h = height if height is not None else CameraConfig.HEIGHT

        self.cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)

        if not self.cap.isOpened():
            logging.error("Cannot open camera.")
            raise RuntimeError(f"Failed to open camera at index {idx}")

    def get_picture(self):
        """获取一帧图片并做预处理"""
        if self.cap is None or not self.cap.isOpened():
            logging.error("Camera is not opened.")
            return None
        ret, frame = self.cap.read()
        if not ret:
            logging.error("Failed to grab frame")
            return None
        return frame

    def is_opened(self):
        """检查摄像头是否已打开"""
        return self.cap is not None and self.cap.isOpened()

    def close(self):
        """关闭摄像头"""
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def __enter__(self):
        """上下文管理器入口"""
        return self

    def __exit__(self, _exc_type, _exc_val, _exc_tb):
        """上下文管理器出口，自动释放资源"""
        self.close()
        return False

    def __del__(self):
        """析构函数，对象销毁时自动释放资源"""
        self.close()
