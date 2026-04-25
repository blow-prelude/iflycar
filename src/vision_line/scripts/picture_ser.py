# picture_ser_pyqt5.py
import logging
import socket
import threading
import time
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PyQt5.QtCore import QObject, Qt, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QGridLayout,
    QLabel,
    QMainWindow,
    QStatusBar,
    QWidget,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


class UDPImageReceiver(QObject):
    """负责UDP接收和图片解码，通过信号通知"""

    # 信号定义
    image_received = pyqtSignal(int, QImage, str)  # img_id, 图片数据, 图像名
    stats_updated = pyqtSignal(float, int)  # fps, 包数

    def __init__(self, host: str, port: int, parent=None):
        super().__init__(parent)
        self.host = host
        self.port = port
        self.socket = None
        self.receive_thread = None
        self.running = [False]
        self.num_images = 0
        self._packet_count = 0
        self._fps = 0.0

        self._init_socket()

    def start(self, num_images: int):
        """启动接收线程

        Args:
            num_images: 预期接收的图片数量

        Raises:
            RuntimeError: 如果socket未初始化
        """
        if self.socket is None:
            raise RuntimeError("Socket is not initialized.")

        if self.receive_thread is not None and self.receive_thread.is_alive():
            raise RuntimeError("Receive thread is already running.")

        self.num_images = num_images
        self.running = [True]
        self._packet_count = 0
        self._fps = 0.0

        self.receive_thread = threading.Thread(
            target=self._receive_worker,
            args=(self.socket, self.running),
        )
        self.receive_thread.daemon = True
        self.receive_thread.start()
        logging.info("Receive thread started.")

    def _init_socket(self):
        """初始化UDP socket并绑定到指定地址"""
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.bind((self.host, self.port))
        self.socket.settimeout(0.1)
        logging.info(f"UDP server listening on {self.host}:{self.port}...")

    def _receive_worker(self, sock: socket.socket, running_flag: List[bool]) -> None:
        """接收并处理图片的工作线程

        Args:
            sock: UDP socket对象
            running_flag: 运行标志列表，running_flag[0]为True时继续运行
        """
        cur_t, pre_t, t_sum = 0.0, time.perf_counter(), 0.0
        t_log = 0
        try:
            while running_flag[0]:
                try:
                    img_id, img_name, img_byte = self.handle_connect(sock)

                    # 处理接收失败（包括超时）
                    if img_id is None or img_byte is None:
                        continue

                    self._packet_count += 1
                    logging.debug(f"Received image {img_id} name={img_name!r}")

                    # 测速：仅在实际收到图片时计数
                    cur_t = time.perf_counter()
                    t_sum += cur_t - pre_t
                    pre_t = cur_t
                    t_log += 1
                    if t_log % 10 == 0:
                        fps = 1 / (t_sum / 10) if t_sum > 0 else 0.0
                        self._fps = fps
                        self.stats_updated.emit(fps, self._packet_count)
                        t_sum = 0.0
                        t_log = 0

                    img = self.process_image(img_byte)
                    qimage = self._array_to_qimage(img)
                    self.image_received.emit(img_id, qimage, img_name)

                except socket.timeout:
                    # 超时是正常的，继续循环检查running_flag
                    continue
                except RuntimeError as e:
                    logging.error(f"Runtime error in receive_worker: {e}")
                    continue
        except Exception as e:
            logging.error(f"Unexpected error in receive_worker: {e}")

    def stop(self):
        """停止接收"""
        self.running[0] = False
        if self.receive_thread is not None:
            self.receive_thread.join(timeout=2)
            self.receive_thread = None

    def close(self):
        """关闭socket"""
        self.stop()
        if self.socket is not None:
            self.socket.close()
            self.socket = None
            logging.info("Socket closed successfully.")
        else:
            logging.warning("Socket already closed or not initialized.")

    def handle_connect(
        self, sock: socket.socket
    ) -> Tuple[Optional[int], Optional[str], Optional[bytes]]:
        """接收UDP数据包

        Args:
            sock: UDP socket对象

        Returns:
            Tuple[img_id, img_name, img_data]: 图片ID、图像名和图片数据
            如果接收失败返回 (None, None, None)
        """
        try:
            data, _ = sock.recvfrom(65535)
        except socket.timeout:
            return None, None, None

        # 验证包头完整性（13字节头部）
        if len(data) < 13:
            logging.warning("Packet too short, dropping")
            return None, None, None

        img_id = int.from_bytes(data[0:1], "big")
        name_raw = data[1:9]
        size = int.from_bytes(data[9:13], "big")

        null_idx = name_raw.find(b"\x00")
        img_name = (
            name_raw[:null_idx].decode("ascii")
            if null_idx != -1
            else name_raw.decode("ascii")
        )

        # 验证数据完整性
        if len(data) < 13 + size:
            logging.warning(f"Incomplete packet, expected {13 + size}, got {len(data)}")
            return None, None, None

        img_data = data[13 : 13 + size]

        # 验证图片ID范围
        if self.num_images > 0:
            if img_id < 0 or img_id >= self.num_images:
                logging.warning(
                    f"Invalid img_id {img_id}, expected 0-{self.num_images - 1}, dropping"
                )
                return None, None, None

        return img_id, img_name, img_data

    def process_image(self, data: bytes) -> np.ndarray:
        """解码图片数据

        Args:
            data: JPEG编码的图片数据

        Returns:
            解码后的图像 (numpy array)

        Raises:
            RuntimeError: 解码失败
        """
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
        if img is None or len(img) == 0:
            raise RuntimeError("Failed to decode image")
        return img

    @staticmethod
    def _array_to_qimage(img: np.ndarray) -> QImage:
        """将numpy数组转换为QImage

        Args:
            img: OpenCV图像 (BGR格式)

        Returns:
            QImage对象
        """
        # 处理不同通道数的图像
        if len(img.shape) == 2:  # 灰度图
            height, width = img.shape
            bytes_per_line = width
            qimage = QImage(
                img.data, width, height, bytes_per_line, QImage.Format_Grayscale8
            )
        elif len(img.shape) == 3:  # 彩色图
            height, width, channel = img.shape
            bytes_per_line = 3 * width
            # BGR转RGB
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            qimage = QImage(
                rgb_img.data, width, height, bytes_per_line, QImage.Format_RGB888
            )
        else:
            raise ValueError(f"Unsupported image shape: {img.shape}")

        return qimage.copy()  # 复制数据避免numpy数组被释放


class ImageDisplayWidget(QMainWindow):
    """图片显示窗口，使用网格布局显示多张图片"""

    def __init__(self, rows: int, cols: int, host: str, port: int, parent=None):
        super().__init__(parent)
        self.rows = rows
        self.cols = cols
        self.host = host
        self.port = port
        self.labels: list = []
        self._fps = 0.0
        self._packet_count = 0
        self._closed = False

        self._setup_ui()

    def _setup_ui(self):
        """创建网格布局的图片显示区域"""
        self.setWindowTitle(f"Image Display - {self.host}:{self.port}")

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QGridLayout(central_widget)
        layout.setSpacing(2)

        self.labels = []
        for r in range(self.rows):
            row_labels = []
            for c in range(self.cols):
                label = QLabel()
                label.setMinimumSize(640, 320)
                label.setAlignment(Qt.AlignCenter)
                label.setStyleSheet("QLabel { background-color: #1e1e1e; }")
                layout.addWidget(label, r, c)
                row_labels.append(label)
            self.labels.append(row_labels)

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Waiting for images...")

    def set_image(self, img_id: int, qimage: QImage, img_name: str = ""):
        """将图片设置到网格中指定位置

        Args:
            img_id: 图片索引，用于计算网格位置
            qimage: 要显示的QImage对象
            img_name: 图像名称，显示在图像左上角
        """
        row = img_id // self.cols
        col = img_id % self.cols

        if row < 0 or row >= self.rows or col < 0 or col >= self.cols:
            logging.warning(
                f"Image index {img_id} out of grid bounds ({self.rows}x{self.cols})"
            )
            return

        pixmap = QPixmap.fromImage(qimage)
        label = self.labels[row][col]
        scaled = pixmap.scaled(
            label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )

        # 在图像左上角绘制图像名
        if img_name:
            from PyQt5.QtGui import QColor, QFont, QPainter

            painter = QPainter(scaled)
            painter.setPen(QColor(255, 0, 255))
            painter.setFont(QFont("Monospace", 10))
            painter.drawText(5, 15, img_name)
            painter.end()

        label.setPixmap(scaled)

    def update_stats(self, fps: float, packet_count: int):
        """更新状态栏统计信息

        Args:
            fps: 当前帧率
            packet_count: 已接收的包数
        """
        self._fps = fps
        self._packet_count = packet_count
        self.status_bar.showMessage(
            f"FPS: {fps:.1f} | Packets: {packet_count} | Grid: {self.rows}x{self.cols}"
        )

    def closeEvent(self, event):
        """处理窗口关闭事件"""
        self._closed = True
        logging.info("ImageDisplayWidget closed.")
        event.accept()


class PyQt5ImageReceiver:
    """组合器：组合UDPImageReceiver和ImageDisplayWidget，提供简单的接收显示API"""

    def __init__(
        self, host: str = "0.0.0.0", port: int = 12345, rows: int = 1, cols: int = 3
    ):
        """初始化接收器

        Args:
            host: 监听地址
            port: 监听端口
            rows: 显示网格行数
            cols: 显示网格列数
        """
        self.host = host
        self.port = port
        self.rows = rows
        self.cols = cols
        self._is_closed = False

        # 创建QApplication（如果尚未存在）
        self.app = QApplication.instance()
        if self.app is None:
            self.app = QApplication([])

        # 创建UDP接收器
        self.receiver = UDPImageReceiver(host, port)

        # 创建显示窗口
        self.widget = ImageDisplayWidget(rows, cols, host, port)

        # 连接信号
        self.receiver.image_received.connect(self.widget.set_image)
        self.receiver.stats_updated.connect(self.widget.update_stats)

    def receive_picture(self, num_images: int = 6) -> None:
        """启动UDP接收器，显示窗口并运行Qt事件循环

        Args:
            num_images: 预期接收的图片数量
        """
        if self._is_closed:
            raise RuntimeError("Receiver is already closed.")

        # 启动UDP接收器
        self.receiver.start(num_images)

        # 显示窗口
        self.widget.show()

        logging.info(
            f"PyQt5ImageReceiver started, waiting for {num_images} images on {self.host}:{self.port}"
        )

        # 运行Qt事件循环
        self.app.exec_()

    def close(self) -> None:
        """停止接收器并关闭UI"""
        if self._is_closed:
            return

        self._is_closed = True

        # 停止UDP接收器
        self.receiver.close()

        # 关闭UI
        self.widget.close()

        logging.info("PyQt5ImageReceiver closed.")

    def __enter__(self):
        """上下文管理器入口，支持 with 语句"""
        return self

    def __exit__(self, _exc_type, _exc_val, _exc_tb):
        """上下文管理器出口，自动释放资源"""
        self.close()
        return False  # 不抑制异常

    def __del__(self):
        """析构函数，对象销毁时自动释放资源"""
        self.close()


if __name__ == "__main__":
    host = "0.0.0.0"
    port = 12345

    try:
        with PyQt5ImageReceiver(host, port, rows=1, cols=2) as img_rec:
            img_rec.receive_picture(num_images=6)
    except KeyboardInterrupt:
        logging.info("Interrupted by user.")
    except RuntimeError as e:
        logging.error(f"Runtime error: {e}")
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
