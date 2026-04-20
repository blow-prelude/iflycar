import logging
import socket
import threading
import time
from typing import List, Optional, Tuple

import cv2
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


class ImageReceiver:
    def __init__(self, host, port) -> None:
        self.port = port
        self.host = host
        self.ser_socket = None
        self.cli_socket = None  # 保留但不使用
        self.connect_thread = None
        self.running = None
        self._is_closed = False
        self.num_images = None  # 添加图片数量
        self.windows = None  # 添加窗口名称列表

        self.receive_thread_t_sum = 0
        self.receive_thread_t_log = 0

        # 创建UDP socket并绑定
        self.ser_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.ser_socket.bind((self.host, self.port))
        # 设置socket超时，以便能够响应running_flag
        self.ser_socket.settimeout(0.1)  # 100ms超时
        logging.info(f"UDP server listening on {self.host}:{self.port}...")

    def receive_picture(self, num_images: int = 3) -> None:
        """接收多张图片，无需连接

        Args:
            num_images: 同时显示的图片窗口数量

        Raises:
            RuntimeError: 如果接收器已关闭或套接字未初始化
        """
        if self._is_closed:
            raise RuntimeError("Receiver is already closed.")

        if self.ser_socket is None:
            raise RuntimeError("Server socket is not initialized.")

        # 初始化窗口配置
        self.num_images = num_images
        self.windows = [f"receive_image_{i}" for i in range(num_images)]
        logging.info(f"Initialized {num_images} display windows")

        # 直接启动接收线程，无需accept
        self.running = [True]
        self.connect_thread = threading.Thread(
            target=self.receive_worker,
            args=(self.ser_socket, self.running),  # 传入socket而不是conn
        )
        self.connect_thread.daemon = True
        self.connect_thread.start()
        logging.info("Receive thread started.")

        # 主线程等待，直到用户按下 ESC 或接收线程结束
        while self.connect_thread.is_alive():
            self.connect_thread.join(timeout=0.1)

    def _cleanup_client(self):
        """清理客户端连接和线程资源"""
        if self.connect_thread is not None:
            self.connect_thread.join(timeout=2)
            self.connect_thread = None
        if self.cli_socket is not None:
            self.cli_socket.close()
            self.cli_socket = None

    def close(self):
        """显式关闭服务器，释放所有资源"""
        if self._is_closed:
            return

        self._is_closed = True

        # 停止接收线程
        if self.running is not None:
            self.running[0] = False

        # 清理客户端资源
        self._cleanup_client()

        # 关闭服务器套接字（UDP不需要shutdown）
        if self.ser_socket is not None:
            try:
                self.ser_socket.close()
                logging.info("Server closed.")
            except Exception as e:
                logging.warning(f"Error closing socket: {e}")
            self.ser_socket = None

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

    def handle_connect(
        self, sock: socket.socket
    ) -> Tuple[Optional[int], Optional[bytes]]:
        """接收UDP数据包

        Args:
            sock: UDP socket对象

        Returns:
            Tuple[img_id, img_data]: 图片ID和图片数据
            如果接收失败返回 (None, None)
        """
        # recvfrom返回数据和发送方地址
        try:
            data, _ = sock.recvfrom(65535)  # UDP最大包
        except socket.timeout:
            return None, None

        # 验证包头完整性
        if len(data) < 5:
            logging.warning("Packet too short, dropping")
            return None, None

        img_id = int.from_bytes(data[0:1], "big")
        size = int.from_bytes(data[1:5], "big")

        # 验证数据完整性
        if len(data) < 5 + size:
            logging.warning(f"Incomplete packet, expected {5 + size}, got {len(data)}")
            return None, None

        img_data = data[5 : 5 + size]

        # 验证图片ID范围
        if self.num_images is not None:
            if img_id < 0 or img_id >= self.num_images:
                logging.warning(
                    f"Invalid img_id {img_id}, expected 0-{self.num_images - 1}, dropping"
                )
                return None, None

        return img_id, img_data

    def process_image(self, data):
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
        if img is None or len(img) == 0:
            raise RuntimeError("Failed to decode image")
        return img

    def receive_worker(self, sock: socket.socket, running_flag: List[bool]) -> None:
        """接收并显示图片的线程函数"""
        cur_t, pre_t, t_sum = 0.0, 0.0, 0.0
        t_log = 0
        try:
            while running_flag[0]:
                try:
                    # 测速
                    cur_t = time.perf_counter()
                    t_sum += cur_t - pre_t
                    pre_t = cur_t
                    t_log += 1
                    if t_log % 10 == 0:
                        logging.info(
                            f"receive thread frequence:{1 / (t_sum / 10):.6f}Hz"
                        )
                        t_sum = 0.0
                        t_log = 0

                    img_id, img_byte = self.handle_connect(sock)

                    # 处理接收失败（包括超时）
                    if img_id is None or img_byte is None:
                        continue

                    # 验证图片ID范围
                    if self.num_images is None or self.windows is None:
                        logging.error("Receiver not initialized")
                        continue

                    logging.debug(f"Received image {img_id}")

                    img = self.process_image(img_byte)
                    window_name = self.windows[img_id]
                    cv2.imshow(window_name, img)

                    if cv2.waitKey(1) & 0xFF == 27:
                        logging.info("interrupt by user.")
                        break

                except socket.timeout:
                    # 超时是正常的，继续循环检查running_flag
                    continue

        except (ConnectionResetError, BrokenPipeError) as e:
            logging.error(f"Connection error: {e}")
        except RuntimeError as e:
            logging.error(f"Runtime error in receive_worker: {e}")
        except Exception as e:
            logging.error(f"Unexpected error in receive_worker: {e}")
        finally:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    host = "0.0.0.0"
    port = 12345

    try:
        with ImageReceiver(host, port) as img_rec:
            img_rec.receive_picture(num_images=3)  # 显示3个窗口
    except KeyboardInterrupt:
        logging.info("Interrupted by user.")
    except RuntimeError as e:
        logging.error(f"Runtime error: {e}")
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
